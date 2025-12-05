# File: src/pymatchit/matchers.py

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import pinv
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod

class BaseMatcher(ABC):
    """
    Abstract Base Class for all matching algorithms.
    """
    
    def __init__(self, ratio: int = 1, replace: bool = False, random_state: Optional[int] = None):
        self.ratio = ratio
        self.replace = replace
        self.random_state = random_state

    @abstractmethod
    def match(self, 
              treatment: pd.Series, 
              distance_measure: Optional[pd.Series] = None, 
              covariates: Optional[pd.DataFrame] = None,
              estimand: str = "ATT", 
              **kwargs
              ) -> Tuple[Dict[int, List[int]], pd.Series]:
        """
        Execute the matching logic.
        """
        pass

    def _build_result(self, matches: Dict[int, List[int]], all_indices: pd.Index) -> Tuple[Dict, pd.Series]:
        """
        Shared helper to construct the weights and final dataframe logic.
        """
        matched_treated = []
        matched_control = []
        
        from collections import Counter
        control_counts = Counter()

        for t, c_list in matches.items():
            matched_treated.append(t)
            matched_control.extend(c_list)
            control_counts.update(c_list)

        weights = pd.Series(0.0, index=all_indices)
        
        # 1. Treated units get weight 1
        weights.loc[matched_treated] = 1.0
        
        # 2. Control units get weight proportional to usage
        for c_idx, count in control_counts.items():
            weights.loc[c_idx] = count 
        
        return matches, weights


class NearestNeighborMatcher(BaseMatcher):
    """
    Implements Nearest Neighbor matching (Greedy).
    """

    def __init__(self, ratio: int = 1, replace: bool = False, caliper: Optional[float] = None, random_state: Optional[int] = None, mahalanobis: bool = False):
        super().__init__(ratio=ratio, replace=replace, random_state=random_state)
        self.caliper = caliper
        self.mahalanobis = mahalanobis

    def match(self, 
              treatment: pd.Series, 
              distance_measure: Optional[pd.Series] = None,
              covariates: Optional[pd.DataFrame] = None,
              estimand: str = "ATT",
              **kwargs
              ) -> Tuple[Dict[int, List[int]], pd.Series]:
        
        if estimand == "ATE":
            # Warning: Basic NN matching is asymmetric and typically estimates ATT.
            pass

        treated_mask = treatment == 1
        control_mask = treatment == 0
        treated_indices = treatment[treated_mask].index.to_numpy()
        control_indices = treatment[control_mask].index.to_numpy()

        if self.mahalanobis:
            if covariates is None:
                raise ValueError("Covariates required for Mahalanobis matching.")
            
            X_treated = covariates[treated_mask].values
            X_control = covariates[control_mask].values
            
            cov_matrix = covariates.cov()
            VI = pinv(cov_matrix.values)
            
            metric = 'mahalanobis'
            metric_params = {'VI': VI}
            threshold = np.inf 

        else:
            if distance_measure is None:
                raise ValueError("Distance measure required for nearest neighbor matching.")
            
            X_treated = distance_measure[treated_mask].values.reshape(-1, 1)
            X_control = distance_measure[control_mask].values.reshape(-1, 1)
            
            metric = 'euclidean'
            metric_params = {}
            
            threshold = np.inf
            if self.caliper is not None:
                std_dev = distance_measure.std()
                threshold = self.caliper * std_dev

        if self.replace:
            matches = self._match_with_replacement(
                X_treated, X_control, treated_indices, control_indices, 
                threshold, metric, metric_params
            )
        else:
            matches = self._match_without_replacement(
                X_treated, X_control, treated_indices, control_indices, 
                threshold, metric, metric_params
            )

        return self._build_result(matches, treatment.index)

    def _match_with_replacement(self, X_treated, X_control, treated_indices, control_indices, threshold, metric, metric_params):
        nn = NearestNeighbors(n_neighbors=self.ratio, metric=metric, metric_params=metric_params, algorithm='auto')
        nn.fit(X_control)
        dists, neighbor_indices = nn.kneighbors(X_treated)

        matches = {}
        for i, t_idx in enumerate(treated_indices):
            valid_neighbors = []
            for j in range(self.ratio):
                dist = dists[i, j]
                c_internal_idx = neighbor_indices[i, j]
                if dist <= threshold:
                    valid_neighbors.append(control_indices[c_internal_idx])
            if valid_neighbors:
                matches[t_idx] = valid_neighbors
        return matches

    def _match_without_replacement(self, X_treated, X_control, treated_indices, control_indices, threshold, metric, metric_params):
        if X_treated.shape[1] == 1:
            sort_order = np.argsort(X_treated.flatten())[::-1]
        else:
            sort_order = np.arange(len(X_treated))

        available_control = set(range(len(X_control)))
        n_fetch = min(len(X_control), self.ratio * 10 + 20)
        
        nn = NearestNeighbors(n_neighbors=n_fetch, metric=metric, metric_params=metric_params)
        nn.fit(X_control)
        all_dists, all_neighbors = nn.kneighbors(X_treated)

        matches = {}
        for i in sort_order:
            t_idx = treated_indices[i]
            needed = self.ratio
            found = []
            possible_neighbors = all_neighbors[i]
            possible_dists = all_dists[i]

            for dist, c_internal_idx in zip(possible_dists, possible_neighbors):
                if len(found) >= needed: break
                if dist > threshold: continue 
                
                if c_internal_idx in available_control:
                    found.append(control_indices[c_internal_idx])
                    available_control.remove(c_internal_idx)
            
            if len(found) > 0:
                matches[t_idx] = found

        return matches


class ExactMatcher(BaseMatcher):
    """
    Implements Exact Matching.
    """
    def match(self, treatment: pd.Series, covariates: pd.DataFrame, estimand: str = "ATT", **kwargs) -> Tuple[Dict[int, List[int]], pd.Series]:
        if covariates is None:
            raise ValueError("Covariates are required for Exact Matching.")

        work_data = covariates.copy()
        work_data['__treat__'] = treatment.values
        work_data['__original_index__'] = treatment.index

        grouped = work_data.groupby(list(covariates.columns))
        matches = {}
        
        for _, group in grouped:
            treated_in_group = group[group['__treat__'] == 1]
            control_in_group = group[group['__treat__'] == 0]

            if not treated_in_group.empty and not control_in_group.empty:
                t_indices = treated_in_group['__original_index__'].tolist()
                c_indices = control_in_group['__original_index__'].tolist()
                for t_idx in t_indices:
                    matches[t_idx] = c_indices

        return self._build_result(matches, treatment.index)


class SubclassMatcher(BaseMatcher):
    """
    Implements Subclassification (Stratification).
    Supports ATT and ATE.
    """
    def __init__(self, n_subclasses: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.n_subclasses = n_subclasses

    def match(self, 
              treatment: pd.Series, 
              distance_measure: pd.Series, 
              estimand: str = "ATT",
              **kwargs
              ) -> Tuple[Dict[int, List[int]], pd.Series]:
        
        if distance_measure is None:
            raise ValueError("Propensity Scores (distance_measure) required for Subclassification.")

        # 1. Identify Cutoffs (Quantiles of Treated)
        treated_scores = distance_measure[treatment == 1]
        
        _, bins = pd.qcut(treated_scores, q=self.n_subclasses, retbins=True, duplicates='drop')
        bins[0] = -np.inf
        bins[-1] = np.inf

        # 2. Assign ALL units to subclasses
        subclass_labels = pd.cut(distance_measure, bins=bins, labels=False, include_lowest=True)
        
        # 3. Calculate Weights
        weights = pd.Series(0.0, index=treatment.index)
        unique_bins = np.unique(subclass_labels.dropna())
        
        for bin_idx in unique_bins:
            in_bin = (subclass_labels == bin_idx)
            
            n_treated = np.sum((treatment == 1) & in_bin)
            n_control = np.sum((treatment == 0) & in_bin)
            n_total_bin = n_treated + n_control
            
            if n_treated == 0 or n_control == 0:
                continue
            
            if estimand == "ATT":
                # Treated = 1
                # Control = N_t / N_c
                weights.loc[(treatment == 1) & in_bin] = 1.0
                weights.loc[(treatment == 0) & in_bin] = n_treated / n_control
            
            elif estimand == "ATE":
                # Weight units to represent the full bin size
                weights.loc[(treatment == 1) & in_bin] = n_total_bin / n_treated
                weights.loc[(treatment == 0) & in_bin] = n_total_bin / n_control

        return {}, weights


class CEMMatcher(BaseMatcher):
    """
    Implements Coarsened Exact Matching (CEM).
    1. Coarsens (bins) continuous covariates.
    2. Performs Exact Matching on the coarsened data.
    """
    def __init__(self, cutpoints: Optional[Dict[str, Union[int, List[float]]]] = None, **kwargs):
        """
        Args:
            cutpoints: Dict mapping column names to number of bins (int) or list of cutpoints.
                       Example: {'age': 10, 'income': [0, 50000, 100000]}
                       If None, defaults to automatic binning (e.g., 5 bins).
        """
        super().__init__(**kwargs)
        self.cutpoints = cutpoints

    def match(self, 
              treatment: pd.Series, 
              covariates: pd.DataFrame, 
              estimand: str = "ATT",
              **kwargs
              ) -> Tuple[Dict[int, List[int]], pd.Series]:
        
        if covariates is None:
            raise ValueError("Covariates required for CEM.")

        # 1. Coarsen the Data
        # We work on a copy to avoid modifying the original dataframe passed in
        coarsened = covariates.copy()
        
        # Detect numeric columns for binning
        numeric_cols = coarsened.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Skip binary/dummy variables (0/1) - no need to bin if low cardinality
            if coarsened[col].nunique() <= 2:
                continue
                
            # Determine how to cut
            if self.cutpoints and col in self.cutpoints:
                cuts = self.cutpoints[col]
            else:
                # Default heuristic: 5 bins
                cuts = 5 
            
            # Apply binning
            # We convert to labels=False (integers) which act as categorical codes
            try:
                coarsened[col] = pd.cut(coarsened[col], bins=cuts, labels=False, include_lowest=True)
            except ValueError:
                # Fallback if cut fails (e.g. constant value)
                pass

        # 2. Perform Exact Matching on Coarsened Data
        # Logic largely mirrors ExactMatcher but calculates CEM-specific weights
        
        work_data = coarsened.copy()
        work_data['__treat__'] = treatment.values
        work_data['__original_index__'] = treatment.index
        
        # Group by ALL covariates (which are now bins/categories)
        grouped = work_data.groupby(list(coarsened.columns))
        
        matches = {}
        weights = pd.Series(0.0, index=treatment.index)
        
        for _, group in grouped:
            treated_in_group = group[group['__treat__'] == 1]
            control_in_group = group[group['__treat__'] == 0]
            
            n_treat = len(treated_in_group)
            n_control = len(control_in_group)
            
            # We need both groups to form a match
            if n_treat > 0 and n_control > 0:
                t_indices = treated_in_group['__original_index__'].tolist()
                c_indices = control_in_group['__original_index__'].tolist()
                
                # Register matches (All-to-All in stratum)
                for t_idx in t_indices:
                    matches[t_idx] = c_indices
                
                # CEM Weights
                # For ATT:
                #   W_treated = 1
                #   W_control = (N_treat_stratum / N_control_stratum)
                #   (Note: We use the simplified stratum-scaling which balances the weighted counts)
                
                if estimand == "ATT":
                    weights.loc[treated_in_group['__original_index__']] = 1.0
                    w_c = n_treat / n_control
                    weights.loc[control_in_group['__original_index__']] = w_c
                    
                elif estimand == "ATE":
                     # ATE: Weight up to total bin size
                     n_total = n_treat + n_control
                     weights.loc[treated_in_group['__original_index__']] = n_total / n_treat
                     weights.loc[control_in_group['__original_index__']] = n_total / n_control

        return matches, weights