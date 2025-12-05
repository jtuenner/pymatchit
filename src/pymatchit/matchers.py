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
              exact: Optional[pd.DataFrame] = None, # <--- NEW PARAMETER
              **kwargs
              ) -> Tuple[Dict[int, List[int]], pd.Series]:
        pass

    def _build_result(self, matches: Dict[int, List[int]], all_indices: pd.Index) -> Tuple[Dict, pd.Series]:
        """
        Helper to construct basic frequency weights (Count).
        """
        matched_treated = []
        
        from collections import Counter
        control_counts = Counter()

        for t, c_list in matches.items():
            matched_treated.append(t)
            control_counts.update(c_list)

        weights = pd.Series(0.0, index=all_indices)
        weights.loc[matched_treated] = 1.0
        
        for c_idx, count in control_counts.items():
            weights.loc[c_idx] = count 
        
        return matches, weights


class NearestNeighborMatcher(BaseMatcher):
    """
    Implements Nearest Neighbor matching (Greedy).
    Supports exact matching constraints via the 'exact' argument.
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
              exact: Optional[pd.DataFrame] = None,
              **kwargs
              ) -> Tuple[Dict[int, List[int]], pd.Series]:
        
        # --- NEW LOGIC: Stratified Matching (Exact Constraints) ---
        if exact is not None:
            # We must iterate through strata and match locally
            return self._match_stratified(
                treatment, distance_measure, covariates, estimand, exact
            )

        # ... (Existing logic for global matching continues below) ...
        return self._match_global(treatment, distance_measure, covariates, estimand)

    def _match_stratified(self, treatment, distance_measure, covariates, estimand, exact_df):
        """
        Helper to perform NN matching within exact strata.
        """
        # Combine data for grouping
        # We need to preserve indices carefully
        stratification_data = exact_df.copy()
        
        # We need a list of column names to group by
        group_cols = list(exact_df.columns)
        grouped = stratification_data.groupby(group_cols)
        
        all_matches = {}
        
        for _, group_indices in grouped.groups.items():
            # Get the indices for this group
            # group_indices is an Index object from the groupby
            
            # Extract subset of data
            # using .loc ensures we get the data for just this stratum
            local_treat = treatment.loc[group_indices]
            
            # Skip if we don't have both Treated and Control in this stratum
            if local_treat.sum() == 0 or (local_treat == 0).sum() == 0:
                continue
                
            local_dist = distance_measure.loc[group_indices] if distance_measure is not None else None
            local_covs = covariates.loc[group_indices] if covariates is not None else None
            
            # Recursively call global matching on this subset
            # We rely on _match_global to handle the 1:k / caliper / mahalanobis logic
            matches, _ = self._match_global(
                local_treat, local_dist, local_covs, estimand
            )
            
            # Merge results
            all_matches.update(matches)
            
        # Re-build weights globally from the combined matches
        # This handles the ratio/estimand logic correctly across all strata
        matches_dict, weights = self._build_result(all_matches, treatment.index)
        
        if estimand == "ATT" and self.ratio > 1:
            control_mask = (treatment == 0)
            weights.loc[control_mask] = weights.loc[control_mask] / self.ratio
            
        return matches_dict, weights

    def _match_global(self, treatment, distance_measure, covariates, estimand):
        """
        The core NN logic (Refactored from previous 'match' method).
        """
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
            
            # Compute Covariance on LOCAL data or GLOBAL? 
            # Standard practice: Mahalanobis metric is defined globally or per stratum?
            # If we are inside _match_global called by stratified, we are using local covariance.
            # This is generally preferred for exact matching (distance relative to stratum).
            
            # Handle singular matrices in small strata
            try:
                cov_matrix = covariates.cov()
                VI = pinv(cov_matrix.values)
            except:
                # Fallback identity if covariance fails (e.g. 1 observation)
                VI = np.eye(covariates.shape[1])
                
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
            matches = self._match_with_replacement(X_treated, X_control, treated_indices, control_indices, threshold, metric, metric_params)
        else:
            matches = self._match_without_replacement(X_treated, X_control, treated_indices, control_indices, threshold, metric, metric_params)

        matches_dict, weights = self._build_result(matches, treatment.index)
        
        if estimand == "ATT" and self.ratio > 1:
            weights.loc[control_mask] = weights.loc[control_mask] / self.ratio
            
        return matches_dict, weights

    def _match_with_replacement(self, X_treated, X_control, treated_indices, control_indices, threshold, metric, metric_params):
        if len(X_control) == 0: return {}
        nn = NearestNeighbors(n_neighbors=min(len(X_control), self.ratio), metric=metric, metric_params=metric_params, algorithm='auto')
        nn.fit(X_control)
        dists, neighbor_indices = nn.kneighbors(X_treated)
        matches = {}
        for i, t_idx in enumerate(treated_indices):
            valid_neighbors = []
            for j in range(dists.shape[1]): # Iterate up to k found
                if dists[i, j] <= threshold:
                    valid_neighbors.append(control_indices[neighbor_indices[i, j]])
            if len(valid_neighbors) > 0:
                matches[t_idx] = valid_neighbors
        return matches

    def _match_without_replacement(self, X_treated, X_control, treated_indices, control_indices, threshold, metric, metric_params):
        if len(X_control) == 0: return {}
        
        if X_treated.shape[1] == 1:
            sort_order = np.argsort(X_treated.flatten())[::-1]
        else:
            sort_order = np.arange(len(X_treated))

        available_control = set(range(len(X_control)))
        # Safety: don't ask for more neighbors than exist
        n_fetch = min(len(X_control), self.ratio * 10 + 20)
        
        nn = NearestNeighbors(n_neighbors=n_fetch, metric=metric, metric_params=metric_params)
        nn.fit(X_control)
        all_dists, all_neighbors = nn.kneighbors(X_treated)

        matches = {}
        for i in sort_order:
            t_idx = treated_indices[i]
            needed = self.ratio
            found = []
            for dist, c_internal_idx in zip(all_dists[i], all_neighbors[i]):
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
        # Exact matcher doesn't take 'exact' arg because it IS exact matching.
        # But for compatibility with BaseMatcher signature we accept **kwargs.
        if covariates is None:
            raise ValueError("Covariates are required for Exact Matching.")

        work_data = covariates.copy()
        work_data['__treat__'] = treatment.values
        work_data['__original_index__'] = treatment.index

        grouped = work_data.groupby(list(covariates.columns))
        matches = {}
        weights = pd.Series(0.0, index=treatment.index)
        
        for _, group in grouped:
            treated_in_group = group[group['__treat__'] == 1]
            control_in_group = group[group['__treat__'] == 0]
            
            n_treat = len(treated_in_group)
            n_control = len(control_in_group)

            if n_treat > 0 and n_control > 0:
                t_indices = treated_in_group['__original_index__'].tolist()
                c_indices = control_in_group['__original_index__'].tolist()
                for t_idx in t_indices:
                    matches[t_idx] = c_indices
                
                if estimand == "ATT":
                    weights.loc[treated_in_group['__original_index__']] = 1.0
                    weights.loc[control_in_group['__original_index__']] = n_treat / n_control
                elif estimand == "ATE":
                    n_total = n_treat + n_control
                    weights.loc[treated_in_group['__original_index__']] = n_total / n_treat
                    weights.loc[control_in_group['__original_index__']] = n_total / n_control

        return matches, weights


class SubclassMatcher(BaseMatcher):
    """
    Implements Subclassification.
    """
    def __init__(self, n_subclasses: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.n_subclasses = n_subclasses

    def match(self, treatment: pd.Series, distance_measure: pd.Series, estimand: str = "ATT", **kwargs) -> Tuple[Dict[int, List[int]], pd.Series]:
        if distance_measure is None:
            raise ValueError("Propensity Scores required for Subclassification.")
        
        treated_scores = distance_measure[treatment == 1]
        _, bins = pd.qcut(treated_scores, q=self.n_subclasses, retbins=True, duplicates='drop')
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        subclass_labels = pd.cut(distance_measure, bins=bins, labels=False, include_lowest=True)
        weights = pd.Series(0.0, index=treatment.index)
        unique_bins = np.unique(subclass_labels.dropna())
        
        for bin_idx in unique_bins:
            in_bin = (subclass_labels == bin_idx)
            n_treated = np.sum((treatment == 1) & in_bin)
            n_control = np.sum((treatment == 0) & in_bin)
            
            if n_treated == 0 or n_control == 0: continue
            
            if estimand == "ATT":
                weights.loc[(treatment == 1) & in_bin] = 1.0
                weights.loc[(treatment == 0) & in_bin] = n_treated / n_control
            elif estimand == "ATE":
                n_total = n_treated + n_control
                weights.loc[(treatment == 1) & in_bin] = n_total / n_treated
                weights.loc[(treatment == 0) & in_bin] = n_total / n_control

        return {}, weights


class CEMMatcher(BaseMatcher):
    """
    Implements Coarsened Exact Matching (CEM).
    """
    def __init__(self, cutpoints: Optional[Dict[str, Union[int, List[float]]]] = None, **kwargs):
        super().__init__(**kwargs)
        self.cutpoints = cutpoints

    def match(self, treatment: pd.Series, covariates: pd.DataFrame, estimand: str = "ATT", **kwargs) -> Tuple[Dict[int, List[int]], pd.Series]:
        if covariates is None: raise ValueError("Covariates required for CEM.")
        coarsened = covariates.copy()
        numeric_cols = coarsened.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if coarsened[col].nunique() <= 2: continue
            cuts = self.cutpoints[col] if (self.cutpoints and col in self.cutpoints) else 5
            try:
                coarsened[col] = pd.cut(coarsened[col], bins=cuts, labels=False, include_lowest=True)
            except ValueError: pass

        work_data = coarsened.copy()
        work_data['__treat__'] = treatment.values
        work_data['__original_index__'] = treatment.index
        grouped = work_data.groupby(list(coarsened.columns))
        
        matches = {}
        weights = pd.Series(0.0, index=treatment.index)
        
        for _, group in grouped:
            treated_in_group = group[group['__treat__'] == 1]
            control_in_group = group[group['__treat__'] == 0]
            n_treat = len(treated_in_group)
            n_control = len(control_in_group)
            
            if n_treat > 0 and n_control > 0:
                t_indices = treated_in_group['__original_index__'].tolist()
                c_indices = control_in_group['__original_index__'].tolist()
                for t_idx in t_indices:
                    matches[t_idx] = c_indices
                
                if estimand == "ATT":
                    weights.loc[treated_in_group['__original_index__']] = 1.0
                    weights.loc[control_in_group['__original_index__']] = n_treat / n_control
                elif estimand == "ATE":
                     n_total = n_treat + n_control
                     weights.loc[treated_in_group['__original_index__']] = n_total / n_treat
                     weights.loc[control_in_group['__original_index__']] = n_total / n_control

        return matches, weights