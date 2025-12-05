# File: src/pymatchit/matchers.py

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import pinv
from typing import Dict, List, Optional, Tuple, Any
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
              estimand: str = "ATT", # <--- NEW PARAMETER
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
            # Usually ATE requires full matching or matching both ways. 
            # We proceed but this is effectively ATT logic applied to the requested estimand.
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

        # 1. Identify Cutoffs
        # For ATT: Quantiles of Treated
        # For ATE: Quantiles of Total Population (usually) or Treated (often used as approximation)
        # R's MatchIt uses quantiles of the distance measure of the *treated* units by default for both, 
        # unless 'sub.by' is changed. We stick to treated quantiles for consistency.
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
                # Treated = N_total / N_treated
                # Control = N_total / N_control
                weights.loc[(treatment == 1) & in_bin] = n_total_bin / n_treated
                weights.loc[(treatment == 0) & in_bin] = n_total_bin / n_control

        return {}, weights