# File: src/pymatchit/matchers.py

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

class BaseMatcher(ABC):
    """
    Abstract Base Class for all matching algorithms.
    Enforces a consistent interface for the MatchIt core.
    """
    
    def __init__(self, ratio: int = 1, replace: bool = False, random_state: Optional[int] = None):
        self.ratio = ratio
        self.replace = replace
        self.random_state = random_state

    @abstractmethod
    def match(self, 
              treatment: pd.Series, 
              distance_measure: pd.Series, 
              **kwargs
              ) -> Tuple[Dict[int, List[int]], pd.Series]:
        """
        Execute the matching logic.

        Must return:
            matches: Dict[Treated_Index, List[Control_Indices]]
            weights: pd.Series (aligned to original data, containing weights)
        """
        pass

    def _build_result(self, matches: Dict[int, List[int]], all_indices: pd.Index) -> Tuple[Dict, pd.Series]:
        """
        Shared helper to construct the weights and final dataframe logic.
        """
        # Collect all Matched IDs
        matched_treated = []
        matched_control = []
        
        # Count frequency of controls (for weights in replacement mode)
        from collections import Counter
        control_counts = Counter()

        for t, c_list in matches.items():
            matched_treated.append(t)
            matched_control.extend(c_list)
            control_counts.update(c_list)

        # Create Weight Vector (Length of ORIGINAL data)
        weights = pd.Series(0.0, index=all_indices)
        
        # 1. Treated units get weight 1
        weights.loc[matched_treated] = 1.0
        
        # 2. Control units get weight proportional to usage
        # In standard ATT 1:k matching, control weights are often just their frequency.
        for c_idx, count in control_counts.items():
            weights.loc[c_idx] = count 
        
        return matches, weights


class NearestNeighborMatcher(BaseMatcher):
    """
    Implements Nearest Neighbor matching (Greedy).
    """

    def __init__(self, ratio: int = 1, replace: bool = False, caliper: Optional[float] = None, random_state: Optional[int] = None):
        super().__init__(ratio=ratio, replace=replace, random_state=random_state)
        self.caliper = caliper

    def match(self, 
              treatment: pd.Series, 
              distance_measure: pd.Series,
              **kwargs
              ) -> Tuple[Dict[int, List[int]], pd.Series]:
        
        # 1. Split Data
        treated_mask = treatment == 1
        control_mask = treatment == 0
        
        # Indices (Preserve original pandas indices)
        treated_indices = treatment[treated_mask].index.to_numpy()
        control_indices = treatment[control_mask].index.to_numpy()

        # Values (2D arrays for sklearn)
        # We assume distance_measure is already linear/logit if needed
        X_treated = distance_measure[treated_mask].values.reshape(-1, 1)
        X_control = distance_measure[control_mask].values.reshape(-1, 1)

        # Caliper calculation
        threshold = np.inf
        if self.caliper is not None:
            std_dev = distance_measure.std()
            threshold = self.caliper * std_dev

        # 2. Execute Matching Strategy
        if self.replace:
            matches = self._match_with_replacement(X_treated, X_control, treated_indices, control_indices, threshold)
        else:
            matches = self._match_without_replacement(X_treated, X_control, treated_indices, control_indices, threshold)

        # 3. Construct Matched DataFrame & Weights
        # Pass the full index from the treatment series to build the full weight vector
        return self._build_result(matches, treatment.index)

    def _match_with_replacement(self, X_treated, X_control, treated_indices, control_indices, threshold):
        """
        Fast vectorized matching using K-Nearest Neighbors.
        """
        nn = NearestNeighbors(n_neighbors=self.ratio, metric='euclidean', algorithm='auto')
        nn.fit(X_control)

        # Find k neighbors for ALL treated units at once
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

    def _match_without_replacement(self, X_treated, X_control, treated_indices, control_indices, threshold):
        """
        Greedy matching loop. 
        """
        # Sort Treated Units (Descending order of Distance Measure helps greedy matching)
        sort_order = np.argsort(X_treated.flatten())[::-1]
        
        available_control = set(range(len(X_control)))
        
        # Pre-fetch neighbors for efficiency
        nn = NearestNeighbors(n_neighbors=min(len(X_control), self.ratio * 10 + 20), metric='euclidean')
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
                if len(found) >= needed:
                    break
                
                if dist > threshold:
                    continue # Caliper violation
                
                if c_internal_idx in available_control:
                    found.append(control_indices[c_internal_idx])
                    available_control.remove(c_internal_idx)
            
            if len(found) == needed:
                matches[t_idx] = found
            elif len(found) > 0:
                # Keep partial matches
                matches[t_idx] = found

        return matches
    
class ExactMatcher(BaseMatcher):
    """
    Implements Exact Matching.
    Units are matched only if they have identical covariate values.
    """

    def match(self, 
              treatment: pd.Series, 
              covariates: pd.DataFrame, 
              **kwargs
              ) -> Tuple[Dict[int, List[int]], pd.Series]:
        
        # 1. Combine Treatment and Covariates temporarily
        work_data = covariates.copy()
        work_data['__treat__'] = treatment.values
        work_data['__original_index__'] = treatment.index

        # 2. Group by all covariates
        # We drop the temp columns from the grouping keys
        group_cols = list(covariates.columns)
        grouped = work_data.groupby(group_cols)

        matches = {}
        
        # 3. Iterate through strata
        for _, group in grouped:
            treated_in_group = group[group['__treat__'] == 1]
            control_in_group = group[group['__treat__'] == 0]

            # We need at least 1 treated and 1 control to match
            if not treated_in_group.empty and not control_in_group.empty:
                t_indices = treated_in_group['__original_index__'].tolist()
                c_indices = control_in_group['__original_index__'].tolist()
                
                # Link every Treated unit to ALL Control units in this exact stratum
                for t_idx in t_indices:
                    matches[t_idx] = c_indices

        # 4. Build Results
        # For Exact Matching, weights are slightly different than NN.
        # Treated = 1
        # Control = (Total Treated in Stratum) / (Total Control in Stratum)
        # However, our _build_result helper calculates weights based on usage count.
        # In Exact matching (all-to-all in stratum), a control is used by EVERY treated unit.
        # So frequency = N_treated. 
        # If we use the standard helper, Control Weight = N_treated.
        # This preserves the ATT estimand correctly (sum of weights = N_treated).
        
        return self._build_result(matches, treatment.index)