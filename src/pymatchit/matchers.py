# File: src/pymatchit/matchers.py

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Optional, Tuple

class NearestNeighborMatcher:
    """
    Implements Nearest Neighbor matching (Greedy).
    """

    def __init__(self, ratio: int = 1, replace: bool = False, caliper: Optional[float] = None, random_state: Optional[int] = None):
        self.ratio = ratio
        self.replace = replace
        self.caliper = caliper
        self.random_state = random_state

    def match(self, 
              treatment: pd.Series, 
              distance_measure: pd.Series
              ) -> Tuple[Dict[int, List[int]], pd.DataFrame]:
        """
        Performs the matching.

        Args:
            treatment (pd.Series): Binary treatment vector (0/1).
            distance_measure (pd.Series): The score to match on (e.g., Logit PS).

        Returns:
            matches (Dict): Mapping of Treated_Index -> List of Control_Indices.
            matched_data (pd.DataFrame): The resulting subset of data with weights.
        """
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
        return self._build_result(matches, treatment.index, treatment.index[control_mask])

    def _match_with_replacement(self, X_treated, X_control, treated_indices, control_indices, threshold):
        """
        Fast vectorized matching using K-Nearest Neighbors.
        """
        # algorithm='auto' is efficient (KDTree/BallTree)
        nn = NearestNeighbors(n_neighbors=self.ratio, metric='euclidean', algorithm='auto')
        nn.fit(X_control)

        # Find k neighbors for ALL treated units at once
        dists, neighbor_indices = nn.kneighbors(X_treated)

        matches = {}
        for i, t_idx in enumerate(treated_indices):
            # Check caliper
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
        R default: Sorts treated units by descending order of propensity score (usually),
        then matches 1-by-1.
        """
        # 1. Sort Treated Units (Descending order of Distance Measure helps greedy matching)
        # This reduces the chance of "stealing" a good match from a hard-to-match unit.
        sort_order = np.argsort(X_treated.flatten())[::-1]
        
        # Set of available control indices (internal 0..N indices)
        available_control = set(range(len(X_control)))
        
        # We need a quick lookup for controls. 
        # For small data, KDTree is fine, but we need to "delete" nodes. 
        # Simplest Greedy approach: Compute distances on the fly or use a mask.
        # For efficiency in Python, we can pre-calculate the KDTree, query *more* than k neighbors,
        # and pick the first one that is available.
        
        nn = NearestNeighbors(n_neighbors=min(len(X_control), self.ratio * 10 + 20), metric='euclidean')
        nn.fit(X_control)
        
        # Query ALL potential matches for ALL treated units first (vectorized)
        # This is faster than re-querying inside the loop.
        all_dists, all_neighbors = nn.kneighbors(X_treated)

        matches = {}
        
        for i in sort_order:
            t_idx = treated_indices[i]
            needed = self.ratio
            found = []

            # Look through the pre-fetched neighbors
            possible_neighbors = all_neighbors[i]
            possible_dists = all_dists[i]

            for dist, c_internal_idx in zip(possible_dists, possible_neighbors):
                if len(found) >= needed:
                    break
                
                # Check constraints
                if dist > threshold:
                    continue # Caliper violation
                
                if c_internal_idx in available_control:
                    # Match found!
                    found.append(control_indices[c_internal_idx])
                    available_control.remove(c_internal_idx)
            
            if len(found) == needed:
                matches[t_idx] = found
            elif len(found) > 0 and len(found) < needed:
                # Partial matches are usually discarded in 1:k exact matching
                # depending on policy. R's matchit usually requires full k matches 
                # or drops the unit if ratio is fixed. 
                # We will keep partials for now or drop? Let's keep for robustness.
                matches[t_idx] = found

        return matches

    def _build_result(self, matches, all_indices, control_pool_indices):
        """
        Constructs the weights and final dataframe.
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

        # Unique IDs for the dataframe
        final_ids = list(set(matched_treated + matched_control))
        
        # Create Weight Vector (Length of ORIGINAL data)
        weights = pd.Series(0.0, index=all_indices)
        
        # 1. Treated units get weight 1
        weights.loc[matched_treated] = 1.0
        
        # 2. Control units get weight proportional to usage
        for c_idx, count in control_counts.items():
            weights.loc[c_idx] = count  # Basic weight = frequency
            
            # Note: In 1:k matching, weights might need normalization 
            # (e.g., 1/k) depending on the estimand (ATT vs ATE).
            # For ATT (MatchIt default), weights are usually 1 for treated 
            # and (frequency) for control to represent the treated population.
        
        # Construct result table
        # We return the indices of rows that have weight > 0
        active_indices = weights[weights > 0].index
        
        return matches, weights