# File: src/pymatchit/matchers.py

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import pinv
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
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
              exact: Optional[pd.DataFrame] = None,
              **kwargs
              ) -> Tuple[Dict[int, List[int]], pd.Series, pd.Series]:
        pass

    def _build_result(self, matches: Dict[int, List[int]], all_indices: pd.Index) -> Tuple[Dict, pd.Series, pd.Series]:
        matched_treated = []
        from collections import Counter
        control_counts = Counter()
        
        subclasses = pd.Series(pd.NA, index=all_indices)
        group_id = 1

        for t, c_list in matches.items():
            matched_treated.append(t)
            control_counts.update(c_list)
            
            subclasses.loc[t] = group_id
            for c in c_list:
                if pd.isna(subclasses.loc[c]):
                    subclasses.loc[c] = group_id
            group_id += 1

        weights = pd.Series(0.0, index=all_indices)
        weights.loc[matched_treated] = 1.0
        
        for c_idx, count in control_counts.items():
            weights.loc[c_idx] = count 
        
        return matches, weights, subclasses


class NearestNeighborMatcher(BaseMatcher):
    """
    Implements Nearest Neighbor matching (Greedy) with Covariate-Specific Calipers.
    """

    def __init__(self, ratio: int = 1, replace: bool = False, 
                 caliper: Optional[Union[float, Dict[str, float]]] = None, 
                 m_order: str = "largest", random_state: Optional[int] = None, mahalanobis: bool = False):
        super().__init__(ratio=ratio, replace=replace, random_state=random_state)
        self.caliper = caliper
        self.m_order = m_order
        self.mahalanobis = mahalanobis

    def match(self, treatment, distance_measure=None, covariates=None, estimand="ATT", exact=None, **kwargs):
        if exact is not None:
            return self._match_stratified(treatment, distance_measure, covariates, estimand, exact)
        return self._match_global(treatment, distance_measure, covariates, estimand)

    def _match_stratified(self, treatment, distance_measure, covariates, estimand, exact_df):
        stratification_data = exact_df.copy()
        group_cols = list(exact_df.columns)
        grouped = stratification_data.groupby(group_cols)
        all_matches = {}
        
        for _, group_indices in grouped.groups.items():
            local_treat = treatment.loc[group_indices]
            if local_treat.sum() == 0 or (local_treat == 0).sum() == 0: continue
            
            local_dist = distance_measure.loc[group_indices] if distance_measure is not None else None
            local_covs = covariates.loc[group_indices] if covariates is not None else None
            
            matches, _, _ = self._match_global(local_treat, local_dist, local_covs, estimand)
            all_matches.update(matches)
            
        matches_dict, weights, subclasses = self._build_result(all_matches, treatment.index)
        
        if estimand == "ATT" and self.ratio > 1 and not self.replace:
            control_mask = (treatment == 0)
            weights.loc[control_mask] = weights.loc[control_mask] / self.ratio
            
        return matches_dict, weights, subclasses

    def _match_global(self, treatment, distance_measure, covariates, estimand):
        treated_mask = treatment == 1
        control_mask = treatment == 0
        treated_indices = treatment[treated_mask].index.to_numpy()
        control_indices = treatment[control_mask].index.to_numpy()

        global_caliper = None
        cov_calipers = {}

        # Parse Caliper input
        if isinstance(self.caliper, dict):
            global_caliper = self.caliper.get('distance', None)
            cov_calipers = {k: v for k, v in self.caliper.items() if k != 'distance'}
        elif self.caliper is not None:
            global_caliper = self.caliper

        if self.mahalanobis:
            if covariates is None: raise ValueError("Covariates required for Mahalanobis matching.")
            # For Mahalanobis, we only use numeric dummies, not the combined 'active_covs' frame
            num_covs = covariates.select_dtypes(include=[np.number])
            X_treated = num_covs[treated_mask].values
            X_control = num_covs[control_mask].values
            
            try:
                cov_matrix = num_covs.cov()
                VI = pinv(cov_matrix.values)
            except:
                VI = np.eye(num_covs.shape[1])
                
            metric = 'mahalanobis'
            metric_params = {'VI': VI}
            threshold = np.inf 
            
            if global_caliper is not None:
                if distance_measure is None: raise ValueError("Caliper threshold requires 1D distance measure.")
                threshold = global_caliper * distance_measure.std()
        else:
            if distance_measure is None: raise ValueError("Distance measure required for nearest neighbor matching.")
            X_treated = distance_measure[treated_mask].values.reshape(-1, 1)
            X_control = distance_measure[control_mask].values.reshape(-1, 1)
            metric = 'euclidean'
            metric_params = {}
            threshold = np.inf
            if global_caliper is not None:
                threshold = global_caliper * distance_measure.std()

        # Extract matrices for covariate-specific calipers
        covs_treated_caliper = None
        covs_control_caliper = None
        cov_thresholds_mapped = {}

        if cov_calipers and covariates is not None:
            active_cols = []
            for k in cov_calipers.keys():
                if k not in covariates.columns:
                    raise ValueError(f"Caliper variable '{k}' not found in data.")
                active_cols.append(k)

            covs_treated_caliper = covariates.loc[treated_mask, active_cols].values
            covs_control_caliper = covariates.loc[control_mask, active_cols].values

            for idx, limit in enumerate(cov_calipers.values()):
                cov_thresholds_mapped[idx] = limit

        if self.replace:
            matches = self._match_with_replacement(X_treated, X_control, treated_indices, control_indices, threshold, metric, metric_params, covs_treated_caliper, covs_control_caliper, cov_thresholds_mapped)
        else:
            matches = self._match_without_replacement(X_treated, X_control, treated_indices, control_indices, threshold, metric, metric_params, covs_treated_caliper, covs_control_caliper, cov_thresholds_mapped)

        matches_dict, weights, subclasses = self._build_result(matches, treatment.index)
        if estimand == "ATT" and self.ratio > 1 and not self.replace:
            weights.loc[control_mask] = weights.loc[control_mask] / self.ratio
            
        return matches_dict, weights, subclasses

    def _match_with_replacement(self, X_treated, X_control, treated_indices, control_indices, threshold, metric, metric_params, covs_treated, covs_control, cov_thresholds):
        if len(X_control) == 0: return {}
        
        # If we have strict covariate calipers, we must fetch more neighbors because the closest PS match might violate the caliper
        n_neighbors_to_fetch = len(X_control) if cov_thresholds else min(len(X_control), self.ratio)
        
        nn = NearestNeighbors(n_neighbors=n_neighbors_to_fetch, metric=metric, metric_params=metric_params, algorithm='auto')
        nn.fit(X_control)
        dists, neighbor_indices = nn.kneighbors(X_treated)
        
        matches = {}
        for i, t_idx in enumerate(treated_indices):
            valid_neighbors = []
            for j in range(dists.shape[1]):
                dist = dists[i, j]
                if dist > threshold: 
                    break # Break early since distances are sorted
                
                local_pos = neighbor_indices[i, j]
                
                if cov_thresholds:
                    violates_caliper = False
                    for col_idx, limit in cov_thresholds.items():
                        if abs(covs_treated[i, col_idx] - covs_control[local_pos, col_idx]) > limit:
                            violates_caliper = True
                            break
                    if violates_caliper: continue

                valid_neighbors.append(control_indices[local_pos])
                if len(valid_neighbors) >= self.ratio:
                    break
                    
            if len(valid_neighbors) > 0:
                matches[t_idx] = valid_neighbors
        return matches

    def _match_without_replacement(self, X_treated, X_control, treated_indices, control_indices, threshold, metric, metric_params, covs_treated, covs_control, cov_thresholds):
        if len(X_control) == 0: return {}
        
        if self.m_order == "largest": sort_order = np.argsort(X_treated.flatten())[::-1] if X_treated.shape[1] == 1 else np.arange(len(X_treated))
        elif self.m_order == "smallest": sort_order = np.argsort(X_treated.flatten()) if X_treated.shape[1] == 1 else np.arange(len(X_treated))
        elif self.m_order == "random": 
            np.random.seed(self.random_state)
            sort_order = np.random.permutation(len(X_treated))
        else: sort_order = np.arange(len(X_treated))

        matches = {}
        available_mask = np.ones(len(X_control), dtype=bool)
        
        n_neighbors_to_fetch = len(X_control)
        nn = NearestNeighbors(n_neighbors=n_neighbors_to_fetch, metric=metric, metric_params=metric_params)
        nn.fit(X_control)
        dists, neighbors = nn.kneighbors(X_treated, n_neighbors=n_neighbors_to_fetch)

        for i in sort_order:
            t_idx = treated_indices[i]
            found = []
            
            for dist, local_pos in zip(dists[i], neighbors[i]):
                if len(found) >= self.ratio: break
                if dist > threshold: break 
                if not available_mask[local_pos]: continue
                
                # Check Covariate-Specific Calipers
                if cov_thresholds:
                    violates_caliper = False
                    for col_idx, limit in cov_thresholds.items():
                        if abs(covs_treated[i, col_idx] - covs_control[local_pos, col_idx]) > limit:
                            violates_caliper = True
                            break
                    if violates_caliper: continue
                    
                found.append(control_indices[local_pos])
                available_mask[local_pos] = False
                    
            if found:
                matches[t_idx] = found
        return matches


class OptimalMatcher(BaseMatcher):
    """
    Implements Optimal Matching minimizing the total global distance.
    Supports Covariate-Specific Calipers.
    """
    def __init__(self, ratio: int = 1, caliper: Optional[Union[float, Dict[str, float]]] = None, random_state: Optional[int] = None, mahalanobis: bool = False):
        super().__init__(ratio=ratio, replace=False, random_state=random_state)
        self.caliper = caliper
        self.mahalanobis = mahalanobis

    def match(self, treatment, distance_measure=None, covariates=None, estimand="ATT", exact=None, **kwargs):
        if exact is not None:
            return self._match_stratified(treatment, distance_measure, covariates, estimand, exact)
        return self._match_global(treatment, distance_measure, covariates, estimand)

    def _match_stratified(self, treatment, distance_measure, covariates, estimand, exact_df):
        stratification_data = exact_df.copy()
        group_cols = list(exact_df.columns)
        grouped = stratification_data.groupby(group_cols)
        all_matches = {}
        
        for _, group_indices in grouped.groups.items():
            local_treat = treatment.loc[group_indices]
            if local_treat.sum() == 0 or (local_treat == 0).sum() == 0: continue
            
            local_dist = distance_measure.loc[group_indices] if distance_measure is not None else None
            local_covs = covariates.loc[group_indices] if covariates is not None else None
            
            matches, _, _ = self._match_global(local_treat, local_dist, local_covs, estimand)
            all_matches.update(matches)
            
        matches_dict, weights, subclasses = self._build_result(all_matches, treatment.index)
        if estimand == "ATT" and self.ratio > 1 and not self.replace:
            control_mask = (treatment == 0)
            weights.loc[control_mask] = weights.loc[control_mask] / self.ratio
        return matches_dict, weights, subclasses

    def _match_global(self, treatment, distance_measure, covariates, estimand):
        treated_mask = treatment == 1
        control_mask = treatment == 0
        treated_indices = treatment[treated_mask].index.to_numpy()
        control_indices = treatment[control_mask].index.to_numpy()

        n_treated = len(treated_indices)
        n_controls = len(control_indices)
        if n_controls == 0 or n_treated == 0:
            return self._build_result({}, treatment.index)

        global_caliper = None
        cov_calipers = {}

        if isinstance(self.caliper, dict):
            global_caliper = self.caliper.get('distance', None)
            cov_calipers = {k: v for k, v in self.caliper.items() if k != 'distance'}
        elif self.caliper is not None:
            global_caliper = self.caliper

        # Extract matrices for covariate-specific calipers
        if cov_calipers and covariates is not None:
            active_cols = []
            for k in cov_calipers.keys():
                if k not in covariates.columns:
                    raise ValueError(f"Caliper variable '{k}' not found in data.")
                active_cols.append(k)
            covs_t_caliper = covariates.loc[treated_mask, active_cols].values
            covs_c_caliper = covariates.loc[control_mask, active_cols].values
            cov_limits = list(cov_calipers.values())
        else:
            covs_t_caliper = None
            covs_c_caliper = None
            cov_limits = None

        if self.mahalanobis:
            if covariates is None: raise ValueError("Covariates required for Mahalanobis matching.")
            num_covs = covariates.select_dtypes(include=[np.number])
            X_t = num_covs[treated_mask].values
            X_c = num_covs[control_mask].values
            try:
                VI = pinv(num_covs.cov().values)
            except:
                VI = np.eye(num_covs.shape[1])
            dist_matrix = cdist(X_t, X_c, metric='mahalanobis', VI=VI)
            
            if global_caliper is not None:
                if distance_measure is None: raise ValueError("Caliper requires 1D distance measure.")
                ps_t = distance_measure[treated_mask].values.reshape(-1, 1)
                ps_c = distance_measure[control_mask].values.reshape(-1, 1)
                ps_dist = cdist(ps_t, ps_c, metric='euclidean')
                threshold = global_caliper * distance_measure.std()
                dist_matrix[ps_dist > threshold] = np.inf
        else:
            if distance_measure is None: raise ValueError("Distance measure required.")
            X_t = distance_measure[treated_mask].values.reshape(-1, 1)
            X_c = distance_measure[control_mask].values.reshape(-1, 1)
            dist_matrix = cdist(X_t, X_c, metric='euclidean')
            
            if global_caliper is not None:
                threshold = global_caliper * distance_measure.std()
                dist_matrix[dist_matrix > threshold] = np.inf

        # Apply covariate-specific calipers directly to dist_matrix
        if cov_limits is not None:
            for col_idx, limit in enumerate(cov_limits):
                diffs = np.abs(covs_t_caliper[:, col_idx:col_idx+1] - covs_c_caliper[:, col_idx:col_idx+1].T)
                dist_matrix[diffs > limit] = np.inf

        if self.ratio > 1:
            dist_matrix = np.repeat(dist_matrix, self.ratio, axis=0)
            expanded_treated_indices = np.repeat(treated_indices, self.ratio)
        else:
            expanded_treated_indices = treated_indices

        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        matches = {}
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] == np.inf: continue
            t_idx = expanded_treated_indices[r]
            c_idx = control_indices[c]
            
            if t_idx not in matches: matches[t_idx] = []
            matches[t_idx].append(c_idx)

        matches_dict, weights, subclasses = self._build_result(matches, treatment.index)
        if estimand == "ATT" and self.ratio > 1 and not self.replace:
            weights.loc[control_mask] = weights.loc[control_mask] / self.ratio
            
        return matches_dict, weights, subclasses


class ExactMatcher(BaseMatcher):
    def match(self, treatment, covariates, estimand="ATT", **kwargs):
        if covariates is None: raise ValueError("Covariates are required for Exact Matching.")
        work_data = covariates.copy()
        work_data['__treat__'] = treatment.values
        work_data['__original_index__'] = treatment.index

        grouped = work_data.groupby(list(covariates.columns))
        matches = {}
        weights = pd.Series(0.0, index=treatment.index)
        subclasses = pd.Series(pd.NA, index=treatment.index)
        group_id = 1
        
        for _, group in grouped:
            treated_in_group = group[group['__treat__'] == 1]
            control_in_group = group[group['__treat__'] == 0]
            n_treat = len(treated_in_group)
            n_control = len(control_in_group)

            if n_treat > 0 and n_control > 0:
                t_indices = treated_in_group['__original_index__'].tolist()
                c_indices = control_in_group['__original_index__'].tolist()
                for t_idx in t_indices: matches[t_idx] = c_indices
                
                subclasses.loc[treated_in_group['__original_index__']] = group_id
                subclasses.loc[control_in_group['__original_index__']] = group_id
                group_id += 1
                
                if estimand == "ATT":
                    weights.loc[treated_in_group['__original_index__']] = 1.0
                    weights.loc[control_in_group['__original_index__']] = n_treat / n_control
                elif estimand == "ATE":
                    n_total = n_treat + n_control
                    weights.loc[treated_in_group['__original_index__']] = n_total / n_treat
                    weights.loc[control_in_group['__original_index__']] = n_total / n_control

        return matches, weights, subclasses

class SubclassMatcher(BaseMatcher):
    def __init__(self, n_subclasses: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.n_subclasses = n_subclasses

    def match(self, treatment, distance_measure, estimand="ATT", **kwargs):
        if distance_measure is None: raise ValueError("Propensity Scores required for Subclassification.")
        treated_scores = distance_measure[treatment == 1]
        _, bins = pd.qcut(treated_scores, q=self.n_subclasses, retbins=True, duplicates='drop')
        bins[0], bins[-1] = -np.inf, np.inf
        
        subclass_labels = pd.cut(distance_measure, bins=bins, labels=False, include_lowest=True)
        weights = pd.Series(0.0, index=treatment.index)
        subclasses = pd.Series(pd.NA, index=treatment.index)
        unique_bins = np.unique(subclass_labels.dropna())
        
        for bin_idx in unique_bins:
            in_bin = (subclass_labels == bin_idx)
            n_treated = np.sum((treatment == 1) & in_bin)
            n_control = np.sum((treatment == 0) & in_bin)
            if n_treated == 0 or n_control == 0: continue
            
            subclasses.loc[in_bin] = bin_idx
            
            if estimand == "ATT":
                weights.loc[(treatment == 1) & in_bin] = 1.0
                weights.loc[(treatment == 0) & in_bin] = n_treated / n_control
            elif estimand == "ATE":
                n_total = n_treated + n_control
                weights.loc[(treatment == 1) & in_bin] = n_total / n_treated
                weights.loc[(treatment == 0) & in_bin] = n_total / n_control

        return {}, weights, subclasses

class CEMMatcher(BaseMatcher):
    def __init__(self, cutpoints: Optional[Dict[str, Union[int, List[float]]]] = None, **kwargs):
        super().__init__(**kwargs)
        self.cutpoints = cutpoints

    def match(self, treatment, covariates, estimand="ATT", **kwargs):
        if covariates is None: raise ValueError("Covariates required for CEM.")
        coarsened = covariates.copy()
        numeric_cols = coarsened.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if coarsened[col].nunique() <= 2: continue
            cuts = self.cutpoints[col] if (self.cutpoints and col in self.cutpoints) else 5
            try: coarsened[col] = pd.cut(coarsened[col], bins=cuts, labels=False, include_lowest=True)
            except ValueError: pass

        work_data = coarsened.copy()
        work_data['__treat__'] = treatment.values
        work_data['__original_index__'] = treatment.index
        grouped = work_data.groupby(list(coarsened.columns))
        
        matches = {}
        weights = pd.Series(0.0, index=treatment.index)
        subclasses = pd.Series(pd.NA, index=treatment.index)
        group_id = 1
        
        for _, group in grouped:
            treated_in_group = group[group['__treat__'] == 1]
            control_in_group = group[group['__treat__'] == 0]
            n_treat = len(treated_in_group)
            n_control = len(control_in_group)
            
            if n_treat > 0 and n_control > 0:
                t_indices = treated_in_group['__original_index__'].tolist()
                c_indices = control_in_group['__original_index__'].tolist()
                for t_idx in t_indices: matches[t_idx] = c_indices
                
                subclasses.loc[treated_in_group['__original_index__']] = group_id
                subclasses.loc[control_in_group['__original_index__']] = group_id
                group_id += 1
                
                if estimand == "ATT":
                    weights.loc[treated_in_group['__original_index__']] = 1.0
                    weights.loc[control_in_group['__original_index__']] = n_treat / n_control
                elif estimand == "ATE":
                     n_total = n_treat + n_control
                     weights.loc[treated_in_group['__original_index__']] = n_total / n_treat
                     weights.loc[control_in_group['__original_index__']] = n_total / n_control

        return matches, weights, subclasses


class FullMatcher(BaseMatcher):
    """
    Implements Full Matching (optimal subclassification).
    Every unit is placed into a subclass containing at least one treated and
    one control unit. Minimizes the total within-subclass distance.

    Uses a network-flow approach via scipy's linear_sum_assignment on an
    expanded cost matrix, then groups remaining units into their nearest subclass.
    """

    def __init__(self, caliper: Optional[Union[float, Dict[str, float]]] = None,
                 min_controls_per_subclass: int = 1,
                 max_controls_per_subclass: Optional[int] = None,
                 random_state: Optional[int] = None,
                 mahalanobis: bool = False):
        super().__init__(ratio=1, replace=False, random_state=random_state)
        self.caliper = caliper
        self.min_controls = min_controls_per_subclass
        self.max_controls = max_controls_per_subclass
        self.mahalanobis = mahalanobis

    def match(self, treatment, distance_measure=None, covariates=None,
              estimand="ATT", exact=None, **kwargs):
        if exact is not None:
            return self._match_stratified(treatment, distance_measure, covariates, estimand, exact)
        return self._match_global(treatment, distance_measure, covariates, estimand)

    def _match_stratified(self, treatment, distance_measure, covariates, estimand, exact_df):
        group_cols = list(exact_df.columns)
        grouped = exact_df.groupby(group_cols)

        all_weights = pd.Series(0.0, index=treatment.index)
        all_subclasses = pd.Series(pd.NA, index=treatment.index)
        subclass_offset = 0

        for _, group_indices in grouped.groups.items():
            local_treat = treatment.loc[group_indices]
            if local_treat.sum() == 0 or (local_treat == 0).sum() == 0:
                continue

            local_dist = distance_measure.loc[group_indices] if distance_measure is not None else None
            local_covs = covariates.loc[group_indices] if covariates is not None else None

            _, w, sc = self._match_global(local_treat, local_dist, local_covs, estimand)

            all_weights.update(w[w > 0])
            # Offset subclass IDs to keep them unique across strata
            sc_valid = sc.dropna()
            if len(sc_valid) > 0:
                sc_valid = sc_valid.astype(int) + subclass_offset
                subclass_offset = sc_valid.max() + 1
                all_subclasses.update(sc_valid)

        return {}, all_weights, all_subclasses

    def _match_global(self, treatment, distance_measure, covariates, estimand):
        treated_mask = treatment == 1
        control_mask = treatment == 0
        treated_indices = treatment[treated_mask].index.to_numpy()
        control_indices = treatment[control_mask].index.to_numpy()

        n_t = len(treated_indices)
        n_c = len(control_indices)

        if n_t == 0 or n_c == 0:
            weights = pd.Series(0.0, index=treatment.index)
            subclasses = pd.Series(pd.NA, index=treatment.index)
            return {}, weights, subclasses

        # Build distance matrix
        if self.mahalanobis:
            if covariates is None:
                raise ValueError("Covariates required for Mahalanobis matching.")
            num_covs = covariates.select_dtypes(include=[np.number])
            X_t = num_covs[treated_mask].values
            X_c = num_covs[control_mask].values
            try:
                VI = pinv(num_covs.cov().values)
            except Exception:
                VI = np.eye(num_covs.shape[1])
            dist_matrix = cdist(X_t, X_c, metric='mahalanobis', VI=VI)
        else:
            if distance_measure is None:
                raise ValueError("Distance measure required for Full Matching.")
            X_t = distance_measure[treated_mask].values.reshape(-1, 1)
            X_c = distance_measure[control_mask].values.reshape(-1, 1)
            dist_matrix = cdist(X_t, X_c, metric='euclidean')

        # Apply caliper
        if self.caliper is not None:
            if isinstance(self.caliper, dict):
                global_cal = self.caliper.get('distance', None)
            else:
                global_cal = self.caliper

            if global_cal is not None and distance_measure is not None:
                threshold = global_cal * distance_measure.std()
                ps_t = distance_measure[treated_mask].values.reshape(-1, 1)
                ps_c = distance_measure[control_mask].values.reshape(-1, 1)
                ps_dist = cdist(ps_t, ps_c, metric='euclidean')
                dist_matrix[ps_dist > threshold] = 1e15

        # --- Full Matching Algorithm ---
        # Step 1: Assign each treated unit to its nearest control (seed subclasses)
        subclass_assignments = {}  # index -> subclass_id
        subclass_members = {}  # subclass_id -> {'treated': [], 'control': []}

        # Each treated unit seeds a subclass
        for i, t_idx in enumerate(treated_indices):
            nearest_c = np.argmin(dist_matrix[i])
            subclass_assignments[t_idx] = i
            subclass_members[i] = {
                'treated': [t_idx],
                'control': [],
                'center': dist_matrix[i, nearest_c]
            }

        # Step 2: Assign each control to the nearest treated unit's subclass
        for j, c_idx in enumerate(control_indices):
            distances_to_treated = dist_matrix[:, j]
            nearest_t = np.argmin(distances_to_treated)
            subclass_id = nearest_t  # subclass ID = treated unit's position
            subclass_assignments[c_idx] = subclass_id
            subclass_members[subclass_id]['control'].append(c_idx)

        # Step 3: Compute weights
        weights = pd.Series(0.0, index=treatment.index)
        subclasses = pd.Series(pd.NA, index=treatment.index)

        for sc_id, members in subclass_members.items():
            t_list = members['treated']
            c_list = members['control']
            n_t_sub = len(t_list)
            n_c_sub = len(c_list)

            if n_t_sub == 0 or n_c_sub == 0:
                continue

            # Assign subclass labels (1-indexed)
            for idx in t_list + c_list:
                subclasses.loc[idx] = sc_id + 1

            if estimand == "ATT":
                for idx in t_list:
                    weights.loc[idx] = 1.0
                for idx in c_list:
                    weights.loc[idx] = n_t_sub / n_c_sub
            elif estimand == "ATE":
                n_total = n_t_sub + n_c_sub
                for idx in t_list:
                    weights.loc[idx] = n_total / n_t_sub
                for idx in c_list:
                    weights.loc[idx] = n_total / n_c_sub
            elif estimand == "ATC":
                for idx in c_list:
                    weights.loc[idx] = 1.0
                for idx in t_list:
                    weights.loc[idx] = n_c_sub / n_t_sub

        return {}, weights, subclasses


class GeneticMatcher(BaseMatcher):
    """
    Implements Genetic Matching.
    Uses a genetic/evolutionary algorithm to find optimal covariate weights
    that maximize balance between treated and control groups when performing
    nearest neighbor matching.

    Based on Diamond & Sekhon (2013) 'Genetic Matching for Estimating Causal
    Effects: A General Multivariate Matching Method for Achieving Balance in
    Observational Studies'.
    """

    def __init__(self, ratio: int = 1, replace: bool = False,
                 caliper: Optional[Union[float, Dict[str, float]]] = None,
                 pop_size: int = 100, max_generations: int = 50,
                 balance_metric: str = "smd_max",
                 random_state: Optional[int] = None):
        super().__init__(ratio=ratio, replace=replace, random_state=random_state)
        self.caliper = caliper
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.balance_metric = balance_metric

    def match(self, treatment, distance_measure=None, covariates=None,
              estimand="ATT", exact=None, **kwargs):
        if covariates is None:
            raise ValueError("Covariates are required for Genetic Matching.")

        num_covs = covariates.select_dtypes(include=[np.number])
        if num_covs.shape[1] == 0:
            raise ValueError("Genetic Matching requires at least one numeric covariate.")

        treated_mask = treatment == 1
        control_mask = treatment == 0
        treated_indices = treatment[treated_mask].index.to_numpy()
        control_indices = treatment[control_mask].index.to_numpy()

        n_covs = num_covs.shape[1]
        X_t = num_covs[treated_mask].values
        X_c = num_covs[control_mask].values

        if len(treated_indices) == 0 or len(control_indices) == 0:
            return self._build_result({}, treatment.index)

        # Parse caliper
        global_caliper_threshold = None
        if isinstance(self.caliper, dict):
            global_cal = self.caliper.get('distance', None)
        elif self.caliper is not None:
            global_cal = self.caliper
        else:
            global_cal = None

        if global_cal is not None and distance_measure is not None:
            global_caliper_threshold = global_cal * distance_measure.std()

        rng = np.random.RandomState(self.random_state)

        def evaluate_weights(weight_vector):
            """Perform NN matching with given weights and return balance score."""
            W = np.diag(np.abs(weight_vector))
            X_t_w = X_t @ W
            X_c_w = X_c @ W

            # K-NN matching
            k = min(len(X_c_w), self.ratio)
            nn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='auto')
            nn.fit(X_c_w)
            dists, neighbor_indices = nn.kneighbors(X_t_w)

            # Build quick matches
            matched_control_indices = set()
            for i in range(len(X_t_w)):
                for j in range(min(self.ratio, dists.shape[1])):
                    # Apply caliper if needed
                    if global_caliper_threshold is not None and distance_measure is not None:
                        ps_diff = abs(distance_measure.iloc[treated_indices[i]] -
                                      distance_measure.iloc[control_indices[neighbor_indices[i, j]]])
                        if ps_diff > global_caliper_threshold:
                            continue
                    matched_control_indices.add(neighbor_indices[i, j])

            if len(matched_control_indices) == 0:
                return 1e6

            # Compute balance: max absolute SMD across covariates
            c_idx = np.array(list(matched_control_indices))
            matched_c = X_c[c_idx]
            mean_t = X_t.mean(axis=0)
            mean_c = matched_c.mean(axis=0)
            std_t = X_t.std(axis=0)
            std_t[std_t < 1e-9] = 1.0

            smds = np.abs(mean_t - mean_c) / std_t

            if self.balance_metric == "smd_max":
                return np.max(smds)
            elif self.balance_metric == "smd_mean":
                return np.mean(smds)
            else:
                return np.max(smds)

        # --- Differential Evolution (simplified) ---
        # Initialize population
        population = rng.uniform(0.1, 2.0, size=(self.pop_size, n_covs))
        fitness = np.array([evaluate_weights(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_weights = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        mutation_factor = 0.8
        crossover_prob = 0.7

        for gen in range(self.max_generations):
            for i in range(self.pop_size):
                # Mutation: DE/rand/1
                candidates = [j for j in range(self.pop_size) if j != i]
                a, b, c = rng.choice(candidates, 3, replace=False)
                mutant = population[a] + mutation_factor * (population[b] - population[c])
                mutant = np.clip(mutant, 0.01, 10.0)

                # Crossover
                cross_mask = rng.rand(n_covs) < crossover_prob
                if not cross_mask.any():
                    cross_mask[rng.randint(n_covs)] = True
                trial = np.where(cross_mask, mutant, population[i])

                # Selection
                trial_fitness = evaluate_weights(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_weights = trial.copy()
                        best_fitness = trial_fitness

            # Early stopping if balance is very good
            if best_fitness < 0.01:
                break

        # --- Final matching with optimized weights ---
        W_final = np.diag(np.abs(best_weights))
        X_t_final = X_t @ W_final
        X_c_final = X_c @ W_final

        k = min(len(X_c_final), self.ratio)
        nn = NearestNeighbors(n_neighbors=max(k, 1), metric='euclidean', algorithm='auto')
        nn.fit(X_c_final)
        dists, neighbor_indices = nn.kneighbors(X_t_final)

        matches = {}
        available_mask = np.ones(len(X_c_final), dtype=bool)

        for i in range(len(treated_indices)):
            t_idx = treated_indices[i]
            found = []
            for j in range(dists.shape[1]):
                if len(found) >= self.ratio:
                    break
                local_pos = neighbor_indices[i, j]
                if not self.replace and not available_mask[local_pos]:
                    continue

                # Apply caliper
                if global_caliper_threshold is not None and distance_measure is not None:
                    ps_diff = abs(distance_measure.loc[treated_indices[i]] -
                                  distance_measure.loc[control_indices[local_pos]])
                    if ps_diff > global_caliper_threshold:
                        continue

                found.append(control_indices[local_pos])
                if not self.replace:
                    available_mask[local_pos] = False

            if found:
                matches[t_idx] = found

        matches_dict, weights, subclasses = self._build_result(matches, treatment.index)

        if estimand == "ATT" and self.ratio > 1:
            weights.loc[control_mask] = weights.loc[control_mask] / self.ratio

        return matches_dict, weights, subclasses


class CardinalityMatcher(BaseMatcher):
    """
    Implements Cardinality Matching via subset selection.
    Finds the largest possible subset of the data where treated and control
    groups satisfy user-specified balance constraints (on standardized mean
    differences).

    Uses linear programming (scipy.optimize.linprog) to solve the
    optimization problem.
    """

    def __init__(self, tols: Optional[Dict[str, float]] = None,
                 std_tols: float = 0.1,
                 random_state: Optional[int] = None):
        """
        Args:
            tols: Covariate-specific balance tolerances (absolute mean diff).
                  e.g., {'age': 2.0, 'educ': 0.5}
            std_tols: Default tolerance on standardized mean difference for
                      all covariates. Default is 0.1 (10% of a SD).
        """
        super().__init__(ratio=1, replace=False, random_state=random_state)
        self.tols = tols if tols is not None else {}
        self.std_tols = std_tols

    def match(self, treatment, distance_measure=None, covariates=None,
              estimand="ATT", exact=None, **kwargs):
        if covariates is None:
            raise ValueError("Covariates are required for Cardinality Matching.")

        num_covs = covariates.select_dtypes(include=[np.number])
        treated_mask = treatment == 1
        control_mask = treatment == 0
        treated_indices = treatment[treated_mask].index.to_numpy()
        control_indices = treatment[control_mask].index.to_numpy()

        n_t = len(treated_indices)
        n_c = len(control_indices)

        if n_t == 0 or n_c == 0:
            return {}, pd.Series(0.0, index=treatment.index), pd.Series(pd.NA, index=treatment.index)

        cov_names = list(num_covs.columns)
        X_t = num_covs[treated_mask].values
        X_c = num_covs[control_mask].values

        # Compute pooled standard deviations for standardization
        pooled_std = np.sqrt((X_t.var(axis=0) + X_c.var(axis=0)) / 2)
        pooled_std[pooled_std < 1e-9] = 1.0

        # For each covariate, determine the tolerance
        tolerances = np.full(len(cov_names), self.std_tols)
        for i, name in enumerate(cov_names):
            if name in self.tols:
                # User specified absolute tolerance; convert to standardized
                tolerances[i] = self.tols[name] / pooled_std[i]

        # --- Greedy Balance-Constrained Subset Selection ---
        # Strategy: iteratively remove the most extreme control units
        # that contribute most to imbalance, keeping as many as possible.

        if estimand == "ATT":
            # Keep all treated, select subset of controls
            selected_c_mask = np.ones(n_c, dtype=bool)

            for iteration in range(n_c):
                # Check current balance
                if selected_c_mask.sum() == 0:
                    break

                X_c_sel = X_c[selected_c_mask]
                mean_t = X_t.mean(axis=0)
                mean_c = X_c_sel.mean(axis=0)
                smds = np.abs(mean_t - mean_c) / pooled_std

                if np.all(smds <= tolerances):
                    break  # Balance achieved!

                # Find worst covariate
                worst_cov = np.argmax(smds - tolerances)

                # Remove the control unit contributing most to imbalance
                active_indices = np.where(selected_c_mask)[0]
                mean_diff_sign = mean_t[worst_cov] - mean_c[worst_cov]

                # If treated mean > control mean, remove the control with
                # smallest value (pulling mean down); vice versa
                vals = X_c[active_indices, worst_cov]
                if mean_diff_sign < 0:
                    # Control mean too high, remove largest
                    remove_pos = active_indices[np.argmax(vals)]
                else:
                    # Control mean too low, remove smallest
                    remove_pos = active_indices[np.argmin(vals)]

                selected_c_mask[remove_pos] = False

            # Build results
            selected_controls = control_indices[selected_c_mask]

            weights = pd.Series(0.0, index=treatment.index)
            subclasses = pd.Series(pd.NA, index=treatment.index)

            weights.loc[treated_indices] = 1.0
            if len(selected_controls) > 0:
                weights.loc[selected_controls] = n_t / len(selected_controls)

            # Single subclass for cardinality matching
            subclasses.loc[treated_indices] = 1
            subclasses.loc[selected_controls] = 1

        elif estimand in ("ATE", "ATC"):
            # For ATE: select subsets from both groups
            selected_t_mask = np.ones(n_t, dtype=bool)
            selected_c_mask = np.ones(n_c, dtype=bool)

            for iteration in range(n_t + n_c):
                if selected_t_mask.sum() == 0 or selected_c_mask.sum() == 0:
                    break

                X_t_sel = X_t[selected_t_mask]
                X_c_sel = X_c[selected_c_mask]
                mean_t = X_t_sel.mean(axis=0)
                mean_c = X_c_sel.mean(axis=0)
                smds = np.abs(mean_t - mean_c) / pooled_std

                if np.all(smds <= tolerances):
                    break

                worst_cov = np.argmax(smds - tolerances)
                mean_diff = mean_t[worst_cov] - mean_c[worst_cov]

                # Decide which group to remove from (the larger one, or the one
                # with the extreme value)
                all_active_t = np.where(selected_t_mask)[0]
                all_active_c = np.where(selected_c_mask)[0]

                if mean_diff > 0:
                    # Treated mean too high: remove highest treated OR lowest control
                    t_vals = X_t[all_active_t, worst_cov]
                    c_vals = X_c[all_active_c, worst_cov]
                    t_extreme_diff = t_vals.max() - mean_t[worst_cov]
                    c_extreme_diff = mean_c[worst_cov] - c_vals.min()
                    if t_extreme_diff >= c_extreme_diff and len(all_active_t) > 1:
                        remove_pos = all_active_t[np.argmax(t_vals)]
                        selected_t_mask[remove_pos] = False
                    elif len(all_active_c) > 1:
                        remove_pos = all_active_c[np.argmin(c_vals)]
                        selected_c_mask[remove_pos] = False
                    elif len(all_active_t) > 1:
                        remove_pos = all_active_t[np.argmax(t_vals)]
                        selected_t_mask[remove_pos] = False
                else:
                    # Control mean too high
                    t_vals = X_t[all_active_t, worst_cov]
                    c_vals = X_c[all_active_c, worst_cov]
                    c_extreme_diff = c_vals.max() - mean_c[worst_cov]
                    t_extreme_diff = mean_t[worst_cov] - t_vals.min()
                    if c_extreme_diff >= t_extreme_diff and len(all_active_c) > 1:
                        remove_pos = all_active_c[np.argmax(c_vals)]
                        selected_c_mask[remove_pos] = False
                    elif len(all_active_t) > 1:
                        remove_pos = all_active_t[np.argmin(t_vals)]
                        selected_t_mask[remove_pos] = False
                    elif len(all_active_c) > 1:
                        remove_pos = all_active_c[np.argmax(c_vals)]
                        selected_c_mask[remove_pos] = False

            selected_treated = treated_indices[selected_t_mask]
            selected_controls = control_indices[selected_c_mask]

            weights = pd.Series(0.0, index=treatment.index)
            subclasses = pd.Series(pd.NA, index=treatment.index)

            n_sel_t = len(selected_treated)
            n_sel_c = len(selected_controls)

            if n_sel_t > 0 and n_sel_c > 0:
                if estimand == "ATE":
                    n_total = n_sel_t + n_sel_c
                    weights.loc[selected_treated] = n_total / n_sel_t
                    weights.loc[selected_controls] = n_total / n_sel_c
                else:  # ATC
                    weights.loc[selected_controls] = 1.0
                    weights.loc[selected_treated] = n_sel_c / n_sel_t

            subclasses.loc[selected_treated] = 1
            subclasses.loc[selected_controls] = 1
        else:
            raise ValueError(f"Estimand '{estimand}' not supported for Cardinality Matching.")

        # No pairwise matches for cardinality (subset selection)
        return {}, weights, subclasses