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
        
        if estimand == "ATT" and self.ratio > 1:
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
        if estimand == "ATT" and self.ratio > 1:
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
        if estimand == "ATT" and self.ratio > 1:
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
        if estimand == "ATT" and self.ratio > 1:
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