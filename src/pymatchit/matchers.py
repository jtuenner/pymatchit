# File: src/pymatchit/matchers.py

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import pinv
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod


def _resolve_mahalanobis_covariates(covariates: pd.DataFrame, mahvars: Optional[List[str]]) -> pd.DataFrame:
    """
    Selects the covariate columns used to compute the Mahalanobis distance.
    If `mahvars` is given, maps each variable name to its design-matrix
    column(s) (categorical variables appear as dummies like 'race[T.Hispanic]').
    """
    if not mahvars:
        return covariates.select_dtypes(include=[np.number])
    cols = []
    for v in mahvars:
        hits = [c for c in covariates.columns if c == v or c.startswith(f"{v}[")]
        if not hits:
            raise ValueError(f"mahvars variable '{v}' not found among model covariates.")
        cols.extend(hits)
    return covariates[cols].select_dtypes(include=[np.number])


def _antiexact_violations(anti_t: np.ndarray, anti_c: np.ndarray) -> np.ndarray:
    """Boolean (n_treated x n_control) matrix: True where any antiexact variable matches."""
    violations = np.zeros((anti_t.shape[0], anti_c.shape[0]), dtype=bool)
    for k in range(anti_t.shape[1]):
        violations |= (anti_t[:, k:k+1] == anti_c[:, k:k+1].T)
    return violations


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
        # Each matched control contributes 1/k_i for every focal unit i it is
        # matched to, where k_i is that unit's number of matches. This keeps
        # weights correct when calipers leave some units with fewer than
        # `ratio` matches, and reduces to use-counts for 1:1 with replacement.
        control_weights: Dict[Any, float] = {}

        subclasses = pd.Series(pd.NA, index=all_indices)
        group_id = 1

        for t, c_list in matches.items():
            matched_treated.append(t)
            k = len(c_list)
            for c in c_list:
                control_weights[c] = control_weights.get(c, 0.0) + 1.0 / k

            subclasses.loc[t] = group_id
            for c in c_list:
                if pd.isna(subclasses.loc[c]):
                    subclasses.loc[c] = group_id
            group_id += 1

        weights = pd.Series(0.0, index=all_indices)
        weights.loc[matched_treated] = 1.0

        for c_idx, w in control_weights.items():
            weights.loc[c_idx] = w

        return matches, weights, subclasses


class NearestNeighborMatcher(BaseMatcher):
    """
    Implements Nearest Neighbor matching (Greedy) with Covariate-Specific Calipers.
    """

    def __init__(self, ratio: int = 1, replace: bool = False,
                 caliper: Optional[Union[float, Dict[str, float]]] = None,
                 m_order: str = "largest", random_state: Optional[int] = None, mahalanobis: bool = False,
                 mahvars: Optional[List[str]] = None):
        super().__init__(ratio=ratio, replace=replace, random_state=random_state)
        self.caliper = caliper
        self.m_order = m_order
        self.mahalanobis = mahalanobis
        self.mahvars = mahvars

    def match(self, treatment, distance_measure=None, covariates=None, estimand="ATT", exact=None, antiexact=None, **kwargs):
        if estimand == "ATC":
            # The control group becomes the focal group: controls are matched
            # to treated units (mirrors R MatchIt's focal-group switch).
            treatment = 1 - treatment
        if exact is not None:
            return self._match_stratified(treatment, distance_measure, covariates, estimand, exact, antiexact)
        return self._match_global(treatment, distance_measure, covariates, estimand, antiexact)

    def _match_stratified(self, treatment, distance_measure, covariates, estimand, exact_df, antiexact=None):
        stratification_data = exact_df.copy()
        group_cols = list(exact_df.columns)
        grouped = stratification_data.groupby(group_cols)
        all_matches = {}
        # Caliper width is defined on the full sample's distance SD, not per stratum
        dist_std = distance_measure.std() if distance_measure is not None else None

        for _, group_indices in grouped.groups.items():
            local_treat = treatment.loc[group_indices]
            if local_treat.sum() == 0 or (local_treat == 0).sum() == 0: continue

            local_dist = distance_measure.loc[group_indices] if distance_measure is not None else None
            local_covs = covariates.loc[group_indices] if covariates is not None else None
            local_anti = antiexact.loc[group_indices] if antiexact is not None else None

            matches, _, _ = self._match_global(local_treat, local_dist, local_covs, estimand, local_anti, dist_std=dist_std)
            all_matches.update(matches)

        return self._build_result(all_matches, treatment.index)

    def _match_global(self, treatment, distance_measure, covariates, estimand, antiexact=None, dist_std=None):
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

        if dist_std is None and distance_measure is not None:
            dist_std = distance_measure.std()

        if self.mahalanobis:
            if covariates is None: raise ValueError("Covariates required for Mahalanobis matching.")
            # For Mahalanobis, we only use numeric dummies, not the combined 'active_covs' frame
            num_covs = _resolve_mahalanobis_covariates(covariates, self.mahvars)
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
                threshold = global_caliper * dist_std
        else:
            if distance_measure is None: raise ValueError("Distance measure required for nearest neighbor matching.")
            X_treated = distance_measure[treated_mask].values.reshape(-1, 1)
            X_control = distance_measure[control_mask].values.reshape(-1, 1)
            metric = 'euclidean'
            metric_params = {}
            threshold = np.inf
            if global_caliper is not None:
                threshold = global_caliper * dist_std

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

        anti_treated = anti_control = None
        if antiexact is not None:
            anti_treated = antiexact.loc[treated_mask].values
            anti_control = antiexact.loc[control_mask].values

        if self.replace:
            matches = self._match_with_replacement(X_treated, X_control, treated_indices, control_indices, threshold, metric, metric_params, covs_treated_caliper, covs_control_caliper, cov_thresholds_mapped, anti_treated, anti_control)
        else:
            matches = self._match_without_replacement(X_treated, X_control, treated_indices, control_indices, threshold, metric, metric_params, covs_treated_caliper, covs_control_caliper, cov_thresholds_mapped, anti_treated, anti_control)

        return self._build_result(matches, treatment.index)

    @staticmethod
    def _violates_pair_constraints(i, local_pos, covs_treated, covs_control, cov_thresholds, anti_treated, anti_control):
        if cov_thresholds:
            for col_idx, limit in cov_thresholds.items():
                if abs(covs_treated[i, col_idx] - covs_control[local_pos, col_idx]) > limit:
                    return True
        if anti_treated is not None:
            if (anti_treated[i] == anti_control[local_pos]).any():
                return True
        return False

    def _match_with_replacement(self, X_treated, X_control, treated_indices, control_indices, threshold, metric, metric_params, covs_treated, covs_control, cov_thresholds, anti_treated=None, anti_control=None):
        if len(X_control) == 0: return {}

        # With covariate calipers or antiexact constraints, the closest match
        # may be invalid, so we must fetch all candidates
        has_pair_constraints = bool(cov_thresholds) or anti_treated is not None
        n_neighbors_to_fetch = len(X_control) if has_pair_constraints else min(len(X_control), self.ratio)

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

                if self._violates_pair_constraints(i, local_pos, covs_treated, covs_control, cov_thresholds, anti_treated, anti_control):
                    continue

                valid_neighbors.append(control_indices[local_pos])
                if len(valid_neighbors) >= self.ratio:
                    break

            if len(valid_neighbors) > 0:
                matches[t_idx] = valid_neighbors
        return matches

    def _match_without_replacement(self, X_treated, X_control, treated_indices, control_indices, threshold, metric, metric_params, covs_treated, covs_control, cov_thresholds, anti_treated=None, anti_control=None):
        if len(X_control) == 0: return {}

        if self.m_order == "largest": sort_order = np.argsort(X_treated.flatten())[::-1] if X_treated.shape[1] == 1 else np.arange(len(X_treated))
        elif self.m_order == "smallest": sort_order = np.argsort(X_treated.flatten()) if X_treated.shape[1] == 1 else np.arange(len(X_treated))
        elif self.m_order == "random":
            rng = np.random.RandomState(self.random_state)
            sort_order = rng.permutation(len(X_treated))
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

                if self._violates_pair_constraints(i, local_pos, covs_treated, covs_control, cov_thresholds, anti_treated, anti_control):
                    continue

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
    def __init__(self, ratio: int = 1, caliper: Optional[Union[float, Dict[str, float]]] = None, random_state: Optional[int] = None, mahalanobis: bool = False,
                 mahvars: Optional[List[str]] = None):
        super().__init__(ratio=ratio, replace=False, random_state=random_state)
        self.caliper = caliper
        self.mahalanobis = mahalanobis
        self.mahvars = mahvars

    def match(self, treatment, distance_measure=None, covariates=None, estimand="ATT", exact=None, antiexact=None, **kwargs):
        if estimand == "ATC":
            # The control group becomes the focal group (see NearestNeighborMatcher)
            treatment = 1 - treatment
        if exact is not None:
            return self._match_stratified(treatment, distance_measure, covariates, estimand, exact, antiexact)
        return self._match_global(treatment, distance_measure, covariates, estimand, antiexact)

    def _match_stratified(self, treatment, distance_measure, covariates, estimand, exact_df, antiexact=None):
        stratification_data = exact_df.copy()
        group_cols = list(exact_df.columns)
        grouped = stratification_data.groupby(group_cols)
        all_matches = {}
        # Caliper width is defined on the full sample's distance SD, not per stratum
        dist_std = distance_measure.std() if distance_measure is not None else None

        for _, group_indices in grouped.groups.items():
            local_treat = treatment.loc[group_indices]
            if local_treat.sum() == 0 or (local_treat == 0).sum() == 0: continue

            local_dist = distance_measure.loc[group_indices] if distance_measure is not None else None
            local_covs = covariates.loc[group_indices] if covariates is not None else None
            local_anti = antiexact.loc[group_indices] if antiexact is not None else None

            matches, _, _ = self._match_global(local_treat, local_dist, local_covs, estimand, local_anti, dist_std=dist_std)
            all_matches.update(matches)

        return self._build_result(all_matches, treatment.index)

    def _match_global(self, treatment, distance_measure, covariates, estimand, antiexact=None, dist_std=None):
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

        if dist_std is None and distance_measure is not None:
            dist_std = distance_measure.std()

        # Pairs violating a caliper or antiexact constraint are marked forbidden.
        # linear_sum_assignment cannot handle np.inf when no complete feasible
        # assignment exists, so forbidden cells get a large finite penalty and
        # forbidden pairs are dropped from the solution afterwards.
        forbidden = np.zeros((n_treated, n_controls), dtype=bool)

        if self.mahalanobis:
            if covariates is None: raise ValueError("Covariates required for Mahalanobis matching.")
            num_covs = _resolve_mahalanobis_covariates(covariates, self.mahvars)
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
                threshold = global_caliper * dist_std
                forbidden |= ps_dist > threshold
        else:
            if distance_measure is None: raise ValueError("Distance measure required.")
            X_t = distance_measure[treated_mask].values.reshape(-1, 1)
            X_c = distance_measure[control_mask].values.reshape(-1, 1)
            dist_matrix = cdist(X_t, X_c, metric='euclidean')

            if global_caliper is not None:
                threshold = global_caliper * dist_std
                forbidden |= dist_matrix > threshold

        # Apply covariate-specific calipers
        if cov_limits is not None:
            for col_idx, limit in enumerate(cov_limits):
                diffs = np.abs(covs_t_caliper[:, col_idx:col_idx+1] - covs_c_caliper[:, col_idx:col_idx+1].T)
                forbidden |= diffs > limit

        # Apply antiexact constraints
        if antiexact is not None:
            anti_t = antiexact.loc[treated_mask].values
            anti_c = antiexact.loc[control_mask].values
            forbidden |= _antiexact_violations(anti_t, anti_c)

        if forbidden.any():
            feasible_vals = dist_matrix[~forbidden]
            max_feasible = feasible_vals.max() if feasible_vals.size > 0 else 1.0
            n_assignable = min(n_treated * self.ratio, n_controls)
            penalty = (abs(max_feasible) + 1.0) * (n_assignable + 1)
            dist_matrix = dist_matrix.copy()
            dist_matrix[forbidden] = penalty

        if self.ratio > 1:
            dist_matrix = np.repeat(dist_matrix, self.ratio, axis=0)
            forbidden = np.repeat(forbidden, self.ratio, axis=0)
            expanded_treated_indices = np.repeat(treated_indices, self.ratio)
        else:
            expanded_treated_indices = treated_indices

        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        matches = {}
        for r, c in zip(row_ind, col_ind):
            if forbidden[r, c]: continue
            t_idx = expanded_treated_indices[r]
            c_idx = control_indices[c]

            if t_idx not in matches: matches[t_idx] = []
            matches[t_idx].append(c_idx)

        return self._build_result(matches, treatment.index)


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
                elif estimand == "ATC":
                    weights.loc[control_in_group['__original_index__']] = 1.0
                    weights.loc[treated_in_group['__original_index__']] = n_control / n_treat

        return matches, weights, subclasses

class SubclassMatcher(BaseMatcher):
    def __init__(self, n_subclasses: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.n_subclasses = n_subclasses

    def match(self, treatment, distance_measure, estimand="ATT", **kwargs):
        if distance_measure is None: raise ValueError("Propensity Scores required for Subclassification.")
        # Bin edges come from the focal group's score distribution (R MatchIt:
        # treated for ATT, control for ATC, everyone for ATE)
        if estimand == "ATC":
            ref_scores = distance_measure[treatment == 0]
        elif estimand == "ATE":
            ref_scores = distance_measure
        else:
            ref_scores = distance_measure[treatment == 1]
        _, bins = pd.qcut(ref_scores, q=self.n_subclasses, retbins=True, duplicates='drop')
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
            elif estimand == "ATC":
                weights.loc[(treatment == 0) & in_bin] = 1.0
                weights.loc[(treatment == 1) & in_bin] = n_control / n_treated

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
                elif estimand == "ATC":
                    weights.loc[control_in_group['__original_index__']] = 1.0
                    weights.loc[treated_in_group['__original_index__']] = n_control / n_treat

        return matches, weights, subclasses


class FullMatcher(BaseMatcher):
    """
    Implements Full Matching (subclassification with variable ratios).
    Every matchable unit is placed into a subclass containing at least one
    treated and one control unit.

    The algorithm is greedy but seeded by an optimal 1:1 assignment
    (scipy's linear_sum_assignment): the majority group's remaining units
    are then attached to the subclass of their nearest feasible opposite
    unit. Units with no within-caliper partner are left unmatched, and
    min/max controls per subclass are enforced. This approximates, but does
    not guarantee, the provably optimal full matching of Hansen & Klopfer
    (2006) used by R's optmatch.
    """

    def __init__(self, caliper: Optional[Union[float, Dict[str, float]]] = None,
                 min_controls_per_subclass: int = 1,
                 max_controls_per_subclass: Optional[int] = None,
                 random_state: Optional[int] = None,
                 mahalanobis: bool = False,
                 mahvars: Optional[List[str]] = None):
        super().__init__(ratio=1, replace=False, random_state=random_state)
        self.caliper = caliper
        self.min_controls = min_controls_per_subclass
        self.max_controls = max_controls_per_subclass
        self.mahalanobis = mahalanobis
        self.mahvars = mahvars

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
            num_covs = _resolve_mahalanobis_covariates(covariates, self.mahvars)
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

        # Caliper feasibility: pairs outside the caliper cannot share a subclass
        feasible = np.ones((n_t, n_c), dtype=bool)
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
                feasible = ps_dist <= threshold

        clusters = self._build_clusters(dist_matrix, feasible, n_t, n_c)

        # Compute weights per subclass
        weights = pd.Series(0.0, index=treatment.index)
        subclasses = pd.Series(pd.NA, index=treatment.index)

        for sc_num, members in enumerate(clusters, start=1):
            t_list = [treated_indices[i] for i in members['treated']]
            c_list = [control_indices[j] for j in members['control']]
            n_t_sub = len(t_list)
            n_c_sub = len(c_list)

            if n_t_sub == 0 or n_c_sub == 0:
                continue

            for idx in t_list + c_list:
                subclasses.loc[idx] = sc_num

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

    def _build_clusters(self, dist_matrix, feasible, n_t, n_c):
        """
        Groups treated/control positions into subclasses.

        Seeds subclasses with an optimal 1:1 assignment between the groups,
        then attaches each remaining majority-group unit to the subclass of
        its nearest feasible partner. Units with no feasible partner stay
        unmatched. Returns a list of {'treated': [...], 'control': [...]}
        with positional indices.
        """
        import warnings

        # Work in an orientation where rows are the smaller group, so the
        # seeding assigns one column unit to every row unit
        transpose = n_t > n_c
        if transpose:
            D = dist_matrix.T
            F = feasible.T
        else:
            D = dist_matrix
            F = feasible
        n_rows, n_cols = D.shape

        if F.any():
            penalty = (D[F].max() + 1.0) * (n_rows + 1)
        else:
            warnings.warn("Full matching: no pair satisfies the caliper; all units unmatched.")
            return []

        cost = np.where(F, D, penalty)
        row_ind, col_ind = linear_sum_assignment(cost)

        clusters = []          # {'rows': [...], 'cols': [...]}
        cluster_of_row = {}
        cluster_of_col = {}
        deferred_rows = []

        for r, c in zip(row_ind, col_ind):
            if F[r, c]:
                cluster_of_row[r] = len(clusters)
                cluster_of_col[c] = len(clusters)
                clusters.append({'rows': [r], 'cols': [c]})
            else:
                deferred_rows.append(r)

        # In the transposed orientation rows are controls, so max_controls
        # caps cluster row counts there; otherwise it caps column counts
        max_rows = self.max_controls if transpose else None
        max_cols = self.max_controls if not transpose else None

        # Rows whose optimal partner was infeasible join the cluster of their
        # nearest feasible column unit (if any); otherwise they stay unmatched
        for r in deferred_rows:
            feas_cols = [c for c in np.where(F[r])[0] if c in cluster_of_col]
            if max_rows is not None:
                feas_cols = [
                    c for c in feas_cols
                    if len(clusters[cluster_of_col[c]]['rows']) < max_rows
                ]
            if not feas_cols:
                continue
            nearest = min(feas_cols, key=lambda c: D[r, c])
            cid = cluster_of_col[nearest]
            clusters[cid]['rows'].append(r)
            cluster_of_row[r] = cid

        # Attach remaining column units to their nearest feasible row's cluster
        remaining_cols = [c for c in range(n_cols) if c not in cluster_of_col]
        for c in remaining_cols:
            feas_rows = np.where(F[:, c])[0]
            feas_rows = [r for r in feas_rows if r in cluster_of_row]
            if not feas_rows:
                continue
            for r in sorted(feas_rows, key=lambda r: D[r, c]):
                cl = clusters[cluster_of_row[r]]
                if max_cols is not None and len(cl['cols']) >= max_cols:
                    continue
                cl['cols'].append(c)
                break

        if self.min_controls > 1:
            if transpose:
                self._merge_for_min_rows(clusters, D, warnings)
            else:
                self._steal_for_min_cols(clusters, D, F, warnings)

        # Translate back to treated/control orientation
        result = []
        for cl in clusters:
            if transpose:
                result.append({'treated': cl['cols'], 'control': cl['rows']})
            else:
                result.append({'treated': cl['rows'], 'control': cl['cols']})
        return result

    def _steal_for_min_cols(self, clusters, D, F, warnings):
        """Move controls (cols) from clusters with surplus into clusters below
        min_controls, choosing the closest feasible donor control."""
        for cl in clusters:
            while len(cl['cols']) < self.min_controls:
                donors = [
                    (D[cl['rows'][0], c], other, c)
                    for other in clusters
                    if other is not cl and len(other['cols']) > self.min_controls
                    for c in other['cols']
                    if all(F[r, c] for r in cl['rows'])
                ]
                if not donors:
                    warnings.warn(
                        "Full matching: could not satisfy min_controls_per_subclass "
                        "for every subclass."
                    )
                    return
                _, donor, c = min(donors, key=lambda d: d[0])
                donor['cols'].remove(c)
                cl['cols'].append(c)

    def _merge_for_min_rows(self, clusters, D, warnings):
        """When controls are rows (more treated than controls), satisfy
        min_controls by merging undersized clusters. Deficient clusters are
        paired with their nearest deficient peer first, so merges don't
        cascade into one giant subclass."""

        def cross_dist(a, b):
            # Nearest control-treated pair across the two clusters
            return min(
                [D[r, c] for r in a['rows'] for c in b['cols']]
                + [D[r, c] for r in b['rows'] for c in a['cols']]
            )

        while True:
            deficient = [cl for cl in clusters if 0 < len(cl['rows']) < self.min_controls]
            if not deficient or len(clusters) < 2:
                if deficient:
                    warnings.warn(
                        "Full matching: could not satisfy min_controls_per_subclass "
                        "for every subclass."
                    )
                return
            cl = deficient[0]
            partners = [o for o in deficient if o is not cl] or [
                o for o in clusters if o is not cl
            ]
            host = min(partners, key=lambda o: cross_dist(cl, o))
            host['rows'].extend(cl['rows'])
            host['cols'].extend(cl['cols'])
            clusters.remove(cl)


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

        if estimand == "ATC":
            # The control group becomes the focal group (see NearestNeighborMatcher)
            treatment = 1 - treatment

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

        # Positional arrays of the distance measure (index labels cannot be
        # used as positions: they differ after discard or with custom indexes)
        dist_t_vals = dist_c_vals = None
        if distance_measure is not None:
            dist_t_vals = distance_measure[treated_mask].values
            dist_c_vals = distance_measure[control_mask].values

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
                    if global_caliper_threshold is not None and dist_t_vals is not None:
                        ps_diff = abs(dist_t_vals[i] - dist_c_vals[neighbor_indices[i, j]])
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

        # Without replacement (or with a caliper), the nearest candidates may
        # be taken or invalid, so all controls must be considered
        if self.replace and global_caliper_threshold is None:
            k = max(min(len(X_c_final), self.ratio), 1)
        else:
            k = len(X_c_final)
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='auto')
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
                if global_caliper_threshold is not None and dist_t_vals is not None:
                    ps_diff = abs(dist_t_vals[i] - dist_c_vals[local_pos])
                    if ps_diff > global_caliper_threshold:
                        continue

                found.append(control_indices[local_pos])
                if not self.replace:
                    available_mask[local_pos] = False

            if found:
                matches[t_idx] = found

        return self._build_result(matches, treatment.index)


class CardinalityMatcher(BaseMatcher):
    """
    Implements Cardinality Matching via subset selection.
    Finds the largest possible subset of the data where treated and control
    groups satisfy user-specified balance constraints (on standardized mean
    differences).

    Solves the subset-selection problem exactly as a mixed-integer linear
    program (scipy.optimize.milp, scipy >= 1.9) using the linearized balance
    constraints of Zubizarreta, Paredes & Rosenbaum (2014). Falls back to a
    greedy removal heuristic — which does not guarantee maximality or that
    the balance constraints are met — when the MILP solver is unavailable
    or fails.
    """

    def __init__(self, tols: Optional[Dict[str, float]] = None,
                 std_tols: float = 0.1,
                 random_state: Optional[int] = None,
                 solver_time_limit: float = 60.0):
        """
        Args:
            tols: Covariate-specific balance tolerances (absolute mean diff).
                  e.g., {'age': 2.0, 'educ': 0.5}
            std_tols: Default tolerance on standardized mean difference for
                      all covariates. Default is 0.1 (10% of a SD).
            solver_time_limit: Time limit (seconds) for the MILP solver.
        """
        super().__init__(ratio=1, replace=False, random_state=random_state)
        self.tols = tols if tols is not None else {}
        self.std_tols = std_tols
        self.solver_time_limit = solver_time_limit

    @staticmethod
    def _milp_select(X, target, eps, time_limit):
        """
        Maximum-cardinality subset of rows of X whose mean is within eps
        (componentwise) of target. |mean(X_sel) - target| <= eps is
        linearized as sum_j z_j * (x_jk - target_k -/+ eps_k) <=/>= 0.
        Returns a boolean mask, or None if the solver is unavailable or
        produced no feasible solution.
        """
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
        except ImportError:
            return None

        n, p = X.shape
        rows, lb, ub = [], [], []
        for k in range(p):
            a = X[:, k] - target[k]
            rows.append(a - eps[k]); lb.append(-np.inf); ub.append(0.0)
            rows.append(a + eps[k]); lb.append(0.0); ub.append(np.inf)
        rows.append(np.ones(n)); lb.append(1.0); ub.append(n)

        try:
            res = milp(
                c=-np.ones(n),
                constraints=LinearConstraint(np.vstack(rows), lb, ub),
                integrality=np.ones(n),
                bounds=Bounds(0, 1),
                options={"time_limit": time_limit},
            )
        except Exception:
            return None

        if res.x is None:
            return None
        sel = res.x > 0.5
        if sel.sum() == 0:
            return None
        # A time-limit incumbent could be infeasible; verify before accepting
        if np.any(np.abs(X[sel].mean(axis=0) - target) > eps + 1e-8):
            return None
        return sel

    @staticmethod
    def _greedy_select(X, target, eps):
        """Fallback: iteratively drop the unit most responsible for the
        worst balance violation against the fixed target."""
        n = X.shape[0]
        sel = np.ones(n, dtype=bool)
        for _ in range(n - 1):
            means = X[sel].mean(axis=0)
            viol = np.abs(means - target) - eps
            if np.all(viol <= 0):
                break
            worst = np.argmax(viol)
            active = np.where(sel)[0]
            vals = X[active, worst]
            if means[worst] > target[worst]:
                remove = active[np.argmax(vals)]
            else:
                remove = active[np.argmin(vals)]
            sel[remove] = False
        return sel

    def _select(self, X, target, eps):
        import warnings
        sel = self._milp_select(X, target, eps, self.solver_time_limit)
        if sel is None:
            warnings.warn(
                "Cardinality matching MILP unavailable or found no feasible solution; "
                "falling back to a greedy heuristic. The result may not be maximal and "
                "balance constraints may be violated."
            )
            sel = self._greedy_select(X, target, eps)
        return sel

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

        # Tolerances in raw covariate units
        eps_raw = tolerances * pooled_std

        weights = pd.Series(0.0, index=treatment.index)
        subclasses = pd.Series(pd.NA, index=treatment.index)

        if estimand == "ATT":
            # Keep all treated; largest control subset balanced to treated means
            sel_c = self._select(X_c, X_t.mean(axis=0), eps_raw)
            selected_controls = control_indices[sel_c]

            weights.loc[treated_indices] = 1.0
            if sel_c.sum() > 0:
                weights.loc[selected_controls] = n_t / sel_c.sum()

            subclasses.loc[treated_indices] = 1
            subclasses.loc[selected_controls] = 1

        elif estimand == "ATC":
            # Mirror of ATT: keep all controls, select treated subset
            sel_t = self._select(X_t, X_c.mean(axis=0), eps_raw)
            selected_treated = treated_indices[sel_t]

            weights.loc[control_indices] = 1.0
            if sel_t.sum() > 0:
                weights.loc[selected_treated] = n_c / sel_t.sum()

            subclasses.loc[control_indices] = 1
            subclasses.loc[selected_treated] = 1

        elif estimand == "ATE":
            # Template matching: each group's subset is balanced to the
            # full-sample means within eps/2, which guarantees the SMD
            # between the selected groups is within the tolerance
            overall = np.vstack([X_t, X_c]).mean(axis=0)
            sel_t = self._select(X_t, overall, eps_raw / 2)
            sel_c = self._select(X_c, overall, eps_raw / 2)
            selected_treated = treated_indices[sel_t]
            selected_controls = control_indices[sel_c]

            n_sel_t = int(sel_t.sum())
            n_sel_c = int(sel_c.sum())
            if n_sel_t > 0 and n_sel_c > 0:
                n_total = n_sel_t + n_sel_c
                weights.loc[selected_treated] = n_total / n_sel_t
                weights.loc[selected_controls] = n_total / n_sel_c

            subclasses.loc[selected_treated] = 1
            subclasses.loc[selected_controls] = 1
        else:
            raise ValueError(f"Estimand '{estimand}' not supported for Cardinality Matching.")

        # No pairwise matches for cardinality (subset selection)
        return {}, weights, subclasses