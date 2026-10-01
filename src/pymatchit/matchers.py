# File: src/pymatchit/matchers.py

import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import pinv
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod


def _resolve_mahalanobis_covariates(
    covariates: pd.DataFrame, mahvars: Optional[List[str]]
) -> pd.DataFrame:
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
            raise ValueError(
                f"mahvars variable '{v}' not found among model covariates."
            )
        cols.extend(hits)
    return covariates[cols].select_dtypes(include=[np.number])


def _mahalanobis_vi(num_covs: pd.DataFrame) -> np.ndarray:
    """Inverse covariance matrix for Mahalanobis distance. Falls back to the
    identity (unscaled Euclidean distance), with a warning, if it cannot be computed."""
    try:
        return pinv(num_covs.cov().values)
    except Exception as e:
        warnings.warn(
            f"Could not invert the covariance matrix for Mahalanobis distance ({e}); "
            "falling back to unscaled Euclidean distance."
        )
        return np.eye(num_covs.shape[1])


def _antiexact_violations(anti_t: np.ndarray, anti_c: np.ndarray) -> np.ndarray:
    """Boolean (n_treated x n_control) matrix: True where any antiexact variable matches."""
    violations = np.zeros((anti_t.shape[0], anti_c.shape[0]), dtype=bool)
    for k in range(anti_t.shape[1]):
        violations |= anti_t[:, k : k + 1] == anti_c[:, k : k + 1].T
    return violations


def _resolve_m_order(m_order: Optional[str], has_scores: bool, estimand: str) -> str:
    """
    Resolves the matching order. As in R MatchIt, the default (None) is
    'largest' when a propensity score is available ('smallest' for the ATC,
    where the control group is focal) and 'data' otherwise.
    """
    if m_order is None:
        if not has_scores:
            return "data"
        return "smallest" if estimand == "ATC" else "largest"
    if m_order not in ("largest", "smallest", "random", "data"):
        raise ValueError(
            f"m_order must be 'largest', 'smallest', 'random' or 'data', got '{m_order}'."
        )
    if m_order in ("largest", "smallest") and not has_scores:
        raise ValueError(
            f"m_order='{m_order}' orders units by the propensity score, but none "
            "is estimated with this distance. Use 'data' or 'random'."
        )
    return m_order


def _matching_order(m_order: str, scores, n: int, rng) -> np.ndarray:
    """Positions of the focal units in the order they are to be matched."""
    if m_order == "largest":
        return np.argsort(scores)[::-1]
    if m_order == "smallest":
        return np.argsort(scores)
    if m_order == "random":
        return rng.permutation(n)
    return np.arange(n)


def _greedy_match(D: np.ndarray, order, ratio: int, replace: bool):
    """
    Greedy nearest neighbor matching on a (focal x other) distance matrix in
    which forbidden pairs are np.inf. Returns two aligned arrays of focal
    and other positions, one entry per matched pair.

    Without replacement, units are matched in `order` and in rounds: every
    focal unit gets its first match before any gets a second.
    """
    n_f, n_o = D.shape
    if replace:
        k = min(ratio, n_o)
        if k == 1:
            nearest = D.argmin(axis=1)[:, None]
        else:
            nearest = np.argsort(D, axis=1, kind="stable")[:, :k]
        f_pos = np.repeat(np.arange(n_f), k)
        o_pos = nearest.ravel()
        keep = np.isfinite(D[f_pos, o_pos])
        return f_pos[keep], o_pos[keep]

    used = np.zeros(n_o)  # np.inf once a unit is taken
    exhausted = np.zeros(n_f, dtype=bool)
    f_pos, o_pos = [], []
    for _ in range(ratio):
        for i in order:
            if exhausted[i]:
                continue
            d = D[i] + used
            j = d.argmin()
            if not np.isfinite(d[j]):
                exhausted[i] = True
                continue
            f_pos.append(i)
            o_pos.append(j)
            used[j] = np.inf
    return np.asarray(f_pos, dtype=int), np.asarray(o_pos, dtype=int)


class BaseMatcher(ABC):
    """
    Abstract Base Class for all matching algorithms.
    """

    def __init__(
        self, ratio: int = 1, replace: bool = False, random_state: Optional[int] = None
    ):
        self.ratio = ratio
        self.replace = replace
        self.random_state = random_state

    @abstractmethod
    def match(
        self,
        treatment: pd.Series,
        distance_measure: Optional[pd.Series] = None,
        covariates: Optional[pd.DataFrame] = None,
        estimand: str = "ATT",
        exact: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Tuple[Dict[int, List[int]], pd.Series, pd.Series]:
        pass

    def _build_result(
        self, matches: Dict[int, List[int]], all_indices: pd.Index
    ) -> Tuple[Dict, pd.Series, pd.Series]:
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

    def __init__(
        self,
        ratio: int = 1,
        replace: bool = False,
        caliper: Optional[Union[float, Dict[str, float]]] = None,
        m_order: Optional[str] = None,
        random_state: Optional[int] = None,
        mahalanobis: bool = False,
        mahvars: Optional[List[str]] = None,
    ):
        super().__init__(ratio=ratio, replace=replace, random_state=random_state)
        self.caliper = caliper
        self.m_order = m_order
        self.mahalanobis = mahalanobis
        self.mahvars = mahvars

    def match(
        self,
        treatment,
        distance_measure=None,
        covariates=None,
        estimand="ATT",
        exact=None,
        antiexact=None,
        **kwargs,
    ):
        if estimand == "ATC":
            # The control group becomes the focal group: controls are matched
            # to treated units (mirrors R MatchIt's focal-group switch).
            treatment = 1 - treatment
        if exact is not None:
            return self._match_stratified(
                treatment, distance_measure, covariates, estimand, exact, antiexact
            )
        return self._match_global(
            treatment, distance_measure, covariates, estimand, antiexact
        )

    def _match_stratified(
        self,
        treatment,
        distance_measure,
        covariates,
        estimand,
        exact_df,
        antiexact=None,
    ):
        stratification_data = exact_df.copy()
        group_cols = list(exact_df.columns)
        grouped = stratification_data.groupby(group_cols)
        all_matches = {}
        # Caliper width is defined on the full sample's distance SD, not per stratum
        dist_std = distance_measure.std() if distance_measure is not None else None

        for _, group_indices in grouped.groups.items():
            local_treat = treatment.loc[group_indices]
            if local_treat.sum() == 0 or (local_treat == 0).sum() == 0:
                continue

            local_dist = (
                distance_measure.loc[group_indices]
                if distance_measure is not None
                else None
            )
            local_covs = (
                covariates.loc[group_indices] if covariates is not None else None
            )
            local_anti = antiexact.loc[group_indices] if antiexact is not None else None

            matches, _, _ = self._match_global(
                local_treat,
                local_dist,
                local_covs,
                estimand,
                local_anti,
                dist_std=dist_std,
            )
            all_matches.update(matches)

        return self._build_result(all_matches, treatment.index)

    def _match_global(
        self,
        treatment,
        distance_measure,
        covariates,
        estimand,
        antiexact=None,
        dist_std=None,
    ):
        treated_mask = treatment == 1
        control_mask = treatment == 0
        treated_indices = treatment[treated_mask].index.to_numpy()
        control_indices = treatment[control_mask].index.to_numpy()

        global_caliper = None
        cov_calipers = {}

        # Parse Caliper input
        if isinstance(self.caliper, dict):
            global_caliper = self.caliper.get("distance", None)
            cov_calipers = {k: v for k, v in self.caliper.items() if k != "distance"}
        elif self.caliper is not None:
            global_caliper = self.caliper

        if dist_std is None and distance_measure is not None:
            dist_std = distance_measure.std()

        if self.mahalanobis:
            if covariates is None:
                raise ValueError("Covariates required for Mahalanobis matching.")
            # For Mahalanobis, we only use numeric dummies, not the combined 'active_covs' frame
            num_covs = _resolve_mahalanobis_covariates(covariates, self.mahvars)
            X_treated = num_covs[treated_mask].values
            X_control = num_covs[control_mask].values

            VI = _mahalanobis_vi(num_covs)

            metric = "mahalanobis"
            metric_params = {"VI": VI}
            threshold = np.inf

            if global_caliper is not None:
                if distance_measure is None:
                    raise ValueError("Caliper threshold requires 1D distance measure.")
                threshold = global_caliper * dist_std
        else:
            if distance_measure is None:
                raise ValueError(
                    "Distance measure required for nearest neighbor matching."
                )
            X_treated = distance_measure[treated_mask].values.reshape(-1, 1)
            X_control = distance_measure[control_mask].values.reshape(-1, 1)
            metric = "euclidean"
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

        # Units are matched in order of the propensity score when one is
        # available, also when the matching itself is on a Mahalanobis distance
        order_scores = (
            distance_measure[treated_mask].values
            if distance_measure is not None
            else None
        )
        m_order = _resolve_m_order(self.m_order, order_scores is not None, estimand)

        if self.replace:
            matches = self._match_with_replacement(
                X_treated,
                X_control,
                treated_indices,
                control_indices,
                threshold,
                metric,
                metric_params,
                covs_treated_caliper,
                covs_control_caliper,
                cov_thresholds_mapped,
                anti_treated,
                anti_control,
            )
        else:
            matches = self._match_without_replacement(
                X_treated,
                X_control,
                treated_indices,
                control_indices,
                threshold,
                metric,
                metric_params,
                covs_treated_caliper,
                covs_control_caliper,
                cov_thresholds_mapped,
                anti_treated,
                anti_control,
                m_order,
                order_scores,
            )

        return self._build_result(matches, treatment.index)

    @staticmethod
    def _violates_pair_constraints(
        i,
        local_pos,
        covs_treated,
        covs_control,
        cov_thresholds,
        anti_treated,
        anti_control,
    ):
        if cov_thresholds:
            for col_idx, limit in cov_thresholds.items():
                if (
                    abs(covs_treated[i, col_idx] - covs_control[local_pos, col_idx])
                    > limit
                ):
                    return True
        if anti_treated is not None:
            if (anti_treated[i] == anti_control[local_pos]).any():
                return True
        return False

    def _match_with_replacement(
        self,
        X_treated,
        X_control,
        treated_indices,
        control_indices,
        threshold,
        metric,
        metric_params,
        covs_treated,
        covs_control,
        cov_thresholds,
        anti_treated=None,
        anti_control=None,
    ):
        if len(X_control) == 0:
            return {}

        # With covariate calipers or antiexact constraints, the closest match
        # may be invalid, so we must fetch all candidates
        has_pair_constraints = bool(cov_thresholds) or anti_treated is not None
        n_neighbors_to_fetch = (
            len(X_control) if has_pair_constraints else min(len(X_control), self.ratio)
        )

        nn = NearestNeighbors(
            n_neighbors=n_neighbors_to_fetch,
            metric=metric,
            metric_params=metric_params,
            algorithm="auto",
        )
        nn.fit(X_control)
        dists, neighbor_indices = nn.kneighbors(X_treated)

        matches = {}
        for i, t_idx in enumerate(treated_indices):
            valid_neighbors = []
            for j in range(dists.shape[1]):
                dist = dists[i, j]
                if dist > threshold:
                    break  # Break early since distances are sorted

                local_pos = neighbor_indices[i, j]

                if self._violates_pair_constraints(
                    i,
                    local_pos,
                    covs_treated,
                    covs_control,
                    cov_thresholds,
                    anti_treated,
                    anti_control,
                ):
                    continue

                valid_neighbors.append(control_indices[local_pos])
                if len(valid_neighbors) >= self.ratio:
                    break

            if len(valid_neighbors) > 0:
                matches[t_idx] = valid_neighbors
        return matches

    def _match_without_replacement(
        self,
        X_treated,
        X_control,
        treated_indices,
        control_indices,
        threshold,
        metric,
        metric_params,
        covs_treated,
        covs_control,
        cov_thresholds,
        anti_treated=None,
        anti_control=None,
        m_order="data",
        order_scores=None,
    ):
        if len(X_control) == 0:
            return {}

        rng = np.random.RandomState(self.random_state)
        sort_order = _matching_order(m_order, order_scores, len(X_treated), rng)

        matches = {}
        available_mask = np.ones(len(X_control), dtype=bool)

        n_neighbors_to_fetch = len(X_control)
        nn = NearestNeighbors(
            n_neighbors=n_neighbors_to_fetch, metric=metric, metric_params=metric_params
        )
        nn.fit(X_control)
        dists, neighbors = nn.kneighbors(X_treated, n_neighbors=n_neighbors_to_fetch)

        # With ratio > 1, matching proceeds in rounds (as in R MatchIt): every
        # unit gets its first match before any unit gets a second, so scarce
        # controls are not used up by the units that happen to come first.
        # next_pos[i] is where unit i resumes scanning its neighbor list;
        # candidates skipped earlier stay unavailable or invalid.
        next_pos = np.zeros(len(X_treated), dtype=int)
        exhausted = np.zeros(len(X_treated), dtype=bool)

        for _ in range(self.ratio):
            for i in sort_order:
                if exhausted[i]:
                    continue
                t_idx = treated_indices[i]
                matched = False

                for j in range(next_pos[i], n_neighbors_to_fetch):
                    if dists[i, j] > threshold:
                        break
                    local_pos = neighbors[i, j]
                    if not available_mask[local_pos]:
                        continue

                    if self._violates_pair_constraints(
                        i,
                        local_pos,
                        covs_treated,
                        covs_control,
                        cov_thresholds,
                        anti_treated,
                        anti_control,
                    ):
                        continue

                    matches.setdefault(t_idx, []).append(control_indices[local_pos])
                    available_mask[local_pos] = False
                    next_pos[i] = j + 1
                    matched = True
                    break

                if not matched:
                    exhausted[i] = True
        return matches


class OptimalMatcher(BaseMatcher):
    """
    Implements Optimal Matching minimizing the total global distance.
    Supports Covariate-Specific Calipers.
    """

    def __init__(
        self,
        ratio: int = 1,
        caliper: Optional[Union[float, Dict[str, float]]] = None,
        random_state: Optional[int] = None,
        mahalanobis: bool = False,
        mahvars: Optional[List[str]] = None,
    ):
        super().__init__(ratio=ratio, replace=False, random_state=random_state)
        self.caliper = caliper
        self.mahalanobis = mahalanobis
        self.mahvars = mahvars

    def match(
        self,
        treatment,
        distance_measure=None,
        covariates=None,
        estimand="ATT",
        exact=None,
        antiexact=None,
        **kwargs,
    ):
        if estimand == "ATC":
            # The control group becomes the focal group (see NearestNeighborMatcher)
            treatment = 1 - treatment
        if exact is not None:
            return self._match_stratified(
                treatment, distance_measure, covariates, estimand, exact, antiexact
            )
        return self._match_global(
            treatment, distance_measure, covariates, estimand, antiexact
        )

    def _match_stratified(
        self,
        treatment,
        distance_measure,
        covariates,
        estimand,
        exact_df,
        antiexact=None,
    ):
        stratification_data = exact_df.copy()
        group_cols = list(exact_df.columns)
        grouped = stratification_data.groupby(group_cols)
        all_matches = {}
        # Caliper width is defined on the full sample's distance SD, not per stratum
        dist_std = distance_measure.std() if distance_measure is not None else None

        for _, group_indices in grouped.groups.items():
            local_treat = treatment.loc[group_indices]
            if local_treat.sum() == 0 or (local_treat == 0).sum() == 0:
                continue

            local_dist = (
                distance_measure.loc[group_indices]
                if distance_measure is not None
                else None
            )
            local_covs = (
                covariates.loc[group_indices] if covariates is not None else None
            )
            local_anti = antiexact.loc[group_indices] if antiexact is not None else None

            matches, _, _ = self._match_global(
                local_treat,
                local_dist,
                local_covs,
                estimand,
                local_anti,
                dist_std=dist_std,
            )
            all_matches.update(matches)

        return self._build_result(all_matches, treatment.index)

    def _match_global(
        self,
        treatment,
        distance_measure,
        covariates,
        estimand,
        antiexact=None,
        dist_std=None,
    ):
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
            global_caliper = self.caliper.get("distance", None)
            cov_calipers = {k: v for k, v in self.caliper.items() if k != "distance"}
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
            if covariates is None:
                raise ValueError("Covariates required for Mahalanobis matching.")
            num_covs = _resolve_mahalanobis_covariates(covariates, self.mahvars)
            X_t = num_covs[treated_mask].values
            X_c = num_covs[control_mask].values
            VI = _mahalanobis_vi(num_covs)
            dist_matrix = cdist(X_t, X_c, metric="mahalanobis", VI=VI)

            if global_caliper is not None:
                if distance_measure is None:
                    raise ValueError("Caliper requires 1D distance measure.")
                ps_t = distance_measure[treated_mask].values.reshape(-1, 1)
                ps_c = distance_measure[control_mask].values.reshape(-1, 1)
                ps_dist = cdist(ps_t, ps_c, metric="euclidean")
                threshold = global_caliper * dist_std
                forbidden |= ps_dist > threshold
        else:
            if distance_measure is None:
                raise ValueError("Distance measure required.")
            X_t = distance_measure[treated_mask].values.reshape(-1, 1)
            X_c = distance_measure[control_mask].values.reshape(-1, 1)
            dist_matrix = cdist(X_t, X_c, metric="euclidean")

            if global_caliper is not None:
                threshold = global_caliper * dist_std
                forbidden |= dist_matrix > threshold

        # Apply covariate-specific calipers
        if cov_limits is not None:
            for col_idx, limit in enumerate(cov_limits):
                diffs = np.abs(
                    covs_t_caliper[:, col_idx : col_idx + 1]
                    - covs_c_caliper[:, col_idx : col_idx + 1].T
                )
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
            if forbidden[r, c]:
                continue
            t_idx = expanded_treated_indices[r]
            c_idx = control_indices[c]

            if t_idx not in matches:
                matches[t_idx] = []
            matches[t_idx].append(c_idx)

        return self._build_result(matches, treatment.index)


class ExactMatcher(BaseMatcher):
    def match(self, treatment, covariates, estimand="ATT", **kwargs):
        if covariates is None:
            raise ValueError("Covariates are required for Exact Matching.")
        work_data = covariates.copy()
        work_data["__treat__"] = treatment.values
        work_data["__original_index__"] = treatment.index

        grouped = work_data.groupby(list(covariates.columns))
        matches = {}
        weights = pd.Series(0.0, index=treatment.index)
        subclasses = pd.Series(pd.NA, index=treatment.index)
        group_id = 1

        for _, group in grouped:
            treated_in_group = group[group["__treat__"] == 1]
            control_in_group = group[group["__treat__"] == 0]
            n_treat = len(treated_in_group)
            n_control = len(control_in_group)

            if n_treat > 0 and n_control > 0:
                t_indices = treated_in_group["__original_index__"].tolist()
                c_indices = control_in_group["__original_index__"].tolist()
                for t_idx in t_indices:
                    matches[t_idx] = c_indices

                subclasses.loc[treated_in_group["__original_index__"]] = group_id
                subclasses.loc[control_in_group["__original_index__"]] = group_id
                group_id += 1

                if estimand == "ATT":
                    weights.loc[treated_in_group["__original_index__"]] = 1.0
                    weights.loc[control_in_group["__original_index__"]] = (
                        n_treat / n_control
                    )
                elif estimand == "ATE":
                    n_total = n_treat + n_control
                    weights.loc[treated_in_group["__original_index__"]] = (
                        n_total / n_treat
                    )
                    weights.loc[control_in_group["__original_index__"]] = (
                        n_total / n_control
                    )
                elif estimand == "ATC":
                    weights.loc[control_in_group["__original_index__"]] = 1.0
                    weights.loc[treated_in_group["__original_index__"]] = (
                        n_control / n_treat
                    )

        return matches, weights, subclasses


class SubclassMatcher(BaseMatcher):
    def __init__(self, n_subclasses: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.n_subclasses = n_subclasses

    def match(self, treatment, distance_measure, estimand="ATT", **kwargs):
        if distance_measure is None:
            raise ValueError("Propensity Scores required for Subclassification.")
        # Bin edges come from the focal group's score distribution (R MatchIt:
        # treated for ATT, control for ATC, everyone for ATE)
        if estimand == "ATC":
            ref_scores = distance_measure[treatment == 0]
        elif estimand == "ATE":
            ref_scores = distance_measure
        else:
            ref_scores = distance_measure[treatment == 1]
        _, bins = pd.qcut(
            ref_scores, q=self.n_subclasses, retbins=True, duplicates="drop"
        )
        bins[0], bins[-1] = -np.inf, np.inf

        subclass_labels = pd.cut(
            distance_measure, bins=bins, labels=False, include_lowest=True
        )
        weights = pd.Series(0.0, index=treatment.index)
        subclasses = pd.Series(pd.NA, index=treatment.index)
        unique_bins = np.unique(subclass_labels.dropna())

        for bin_idx in unique_bins:
            in_bin = subclass_labels == bin_idx
            n_treated = np.sum((treatment == 1) & in_bin)
            n_control = np.sum((treatment == 0) & in_bin)
            if n_treated == 0 or n_control == 0:
                continue

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
    def __init__(
        self, cutpoints: Optional[Dict[str, Union[int, List[float]]]] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.cutpoints = cutpoints

    def match(self, treatment, covariates, estimand="ATT", **kwargs):
        if covariates is None:
            raise ValueError("Covariates required for CEM.")
        coarsened = covariates.copy()
        numeric_cols = coarsened.select_dtypes(include=[np.number]).columns

        # Sturges' rule: ceil(log2(n) + 1). Matches the default in R's `cem`
        # package (via nclass.Sturges) and produces meaningfully more strata
        # than a fixed default on realistic sample sizes.
        n_obs = len(coarsened)
        sturges_bins = int(np.ceil(np.log2(n_obs) + 1)) if n_obs > 1 else 1

        for col in numeric_cols:
            if coarsened[col].nunique() <= 2:
                continue
            cuts = (
                self.cutpoints[col]
                if (self.cutpoints and col in self.cutpoints)
                else sturges_bins
            )
            try:
                coarsened[col] = pd.cut(
                    coarsened[col], bins=cuts, labels=False, include_lowest=True
                )
            except ValueError:
                pass

        work_data = coarsened.copy()
        work_data["__treat__"] = treatment.values
        work_data["__original_index__"] = treatment.index
        grouped = work_data.groupby(list(coarsened.columns))

        matches = {}
        weights = pd.Series(0.0, index=treatment.index)
        subclasses = pd.Series(pd.NA, index=treatment.index)
        group_id = 1

        for _, group in grouped:
            treated_in_group = group[group["__treat__"] == 1]
            control_in_group = group[group["__treat__"] == 0]
            n_treat = len(treated_in_group)
            n_control = len(control_in_group)

            if n_treat > 0 and n_control > 0:
                t_indices = treated_in_group["__original_index__"].tolist()
                c_indices = control_in_group["__original_index__"].tolist()
                for t_idx in t_indices:
                    matches[t_idx] = c_indices

                subclasses.loc[treated_in_group["__original_index__"]] = group_id
                subclasses.loc[control_in_group["__original_index__"]] = group_id
                group_id += 1

                if estimand == "ATT":
                    weights.loc[treated_in_group["__original_index__"]] = 1.0
                    weights.loc[control_in_group["__original_index__"]] = (
                        n_treat / n_control
                    )
                elif estimand == "ATE":
                    n_total = n_treat + n_control
                    weights.loc[treated_in_group["__original_index__"]] = (
                        n_total / n_treat
                    )
                    weights.loc[control_in_group["__original_index__"]] = (
                        n_total / n_control
                    )
                elif estimand == "ATC":
                    weights.loc[control_in_group["__original_index__"]] = 1.0
                    weights.loc[treated_in_group["__original_index__"]] = (
                        n_control / n_treat
                    )

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

    def __init__(
        self,
        caliper: Optional[Union[float, Dict[str, float]]] = None,
        min_controls_per_subclass: int = 1,
        max_controls_per_subclass: Optional[int] = None,
        random_state: Optional[int] = None,
        mahalanobis: bool = False,
        mahvars: Optional[List[str]] = None,
    ):
        super().__init__(ratio=1, replace=False, random_state=random_state)
        self.caliper = caliper
        self.min_controls = min_controls_per_subclass
        self.max_controls = max_controls_per_subclass
        self.mahalanobis = mahalanobis
        self.mahvars = mahvars

    def match(
        self,
        treatment,
        distance_measure=None,
        covariates=None,
        estimand="ATT",
        exact=None,
        antiexact=None,
        **kwargs,
    ):
        if exact is not None:
            return self._match_stratified(
                treatment, distance_measure, covariates, estimand, exact, antiexact
            )
        return self._match_global(
            treatment, distance_measure, covariates, estimand, antiexact
        )

    def _match_stratified(
        self,
        treatment,
        distance_measure,
        covariates,
        estimand,
        exact_df,
        antiexact=None,
    ):
        group_cols = list(exact_df.columns)
        grouped = exact_df.groupby(group_cols)

        all_weights = pd.Series(0.0, index=treatment.index)
        all_subclasses = pd.Series(pd.NA, index=treatment.index)
        subclass_offset = 0
        # Caliper width is defined on the full sample's distance SD, not per stratum
        dist_std = distance_measure.std() if distance_measure is not None else None

        for _, group_indices in grouped.groups.items():
            local_treat = treatment.loc[group_indices]
            if local_treat.sum() == 0 or (local_treat == 0).sum() == 0:
                continue

            local_dist = (
                distance_measure.loc[group_indices]
                if distance_measure is not None
                else None
            )
            local_covs = (
                covariates.loc[group_indices] if covariates is not None else None
            )
            local_anti = antiexact.loc[group_indices] if antiexact is not None else None

            _, w, sc = self._match_global(
                local_treat,
                local_dist,
                local_covs,
                estimand,
                local_anti,
                dist_std=dist_std,
            )

            all_weights.update(w[w > 0])
            # Offset subclass IDs to keep them unique across strata
            sc_valid = sc.dropna()
            if len(sc_valid) > 0:
                sc_valid = sc_valid.astype(int) + subclass_offset
                subclass_offset = sc_valid.max() + 1
                all_subclasses.update(sc_valid)

        return {}, all_weights, all_subclasses

    def _match_global(
        self,
        treatment,
        distance_measure,
        covariates,
        estimand,
        antiexact=None,
        dist_std=None,
    ):
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
            VI = _mahalanobis_vi(num_covs)
            dist_matrix = cdist(X_t, X_c, metric="mahalanobis", VI=VI)
        else:
            if distance_measure is None:
                raise ValueError("Distance measure required for Full Matching.")
            X_t = distance_measure[treated_mask].values.reshape(-1, 1)
            X_c = distance_measure[control_mask].values.reshape(-1, 1)
            dist_matrix = cdist(X_t, X_c, metric="euclidean")

        # Feasibility: pairs outside a caliper, or sharing an antiexact value,
        # cannot be in the same subclass
        feasible = np.ones((n_t, n_c), dtype=bool)
        if antiexact is not None:
            feasible &= ~_antiexact_violations(
                antiexact.loc[treated_mask].values, antiexact.loc[control_mask].values
            )
        if self.caliper is not None:
            if isinstance(self.caliper, dict):
                global_cal = self.caliper.get("distance", None)
                cov_calipers = {
                    k: v for k, v in self.caliper.items() if k != "distance"
                }
            else:
                global_cal = self.caliper
                cov_calipers = {}

            if global_cal is not None:
                if distance_measure is None:
                    raise ValueError("Caliper requires 1D distance measure.")
                if dist_std is None:
                    dist_std = distance_measure.std()
                threshold = global_cal * dist_std
                ps_t = distance_measure[treated_mask].values.reshape(-1, 1)
                ps_c = distance_measure[control_mask].values.reshape(-1, 1)
                ps_dist = cdist(ps_t, ps_c, metric="euclidean")
                feasible &= ps_dist <= threshold

            # Covariate-specific calipers
            for name, limit in cov_calipers.items():
                if covariates is None or name not in covariates.columns:
                    raise ValueError(f"Caliper variable '{name}' not found in data.")
                v_t = covariates.loc[treated_mask, name].values.reshape(-1, 1)
                v_c = covariates.loc[control_mask, name].values.reshape(-1, 1)
                feasible &= np.abs(v_t - v_c.T) <= limit

        clusters = self._build_clusters(dist_matrix, feasible, n_t, n_c)

        # Compute weights per subclass
        weights = pd.Series(0.0, index=treatment.index)
        subclasses = pd.Series(pd.NA, index=treatment.index)

        for sc_num, members in enumerate(clusters, start=1):
            t_list = [treated_indices[i] for i in members["treated"]]
            c_list = [control_indices[j] for j in members["control"]]
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
            warnings.warn(
                "Full matching: no pair satisfies the caliper/antiexact "
                "constraints; all units unmatched."
            )
            return []

        cost = np.where(F, D, penalty)
        row_ind, col_ind = linear_sum_assignment(cost)

        clusters = []  # {'rows': [...], 'cols': [...]}
        cluster_of_row = {}
        cluster_of_col = {}
        deferred_rows = []

        for r, c in zip(row_ind, col_ind):
            if F[r, c]:
                cluster_of_row[r] = len(clusters)
                cluster_of_col[c] = len(clusters)
                clusters.append({"rows": [r], "cols": [c]})
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
                    c
                    for c in feas_cols
                    if len(clusters[cluster_of_col[c]]["rows"]) < max_rows
                ]
            if not feas_cols:
                continue
            nearest = min(feas_cols, key=lambda c: D[r, c])
            cid = cluster_of_col[nearest]
            clusters[cid]["rows"].append(r)
            cluster_of_row[r] = cid

        # Attach remaining column units to their nearest feasible row's cluster.
        # The unit must be feasible with every row unit already in that
        # cluster, so the constraints hold for all pairs within a subclass.
        remaining_cols = [c for c in range(n_cols) if c not in cluster_of_col]
        for c in remaining_cols:
            feas_rows = np.where(F[:, c])[0]
            feas_rows = [r for r in feas_rows if r in cluster_of_row]
            if not feas_rows:
                continue
            for r in sorted(feas_rows, key=lambda r: D[r, c]):
                cl = clusters[cluster_of_row[r]]
                if max_cols is not None and len(cl["cols"]) >= max_cols:
                    continue
                if not all(F[other, c] for other in cl["rows"]):
                    continue
                cl["cols"].append(c)
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
                result.append({"treated": cl["cols"], "control": cl["rows"]})
            else:
                result.append({"treated": cl["rows"], "control": cl["cols"]})
        return result

    def _steal_for_min_cols(self, clusters, D, F, warnings):
        """Move controls (cols) from clusters with surplus into clusters below
        min_controls, choosing the closest feasible donor control."""
        for cl in clusters:
            while len(cl["cols"]) < self.min_controls:
                donors = [
                    (D[cl["rows"][0], c], other, c)
                    for other in clusters
                    if other is not cl and len(other["cols"]) > self.min_controls
                    for c in other["cols"]
                    if all(F[r, c] for r in cl["rows"])
                ]
                if not donors:
                    warnings.warn(
                        "Full matching: could not satisfy min_controls_per_subclass "
                        "for every subclass."
                    )
                    return
                _, donor, c = min(donors, key=lambda d: d[0])
                donor["cols"].remove(c)
                cl["cols"].append(c)

    def _merge_for_min_rows(self, clusters, D, warnings):
        """When controls are rows (more treated than controls), satisfy
        min_controls by merging undersized clusters. Deficient clusters are
        paired with their nearest deficient peer first, so merges don't
        cascade into one giant subclass."""

        def cross_dist(a, b):
            # Nearest control-treated pair across the two clusters
            return min(
                [D[r, c] for r in a["rows"] for c in b["cols"]]
                + [D[r, c] for r in b["rows"] for c in a["cols"]]
            )

        while True:
            deficient = [
                cl for cl in clusters if 0 < len(cl["rows"]) < self.min_controls
            ]
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
            host["rows"].extend(cl["rows"])
            host["cols"].extend(cl["cols"])
            clusters.remove(cl)


class GeneticMatcher(BaseMatcher):
    """
    Implements Genetic Matching.
    Nearest neighbor matching on a generalized Mahalanobis distance: every
    matching variable is scaled by its standard deviation and given a
    weight, and an evolutionary search picks the weights that give the best
    covariate balance in the resulting matched sample.

    As in R MatchIt, the covariates play separate roles. Balance is always
    optimized on all covariates in the formula. The distance is computed on
    `mahvars` if given; otherwise on the covariates plus the propensity
    score (when one is estimated or supplied). `exact`, `antiexact` and
    calipers restrict which pairs are allowed.

    Based on Diamond & Sekhon (2013) 'Genetic Matching for Estimating Causal
    Effects: A General Multivariate Matching Method for Achieving Balance in
    Observational Studies'. Unlike R's Matching::GenMatch, balance is scored
    by standardized mean differences rather than p-values.
    """

    def __init__(
        self,
        ratio: int = 1,
        replace: bool = False,
        caliper: Optional[Union[float, Dict[str, float]]] = None,
        pop_size: int = 100,
        max_generations: int = 50,
        balance_metric: str = "smd_max",
        random_state: Optional[int] = None,
        m_order: Optional[str] = None,
        mahvars: Optional[List[str]] = None,
    ):
        super().__init__(ratio=ratio, replace=replace, random_state=random_state)
        self.caliper = caliper
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.balance_metric = balance_metric
        self.m_order = m_order
        self.mahvars = mahvars

    def match(
        self,
        treatment,
        distance_measure=None,
        covariates=None,
        estimand="ATT",
        exact=None,
        antiexact=None,
        **kwargs,
    ):
        if covariates is None:
            raise ValueError("Covariates are required for Genetic Matching.")

        if estimand == "ATC":
            # The control group becomes the focal group (see NearestNeighborMatcher)
            treatment = 1 - treatment

        num_covs = covariates.select_dtypes(include=[np.number])
        if num_covs.shape[1] == 0:
            raise ValueError(
                "Genetic Matching requires at least one numeric covariate."
            )

        treated_mask = treatment == 1
        control_mask = treatment == 0
        treated_indices = treatment[treated_mask].index.to_numpy()
        control_indices = treatment[control_mask].index.to_numpy()
        n_t = len(treated_indices)
        n_c = len(control_indices)

        if n_t == 0 or n_c == 0:
            return self._build_result({}, treatment.index)

        # Covariates whose balance is optimized: always the full formula
        B_t = num_covs[treated_mask].values.astype(float)
        B_c = num_covs[control_mask].values.astype(float)
        std_t = B_t.std(axis=0)
        std_t[std_t < 1e-9] = 1.0

        # Positional arrays of the distance measure (index labels cannot be
        # used as positions: they differ after discard or with custom indexes)
        dist_t_vals = dist_c_vals = None
        if distance_measure is not None:
            dist_t_vals = distance_measure[treated_mask].values
            dist_c_vals = distance_measure[control_mask].values

        # Variables that enter the generalized Mahalanobis distance
        if self.mahvars:
            match_vars = _resolve_mahalanobis_covariates(
                covariates, self.mahvars
            ).values.astype(float)
        else:
            match_vars = num_covs.values.astype(float)
            if distance_measure is not None:
                match_vars = np.column_stack([match_vars, distance_measure.values])
        scale = match_vars.std(axis=0)
        scale[scale < 1e-9] = 1.0
        Z = match_vars / scale
        Z_t = Z[treated_mask.values]
        Z_c = Z[control_mask.values]
        n_dims = Z.shape[1]

        # Parse caliper
        cov_calipers = {}
        if isinstance(self.caliper, dict):
            global_cal = self.caliper.get("distance", None)
            cov_calipers = {k: v for k, v in self.caliper.items() if k != "distance"}
        elif self.caliper is not None:
            global_cal = self.caliper
        else:
            global_cal = None

        # allowed[i, j]: treated i and control j may be paired (None when
        # nothing restricts the pairing)
        allowed = None

        def restrict(ok):
            nonlocal allowed
            allowed = ok if allowed is None else allowed & ok

        if global_cal is not None:
            if distance_measure is None:
                raise ValueError("Caliper requires 1D distance measure.")
            threshold = global_cal * distance_measure.std()
            restrict(np.abs(dist_t_vals[:, None] - dist_c_vals[None, :]) <= threshold)

        for name, limit in cov_calipers.items():
            if name not in covariates.columns:
                raise ValueError(f"Caliper variable '{name}' not found in data.")
            v_t = covariates.loc[treated_mask, name].values
            v_c = covariates.loc[control_mask, name].values
            restrict(np.abs(v_t[:, None] - v_c[None, :]) <= limit)

        if exact is not None:
            strata = exact.groupby(list(exact.columns), sort=False).ngroup()
            s_t = strata[treated_mask].values
            s_c = strata[control_mask].values
            restrict(s_t[:, None] == s_c[None, :])

        if antiexact is not None:
            restrict(
                ~_antiexact_violations(
                    antiexact.loc[treated_mask].values,
                    antiexact.loc[control_mask].values,
                )
            )

        rng = np.random.RandomState(self.random_state)
        m_order = _resolve_m_order(self.m_order, dist_t_vals is not None, estimand)
        order = _matching_order(m_order, dist_t_vals, n_t, rng)

        def match_positions(weight_vector):
            """Nearest neighbor matching under the given variable weights.
            Returns positional (treated, control) pairs."""
            sw = np.sqrt(np.abs(weight_vector))
            D = cdist(Z_t * sw, Z_c * sw, metric="euclidean")
            if allowed is not None:
                D[~allowed] = np.inf
            return _greedy_match(D, order, self.ratio, self.replace)

        def evaluate_weights(weight_vector):
            """Balance of the matched sample these weights produce (lower is better)."""
            t_pos, c_pos = match_positions(weight_vector)
            if len(t_pos) == 0:
                return 1e6

            # Same weights as the final result: each control gets 1/k_i per
            # match, where k_i is the number of matches of its treated unit
            k = np.bincount(t_pos, minlength=n_t)
            c_w = np.bincount(c_pos, weights=1.0 / k[t_pos], minlength=n_c)
            mean_t = B_t[k > 0].mean(axis=0)
            mean_c = c_w @ B_c / c_w.sum()

            smds = np.abs(mean_t - mean_c) / std_t

            if self.balance_metric == "smd_mean":
                return np.mean(smds)
            return np.max(smds)

        # --- Differential Evolution (simplified) ---
        # Initialize population
        population = rng.uniform(0.1, 2.0, size=(self.pop_size, n_dims))
        fitness = np.array([evaluate_weights(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_weights = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        mutation_factor = 0.8
        crossover_prob = 0.7

        for gen in range(self.max_generations):
            # Early stopping if balance is very good
            if best_fitness < 0.01:
                break

            for i in range(self.pop_size):
                # Mutation: DE/rand/1
                candidates = [j for j in range(self.pop_size) if j != i]
                a, b, c = rng.choice(candidates, 3, replace=False)
                mutant = population[a] + mutation_factor * (
                    population[b] - population[c]
                )
                mutant = np.clip(mutant, 0.01, 10.0)

                # Crossover
                cross_mask = rng.rand(n_dims) < crossover_prob
                if not cross_mask.any():
                    cross_mask[rng.randint(n_dims)] = True
                trial = np.where(cross_mask, mutant, population[i])

                # Selection
                trial_fitness = evaluate_weights(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_weights = trial.copy()
                        best_fitness = trial_fitness

        # --- Final matching with optimized weights ---
        t_pos, c_pos = match_positions(best_weights)
        matches = {}
        for i, j in zip(t_pos, c_pos):
            matches.setdefault(treated_indices[i], []).append(control_indices[j])

        return self._build_result(matches, treatment.index)


class CardinalityMatcher(BaseMatcher):
    """
    Implements cardinality and profile matching via subset selection.
    Finds the largest possible subset of the data in which the groups
    satisfy user-specified balance constraints (on standardized mean
    differences). Following R MatchIt, `ratio` selects the variant:

    - ratio=None (default), profile matching (Cohn & Zubizarreta 2022): for
      ATT/ATC the focal group is kept intact and the largest balanced subset
      of the other group is selected; for ATE each group's subset is
      balanced to the full sample.
    - ratio=k, cardinality matching (Zubizarreta, Paredes & Rosenbaum 2014):
      the largest sample with k non-focal units per focal unit whose groups
      are balanced. Focal units can be dropped, so the result no longer
      targets the ATT/ATC exactly.

    With `exact`, the optimization is solved separately within each stratum.
    With `mahvars` (and a whole-number ratio), the selected units are then
    optimally paired on the Mahalanobis distance of those variables, which
    leaves balance unchanged ("matching for balance, pairing for
    heterogeneity").

    Solved exactly as a mixed-integer linear program (scipy.optimize.milp,
    scipy >= 1.9). Profile matching falls back to a greedy removal heuristic,
    which guarantees neither maximality nor the balance constraints, when
    the solver is unavailable or fails.
    """

    def __init__(
        self,
        tols: Optional[Dict[str, float]] = None,
        std_tols: float = 0.1,
        random_state: Optional[int] = None,
        solver_time_limit: float = 60.0,
        ratio: Optional[int] = None,
        mahvars: Optional[List[str]] = None,
    ):
        """
        Args:
            tols: Covariate-specific balance tolerances (absolute mean diff).
                  e.g., {'age': 2.0, 'educ': 0.5}
            std_tols: Default tolerance on standardized mean difference for
                      all covariates. Default is 0.1 (10% of a SD). The
                      standardization factor is the one used by summary():
                      the focal group's SD for ATT/ATC, the pooled SD for ATE.
            solver_time_limit: Time limit (seconds) for the MILP solver.
            ratio: Non-focal units per focal unit; None for profile matching.
            mahvars: Variables to pair the selected units on (needs `ratio`).
        """
        super().__init__(ratio=ratio, replace=False, random_state=random_state)
        self.tols = tols if tols is not None else {}
        self.std_tols = std_tols
        self.solver_time_limit = solver_time_limit
        self.mahvars = mahvars

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
            rows.append(a - eps[k])
            lb.append(-np.inf)
            ub.append(0.0)
            rows.append(a + eps[k])
            lb.append(0.0)
            ub.append(np.inf)
        rows.append(np.ones(n))
        lb.append(1.0)
        ub.append(n)

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
    def _milp_fixed_ratio(X_f, X_o, eps, ratio, time_limit, target=None):
        """
        Largest subsets of the focal rows X_f and the other rows X_o with
        exactly `ratio` other units per focal unit.

        With target=None (cardinality matching) the two subsets are balanced
        against each other: |mean(X_f sel) - mean(X_o sel)| <= eps, which is
        linear given the fixed ratio. Otherwise (profile matching for the
        ATE) each subset's mean is within eps of `target`.
        Returns (focal mask, other mask), or None if no solution was found.
        """
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
        except ImportError:
            return None

        n_f, n_o = len(X_f), len(X_o)
        p = X_f.shape[1]
        # Work on standardized columns: equivalent constraints, better conditioned
        both = np.vstack([X_f, X_o])
        center = both.mean(axis=0) if target is None else np.asarray(target)
        scale = both.std(axis=0)
        scale[scale < 1e-9] = 1.0
        A_f = (X_f - center) / scale
        A_o = (X_o - center) / scale
        e = eps / scale
        zeros_f, zeros_o = np.zeros(n_f), np.zeros(n_o)

        rows, lb, ub = [], [], []
        # Group sizes: |other| = ratio * |focal|, and at least one focal unit
        rows.append(np.concatenate([-ratio * np.ones(n_f), np.ones(n_o)]))
        lb.append(0.0)
        ub.append(0.0)
        rows.append(np.concatenate([np.ones(n_f), zeros_o]))
        lb.append(1.0)
        ub.append(n_f)
        for k in range(p):
            if target is None:
                lower = [
                    np.concatenate([A_f[:, k] - e[k], -A_o[:, k] / ratio]),
                ]
                upper = [
                    np.concatenate([A_f[:, k] + e[k], -A_o[:, k] / ratio]),
                ]
            else:
                lower = [
                    np.concatenate([A_f[:, k] - e[k], zeros_o]),
                    np.concatenate([zeros_f, A_o[:, k] - e[k]]),
                ]
                upper = [
                    np.concatenate([A_f[:, k] + e[k], zeros_o]),
                    np.concatenate([zeros_f, A_o[:, k] + e[k]]),
                ]
            for row in lower:
                rows.append(row)
                lb.append(-np.inf)
                ub.append(0.0)
            for row in upper:
                rows.append(row)
                lb.append(0.0)
                ub.append(np.inf)

        try:
            res = milp(
                c=-np.ones(n_f + n_o),
                constraints=LinearConstraint(np.vstack(rows), lb, ub),
                integrality=np.ones(n_f + n_o),
                bounds=Bounds(0, 1),
                options={"time_limit": time_limit},
            )
        except Exception:
            return None

        if res.x is None:
            return None
        sel_f = res.x[:n_f] > 0.5
        sel_o = res.x[n_f:] > 0.5
        if sel_f.sum() == 0 or sel_o.sum() != ratio * sel_f.sum():
            return None
        # A time-limit incumbent could be infeasible; verify before accepting
        mean_f = X_f[sel_f].mean(axis=0)
        mean_o = X_o[sel_o].mean(axis=0)
        if target is None:
            ok = np.all(np.abs(mean_f - mean_o) <= eps + 1e-8)
        else:
            ok = np.all(np.abs(mean_f - target) <= eps + 1e-8) and np.all(
                np.abs(mean_o - target) <= eps + 1e-8
            )
        return (sel_f, sel_o) if ok else None

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
        sel = self._milp_select(X, target, eps, self.solver_time_limit)
        if sel is None:
            warnings.warn(
                "Cardinality matching MILP unavailable or found no feasible solution; "
                "falling back to a greedy heuristic. The result may not be maximal and "
                "balance constraints may be violated."
            )
            sel = self._greedy_select(X, target, eps)
        return sel

    def _select_stratum(self, X_f, X_o, eps, estimand):
        """Selects units within one stratum. Returns (focal mask, other mask),
        or None when a fixed-ratio problem has no solution."""
        if self.ratio is not None:
            target = None
            if estimand == "ATE":
                target = np.vstack([X_f, X_o]).mean(axis=0)
                eps = eps / 2
            return self._milp_fixed_ratio(
                X_f, X_o, eps, self.ratio, self.solver_time_limit, target=target
            )

        if estimand == "ATE":
            # Each group's subset is balanced to the full-sample means within
            # eps/2, which guarantees the SMD between the selected groups is
            # within the tolerance
            overall = np.vstack([X_f, X_o]).mean(axis=0)
            return (
                self._select(X_f, overall, eps / 2),
                self._select(X_o, overall, eps / 2),
            )

        # Keep the whole focal group; largest other subset balanced to its means
        return np.ones(len(X_f), dtype=bool), self._select(X_o, X_f.mean(axis=0), eps)

    def match(
        self,
        treatment,
        distance_measure=None,
        covariates=None,
        estimand="ATT",
        exact=None,
        **kwargs,
    ):
        if covariates is None:
            raise ValueError("Covariates are required for Cardinality Matching.")
        if estimand not in ("ATT", "ATC", "ATE"):
            raise ValueError(
                f"Estimand '{estimand}' not supported for Cardinality Matching."
            )
        if self.mahvars and self.ratio is None:
            raise ValueError(
                "mahvars can only be used with cardinality matching when ratio "
                "is a whole number: pairing needs a fixed number of matches per unit."
            )

        weights = pd.Series(0.0, index=treatment.index)
        subclasses = pd.Series(pd.NA, index=treatment.index)

        num_covs = covariates.select_dtypes(include=[np.number])
        X = num_covs.values.astype(float)
        is_treated = (treatment == 1).values
        if is_treated.sum() == 0 or (~is_treated).sum() == 0:
            return {}, weights, subclasses

        # The focal group defines the ratio and, for ATT/ATC, the target
        is_focal = ~is_treated if estimand == "ATC" else is_treated

        # Standardization factor, as in summary(): the focal group's SD for
        # ATT/ATC, the pooled SD for ATE
        if estimand == "ATE":
            sd = np.sqrt(
                (X[is_treated].var(axis=0, ddof=1) + X[~is_treated].var(axis=0, ddof=1))
                / 2
            )
        else:
            sd = X[is_focal].std(axis=0, ddof=1)
        sd = np.where(np.isnan(sd) | (sd < 1e-9), 1.0, sd)

        # Tolerances in raw covariate units; `tols` entries are absolute
        eps_raw = self.std_tols * sd
        for i, name in enumerate(num_covs.columns):
            if name in self.tols:
                eps_raw[i] = self.tols[name]

        if exact is not None:
            strata = [
                treatment.index.get_indexer(idx)
                for idx in exact.groupby(list(exact.columns)).groups.values()
            ]
        else:
            strata = [np.arange(len(treatment))]

        if self.mahvars:
            mah_covs = _resolve_mahalanobis_covariates(covariates, self.mahvars)
            mah_X = mah_covs.values.astype(float)
            VI = _mahalanobis_vi(mah_covs)

        labels = treatment.index.to_numpy()
        matches = {}
        n_failed = 0
        group_id = 1

        for pos in strata:
            f_pos = pos[is_focal[pos]]
            o_pos = pos[~is_focal[pos]]
            if len(f_pos) == 0 or len(o_pos) == 0:
                continue

            selected = self._select_stratum(X[f_pos], X[o_pos], eps_raw, estimand)
            if selected is None:
                n_failed += 1
                continue
            sel_f, sel_o = f_pos[selected[0]], o_pos[selected[1]]
            if len(sel_f) == 0 or len(sel_o) == 0:
                continue

            # Both groups' weights sum to the same total within a stratum
            if estimand == "ATE" and self.ratio is None:
                n_total = len(sel_f) + len(sel_o)
                weights.iloc[sel_f] = n_total / len(sel_f)
                weights.iloc[sel_o] = n_total / len(sel_o)
            else:
                weights.iloc[sel_f] = 1.0
                weights.iloc[sel_o] = len(sel_f) / len(sel_o)

            if not self.mahvars:
                subclasses.iloc[np.concatenate([sel_f, sel_o])] = group_id
                group_id += 1
                continue

            # Pair the selected units: optimal `ratio`:1 assignment on the
            # Mahalanobis distance of mahvars
            D = cdist(mah_X[sel_f], mah_X[sel_o], metric="mahalanobis", VI=VI)
            rows, cols = linear_sum_assignment(np.repeat(D, self.ratio, axis=0))
            for r in range(len(sel_f)):
                partners = sel_o[cols[rows // self.ratio == r]]
                matches[labels[sel_f[r]]] = list(labels[partners])
                subclasses.iloc[np.concatenate([[sel_f[r]], partners])] = group_id
                group_id += 1

        if n_failed:
            message = (
                "Cardinality matching found no balanced sample with "
                f"ratio={self.ratio}"
                + (
                    f" in {n_failed} of {len(strata)} strata"
                    if exact is not None
                    else ""
                )
                + ". Try larger tolerances (std_tols/tols) or a different ratio."
            )
            if not (weights > 0).any():
                raise ValueError(message)
            warnings.warn(message)

        return matches, weights, subclasses
