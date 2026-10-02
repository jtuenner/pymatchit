# Changelog

## 0.6.1

This release fixes two matching bugs. **Results change for full matching, and for nearest neighbor matching that combines `mahvars` with a caliper**; other methods are unaffected.

### Changed results

- **Full matching** is now optimal: it finds the subclasses with the smallest total distance between treated and control units, as R's `optmatch` does. A subclass is one treated unit with one or more controls, or one control with one or more treated units. Before, every unit of the smaller group was first paired with a unit of its own, which forced distant pairs when the groups overlap little: on the Lalonde data the standardized mean difference of `black` after matching was 1.02 and is now 0.02.
- **`min_controls_per_subclass` and `max_controls_per_subclass`** are part of the optimization instead of being applied afterwards. Controls that `max_controls_per_subclass` leaves no room for stay unmatched, as few as possible. If there are too few controls to give every treated unit `min_controls_per_subclass` of its own, subclasses are merged, now with a warning.
- **Nearest neighbor matching with `mahvars` and a caliper** applies the caliper to the propensity score. It was compared with the Mahalanobis distance instead, which left most units unmatched and allowed pairs that were far apart on the propensity score. Optimal, full and genetic matching were not affected.

### Fixed

- A caliper on the distance measure with `distance="mahalanobis"` (which estimates no propensity score), and invalid `min_controls_per_subclass` or `max_controls_per_subclass` values, raise an error that says what to change.

## 0.6.0

This release aligns pymatchit with R MatchIt (checked against version 4.8.1) and fixes several cases where options were silently ignored. **Results change for some methods**; see "Changed results" below before upgrading an existing analysis.

### New

- **Genetic matching** supports `exact`, `antiexact`, `mahvars` and `m_order`.
- **Full matching** supports `antiexact` and covariate-specific calipers (`caliper={"age": 2}`).
- **Cardinality matching** supports `exact` (each stratum is solved separately), `ratio` and `mahvars`:
  - `ratio=k` selects the largest balanced sample with k controls per treated unit. Leaving `ratio` unset keeps profile matching, where the whole treated group is retained.
  - `mahvars` pairs the selected units, and `matches()` returns the pairs.
- **GLM links** `cloglog` and `cauchit`, plus `linear.probit`. `linear` is accepted as an alias for `linear.logit`.
- **Optimal matching** supports `antiexact` and `mahvars`.
- `estimand="ATC"` is implemented for all methods. For nearest, optimal and genetic matching the control group becomes the focal group, and `matches()` is keyed by control unit.

### Unused options now warn

An option the chosen method cannot use (for example `caliper` with `method="subclass"`) is ignored with a warning that names the option, as in R. Before, most were ignored silently. This lets you switch `method` without rewriting the call. To turn these warnings into errors, use `warnings.simplefilter("error")`.

Contradictory requests raise an error: `estimand="ATE"` with nearest, optimal or genetic matching; an unknown link; `m_order="largest"` without a propensity score; `mahvars` with cardinality matching but no `ratio`.

### Changed results

- **`link`**: `logit` and `probit` now match on the predicted probability, as in R. Use `link="linear.logit"` for the previous behaviour (matching on the linear predictor). Calipers are in standard deviations of this measure, so their width changes too.
- **User-supplied distances** (`distance=` an array) are used as-is; they are no longer logit-transformed.
- **Nearest matching with `ratio > 1`** assigns matches in rounds: every unit gets a first match before any gets a second. Control weights are 1/k per match, where k is the number of matches the treated unit actually received.
- **Calipers with `exact=`** use the full-sample standard deviation, not each stratum's.
- **Matching order**: `m_order` defaults to `largest` when a propensity score is available (`smallest` for `ATC`) and to row order otherwise. `mahvars` matching is ordered by the propensity score.
- **Genetic matching** uses a weighted distance on standardized variables, including the propensity score, and optimizes the matching it actually returns.
- **CBPS** is the just-identified estimator, which balances the covariates exactly, and follows `estimand`.
- **Cardinality matching** is solved exactly as a mixed-integer program. `std_tols` uses the same standardization as `summary()`.
- **Full matching** seeds subclasses with an optimal 1:1 assignment and enforces calipers and `min`/`max_controls_per_subclass`.
- **Subclassification** bins on the focal group's scores (treated for ATT, control for ATC, everyone for ATE).
- **CEM** bins numeric covariates with Sturges' rule by default instead of 5 bins.
- **KS statistics** for the matched sample use the matching weights.

### Fixed

- Optimal matching crashed when a caliper excluded pairs.
- Genetic matching with a caliper crashed on non-default indexes and dropped most treated units when matching without replacement.
- `m_order="random"` reseeded NumPy's global random state.
- `matches()` returned missing values for string-labelled indexes.

### Signature

- `ratio` defaults to `None` (1:1 for pair matching; profile matching for cardinality). It was `1`.
- `m_order` defaults to `None`. It was `"largest"`.

### Development

- Tests run on Python 3.9–3.14 and on the oldest and newest supported pandas. The code is formatted with black.

Thanks to Nick Eubank (@nickeubank) for the bug review and the pull requests behind most of this release, and to @codeteme for the report on control weights.
