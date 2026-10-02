# pymatchit-causal: Propensity Score Matching in Python

[![Tests](https://github.com/jtuenner/pymatchit/actions/workflows/test.yml/badge.svg)](https://github.com/jtuenner/pymatchit/actions/workflows/test.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17839522.svg)](https://doi.org/10.5281/zenodo.17839522)
[![PyPI version](https://badge.fury.io/py/pymatchit-causal.svg)](https://badge.fury.io/py/pymatchit-causal)

**Scalable Causal Inference, Propensity Score Matching (PSM), and Coarsened Exact Matching (CEM).**

`pymatchit-causal` is a Python port of the standard R package `MatchIt`. It allows data scientists to preprocess data for causal inference by balancing covariates between treated and control groups using state-of-the-art matching methods. The 0.6 releases align its results and options with R `MatchIt` 4.8.1; if you are coming from an earlier version, read [Upgrading](#upgrading) first.

## Features
* **Matching Methods:** Nearest Neighbor (Greedy), Optimal Matching, Exact, Subclassification, Coarsened Exact Matching (CEM), Full Matching, Genetic Matching, and Cardinality Matching.
* **Propensity Score Estimation:** Logistic Regression (GLM) with logit, probit, cloglog, or cauchit links, CBPS, Random Forest, GBM, Neural Networks, Decision Trees, AdaBoost, Penalized approaches (Lasso / Ridge / ElasticNet), Mahalanobis distance, or user-supplied scores.
* **Advanced Configurations:** Target `ATT`, `ATC`, or `ATE`, discard units outside common support, combine Mahalanobis distance with a Propensity Score caliper (`mahvars`), and enforce exact (`exact`) or anti-exact (`antiexact`) matching on chosen variables.
* **Diagnostics:** Summary Tables (SMD, Variance Ratios, sample sizes) and publication-ready plots: Love Plots (Covariate Balance), Propensity Density Plots, Jitter Plots, ECDF plots, and QQ plots.
* **Parity:** Designed to mirror the R `MatchIt` API (`matchit(formula, data, method=...)`), including which options each method accepts.

---

## Installation

```bash
pip install pymatchit-causal
```

The package is installed as `pymatchit-causal` and imported as `pymatchit`. It requires Python 3.9 or newer.

Dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `scikit-learn`, `seaborn`, `patsy`.

---

## Example Workflow

**Scenario**: The Lalonde (1986) job training dataset ships with the package, so this example runs as-is. The treatment is `treat` (took part in the training programme), the outcome is `re78` (earnings in 1978), and the confounders are demographics and earlier earnings.

### 1. Load Data
```python
from pymatchit import MatchIt, load_lalonde

df = load_lalonde()  # 614 rows: 185 treated, 429 control
```

To use your own data, load it into a `pandas` DataFrame (see [Data Requirements](#data-requirements)).

### 2. Initialize and Match
We will use 1:1 **Nearest Neighbor** matching on a propensity score from a **logistic regression**, applying a **caliper** to ensure good matches.

```python
# Initialize the matching model
m = MatchIt(
    data=df,
    method='nearest',   # 1:1 Nearest Neighbor matching
    distance='glm',     # Logistic regression for Propensity Scores
    caliper=0.1,        # Max. distance: 0.1 standard deviations of the score
    random_state=42
)

# Fit the model using an R-style formula: treatment ~ covariates
m.fit("treat ~ age + educ + black + hispan + married + nodegree + re74 + re75")
```

The caliper leaves treated units without a close enough control unmatched: here 111 of the 185 treated units are matched.

### 3. Assess Balance (Diagnostics)
Verify that the treatment and control groups are balanced with cohesive visualization tools.

```python
# 1. Statistical Summary (prints the tables and returns them)
summary = m.summary()

# 2. Visual Inspection: Love Plot
m.plot(type='balance', threshold=0.1)
```
![Love Plot](https://raw.githubusercontent.com/jtuenner/pymatchit/main/assets/love_plot.png)

```python
# 3. Visual Inspection: Propensity Jitter Plot
m.plot(type='jitter')
```
![Jitter Plot](https://raw.githubusercontent.com/jtuenner/pymatchit/main/assets/jitter_plot.png)

```python
# 4. Visual Inspection: Propensity Density Overlap
m.plot(type='propensity')
```
![Propensity Density Plot](https://raw.githubusercontent.com/jtuenner/pymatchit/main/assets/propensity_plot.png)

```python
# 5. Visual Inspection: ECDF and QQ Plots for a single covariate
m.plot(type='ecdf', variable='age')
m.plot(type='qq', variable='age')
```

### 4. Extract Matched Data
If balance is satisfactory, extract the data for analysis.

```python
# Get the matched pairs: one row per (treated_index, control_index) pair
matched_pairs = m.matches(format='long')

# Get the matched units with their weights and subclass
final_analysis_set = m.matched_data
```

### 5. Downstream Inference
Estimate the effect on the matched sample. Always pass the matching `weights`; for pair matching, cluster the standard errors on the matched pair (`subclass`):

```python
import statsmodels.formula.api as smf

model = smf.wls("re78 ~ treat", data=final_analysis_set, weights=final_analysis_set['weights'])
results = model.fit(cov_type='cluster', cov_kwds={'groups': final_analysis_set['subclass']})
print(results.summary())
```

---

## Data Requirements

* **Treatment:** a binary column coded `0`/`1` (or `False`/`True`) without missing values.
* **Covariates:** no missing values. Impute or drop incomplete rows first.
* **Index:** the DataFrame index must be unique (use `df.reset_index(drop=True)` if it is not). `MatchIt` works on a copy and never modifies your DataFrame.
* **Formula:** `"treatment ~ cov1 + cov2 + ..."`. String and categorical columns are dummy-coded automatically, and `patsy` syntax such as `C(region)`, `age:severity`, or `I(age**2)` is supported. The outcome does not belong in the formula.

### A note on machine-learning propensity scores

The propensity score is predicted for the same rows the model was trained on. A flexible model with default settings (`randomforest`, `decisiontree`, `gbm`) can therefore separate the groups almost perfectly, which leaves no treated and control units with similar scores: a caliper then finds few or no matches. Start with `distance='glm'`, and if you use an ML model, regularize it through `distance_options` and check the overlap with `m.plot(type='propensity')`:

```python
m = MatchIt(
    data=df,
    distance='randomforest',
    distance_options={'n_estimators': 500, 'min_samples_leaf': 20},
    caliper=0.1,
    random_state=42
)
```

---

## More Examples

All examples use the Lalonde data and formula from above:

```python
formula = "treat ~ age + educ + black + hispan + married + nodegree + re74 + re75"
```

**2:1 matching with replacement**
```python
m = MatchIt(df, method='nearest', ratio=2, replace=True).fit(formula)
```

**Mahalanobis matching inside a propensity score caliper, exact on a variable**
```python
m = MatchIt(
    df,
    method='nearest',                         # or 'optimal'
    mahvars=['age', 'educ', 're74', 're75'],  # matched on these
    caliper=0.25,                             # on the propensity score
    exact='married',
    discard='both'                            # drop units outside common support
).fit(formula)
```

**Calipers on covariates**
```python
m = MatchIt(df, caliper={'distance': 0.2, 'age': 5}).fit(formula)  # age within 5 years
```

**Full matching: every unit is used, in subclasses of varying size**
```python
m = MatchIt(df, method='full').fit(formula)                  # ATT
m = MatchIt(df, method='full', estimand='ATE').fit(formula)  # or the ATE
```

**Subclassification for the ATE**
```python
m = MatchIt(df, method='subclass', subclass=6, estimand='ATE').fit(formula)
```

**Coarsened Exact Matching with custom bins**
```python
m = MatchIt(
    df,
    method='cem',
    cutpoints={'age': 4, 're74': [0, 1, 5000, 15000, 40000]}  # bin count or cut points
).fit("treat ~ age + educ + married + re74")
```

**Cardinality matching with balance tolerances**
```python
m = MatchIt(
    df,
    method='cardinality',
    std_tols=0.05,           # max. standardized mean difference
    tols={'age': 1.0},       # max. raw mean difference for age
    ratio=1                  # omit for profile matching
).fit(formula)
```

**Your own propensity scores**
```python
scores = my_model.predict_proba(X)[:, 1]  # one value per row of df
m = MatchIt(df, distance=scores).fit(formula)
```

---

## API Reference

### The `MatchIt` Class

```python
class MatchIt(
    data: pd.DataFrame,
    method: str = "nearest",
    distance: Union[str, np.ndarray, pd.Series] = "glm",
    link: str = "logit",
    replace: bool = False,
    caliper: Union[float, Dict[str, float]] = None,
    ratio: int = None,
    estimand: str = "ATT",
    subclass: int = 6,
    discard: str = "none",
    exact: Union[str, List[str]] = None,
    antiexact: Union[str, List[str]] = None,
    m_order: str = None,
    mahvars: Union[str, List[str]] = None,
    cutpoints: Dict = None,
    tols: Dict[str, float] = None,
    std_tols: float = 0.1,
    pop_size: int = 100,
    max_generations: int = 50,
    min_controls_per_subclass: int = 1,
    max_controls_per_subclass: int = None,
    distance_options: Dict = None,
    random_state: int = None
)
```

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`data`** | `pd.DataFrame` | *Required* | The input dataset containing treatment, outcome, and covariates. |
| **`method`** | `str` | `"nearest"` | The matching algorithm to use. <br>• **`nearest`**: Nearest Neighbor (Greedy) matching. <br>• **`optimal`**: Optimal matching. <br>• **`exact`**: Exact matching. <br>• **`subclass`**: Subclassification (Stratification). <br>• **`cem`**: Coarsened Exact Matching. <br>• **`full`**: Full Matching. <br>• **`genetic`**: Genetic Matching. <br>• **`cardinality`**: Cardinality Matching. |
| **`distance`** | `str`/array | `"glm"` | The method used to estimate propensity scores or distance. Options include `glm`, `cbps`, `mahalanobis`, or ML methods (`randomforest`, `gbm`, `neuralnet`, `decisiontree`, `adaboost`, `lasso`, `ridge`, `elasticnet`). You may also pass a `numpy` array or `pandas` Series of pre-computed scores, one per row of `data`; they are used as-is. |
| **`link`** | `str` | `"logit"` | The scale of the estimated distance measure, as in R `MatchIt`. A plain link (`logit`) matches on the predicted probability; a `linear.` prefix (`linear.logit`) matches on the linear predictor. `glm` accepts `logit`, `probit`, `cloglog`, and `cauchit`; other estimators accept `logit` and `linear.logit`, except `randomforest`, `decisiontree`, and `neuralnet`, which only provide probabilities. |
| **`replace`** | `bool` | `False` | Whether to match with replacement. |
| **`caliper`** | `float`/`dict` | `None` | The maximum allowed distance between matches: a float in standard deviations of the distance measure, or a dict adding covariate limits in their own units, e.g. `{"distance": 0.1, "age": 2}`. |
| **`ratio`** | `int` | `None` | The number of control units to match to each treated unit (1 if not set). For `cardinality`, a whole number requests cardinality matching with that ratio; leaving it unset requests profile matching, which keeps the whole treated group. |
| **`estimand`** | `str` | `"ATT"` | `ATT`, `ATC`, or `ATE`. For `ATC` the control group is the focal group, and `matches()` is keyed by control unit. With `distance="cbps"`, the balance conditions follow the estimand. |
| **`subclass`** | `int` | `6` | Number of subclasses for `method="subclass"`. |
| **`discard`** | `str` | `"none"` | Units to drop before matching for falling outside the common support of the propensity score: `none`, `treated`, `control`, or `both`. |
| **`exact`** | `str`/`list` | `None` | Variables that matched units must share exactly. |
| **`antiexact`** | `str`/`list` | `None` | Variables on which matched units must differ. |
| **`m_order`** | `str` | `None` | Order in which units are matched: `largest`, `smallest`, `random`, or `data`. Defaults to `largest` when a propensity score is available (`smallest` for `ATC`), else `data`. |
| **`mahvars`** | `str`/`list` | `None` | Variables for Mahalanobis distance matching, with the propensity score kept for calipers. In `genetic` matching, the variables of the weighted distance; in `cardinality` matching with a `ratio`, the variables the selected units are paired on. |
| **`cutpoints`** | `dict` | `None` | For `cem`: per-covariate binning, mapping a column to a number of bins or a list of cut points. Numeric covariates not listed are binned with Sturges' rule. |
| **`tols`** | `dict` | `None` | For `cardinality`: covariate-specific balance tolerances, as absolute mean differences. |
| **`std_tols`** | `float` | `0.1` | For `cardinality`: standardized mean difference tolerance for covariates not in `tols`. |
| **`pop_size`** | `int` | `100` | For `genetic`: population size. |
| **`max_generations`** | `int` | `50` | For `genetic`: maximum number of generations. |
| **`min_controls_per_subclass`** | `int` | `1` | For `full`: minimum number of control units per subclass. With 2 or more, every subclass has one treated unit; if there are too few controls for that, subclasses are merged, with a warning. |
| **`max_controls_per_subclass`** | `int` | `None` | For `full`: maximum number of control units per subclass (no limit if not set). Controls the limit leaves no room for stay unmatched. |
| **`distance_options`** | `dict` | `None` | Options passed to the propensity score model, e.g. `{"n_estimators": 500}` for `randomforest`. |
| **`random_state`** | `int` | `None` | Seed for the stochastic steps (`m_order="random"`, genetic matching, ML-based distance estimators). |

#### Which options work with which method

This follows R `MatchIt`. As in R, an option the chosen method cannot use is ignored with a warning, so you can switch `method` without rewriting the call. To turn these warnings into errors, use `warnings.simplefilter("error")`.

| Option | `nearest` | `optimal` | `full` | `genetic` | `cardinality` | `subclass` / `exact` / `cem` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `exact` | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| `antiexact` | ✓ | ✓ | ✓ | ✓ | – | – |
| `mahvars` | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| `caliper` | ✓ | ✓ | ✓ | ✓ | – | – |
| `replace` | ✓ | – | – | ✓ | – | – |
| `m_order` | ✓ | – | – | ✓ | – | – |
| `ratio` | ✓ | ✓ | – | ✓ | ✓ | – |
| `estimand="ATE"` | – | – | ✓ | – | ✓ | ✓ |

Contradictory requests raise an error instead: `estimand="ATE"` with `nearest`, `optimal`, or `genetic`; `mahvars` with `distance="mahalanobis"`; and `mahvars` with `cardinality` but no `ratio`.

#### Methods

| Method | Description |
| :--- | :--- |
| **`fit(formula)`** | Estimates the distance measure and performs the matching. Returns the `MatchIt` object, so calls can be chained. |
| **`summary(print_output=True)`** | Returns a dict with the balance tables `"unmatched"` and `"matched"` (means, mean difference, standardized mean difference, variance ratio per covariate) and `"sample_sizes"`. Prints them unless `print_output=False`. |
| **`matches(format="long")`** | Returns the matched pairs as index labels of `data`. `"long"` has one row per pair (`treated_index`, `control_index`); `"wide"` has one row per treated unit (`control_1`, `control_2`, ...). Empty for methods that do not form pairs (`subclass`, `full`, and `cardinality` without `mahvars`). |
| **`plot(type="balance", variable=None, save_fig=None, **kwargs)`** | Draws a diagnostic plot and returns the matplotlib `Axes` (an array of two `Axes` for `propensity` and `qq`). `type` is `balance` (Love Plot), `propensity`, `jitter`, `ecdf`, or `qq`; the last two need `variable=`. `save_fig="plot.png"` saves the figure. Extra arguments include `title`, `figsize`, and, for the Love Plot, `threshold` and `var_names` (a dict of display names). |

#### Attributes (after `fit`)

| Attribute | Description |
| :--- | :--- |
| **`matched_data`** | The matched units (weight > 0) with all original columns plus `weights`, `subclass`, and, when a propensity score was estimated, `propensity_score` and `distance_measure`. |
| **`weights`** | Matching weights for every row of `data` (0 for unmatched units). |
| **`propensity_scores`** | Estimated propensity scores (`None` for pure Mahalanobis matching). |
| **`data`** | A copy of the input data with the columns above added. |

#### Helper functions

```python
from pymatchit import load_lalonde, compute_effective_sample_size, compute_ks_statistics

# Effective sample size of the weighted groups: {'Treated': ..., 'Control': ..., 'Total': ...}
compute_effective_sample_size(m.weights, m.data['treat'])

# Kolmogorov-Smirnov statistics per covariate, before and after matching
compute_ks_statistics(m.data, ['age', 're74'], 'treat', m.weights)
```

See the `MatchIt` docstring (`help(MatchIt)`) for further detail on each parameter.

---

## Matching Methods Details

1.  **Nearest Neighbor (`method='nearest'`)**:
    * Greedy matching. Selects the closest control unit based on distance measure.
    * With `ratio > 1`, matches are assigned in rounds: every unit gets a first match before any gets a second.
2. **Optimal Matching (`method='optimal'`)**:
    * Minimizes the global total distance across all matched pairs.
3.  **Exact Matching (`method='exact'`)**:
    * Matches units that have identical values for *all* covariates.
4.  **Subclassification (`method='subclass'`)**:
    * Divides the sample into subclasses (bins) based on propensity score quantiles of the focal group (treated for `ATT`, control for `ATC`, everyone for `ATE`).
5.  **Coarsened Exact Matching (`method='cem'`)**:
    * Coarsens continuous variables into bins (Sturges' rule, or as defined by `cutpoints`) and matches exactly on these coarsened bins.
6.  **Full Matching (`method='full'`)**:
    * Optimal full matching, as in R's `optmatch`: places every unit in a subclass such that the total distance between treated and control units within subclasses is as small as possible. A subclass is one treated unit with one or more controls, or one control with one or more treated units.
    * Uses all units, so it suits data where the groups overlap too little for every treated unit to get a control of its own. The price is uneven weights: check the effective sample size with `compute_effective_sample_size`.
7.  **Genetic Matching (`method='genetic'`)**:
    * Uses a genetic/evolutionary algorithm to find optimal covariate weights that maximize balance between groups prior to matching. The distance is computed on the covariates plus the propensity score, or on `mahvars` if given; balance is always optimized on all covariates.
8.  **Cardinality Matching (`method='cardinality'`)**:
    * Finds the largest possible subset of the data where treated and control groups satisfy user-specified balance constraints: `std_tols` (standardized mean differences) and `tols` (raw mean differences for individual covariates).
    * By default this is *profile matching*: the treated group is kept whole and the largest balanced control subset is selected. Setting `ratio` requests *cardinality matching*: the largest balanced sample with that many controls per treated unit. Adding `mahvars` then pairs the selected units.

---

## Upgrading

### From 0.6.0

Version 0.6.1 fixes two matching bugs, which changes the results of the affected calls:

* **Full matching** is now optimal. Before, units could be forced into distant subclasses when the groups overlap little, which left covariates unbalanced.
* **Nearest neighbor matching with `mahvars` and a caliper** applies the caliper to the propensity score. It was applied to the Mahalanobis distance, which left most units unmatched.

### From 0.5.0

Version 0.6.0 changes the results of some methods to agree with R `MatchIt`. The changes most likely to affect an existing analysis:

* **`link`**: `logit` and `probit` now match on the predicted probability. Use `link="linear.logit"` for the previous behaviour (matching on the linear predictor). Calipers are in standard deviations of this measure, so their width changes too.
* **User-supplied distances** are used as-is; they are no longer logit-transformed.
* **`ratio`** and **`m_order`** default to `None` (previously `1` and `"largest"`). Matching still defaults to 1:1, and to `largest` order when a propensity score is available.
* **Unused options warn**: an option the chosen method cannot use is now reported instead of being silently ignored.
* **CEM** bins numeric covariates with Sturges' rule by default instead of 5 bins.

The [changelog](https://github.com/jtuenner/pymatchit/blob/main/CHANGELOG.md) has the full list, including the changes to genetic, cardinality, full matching, and CBPS.

---

## Contributing

Bug reports and pull requests are welcome; see [CONTRIBUTING.md](https://github.com/jtuenner/pymatchit/blob/main/CONTRIBUTING.md). The package is released under the [MIT License](https://github.com/jtuenner/pymatchit/blob/main/LICENSE).

---

## Citation

If you use `pymatchit-causal` in your research, please cite it:

> Tünnermann, J. (2026). pymatchit: Propensity Score Matching and Causal Inference in Python (Version 0.6.1). Zenodo. https://doi.org/10.5281/zenodo.17839522

**BibTeX:**
```bibtex
@software{pymatchit_causal,
  author       = {Jonas Tünnermann},
  title        = {pymatchit: Propensity Score Matching and Causal Inference in Python},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {0.6.1},
  doi          = {10.5281/zenodo.17839522},
  url          = {https://doi.org/10.5281/zenodo.17839522}
}
```
