# pymatchit-causal: Propensity Score Matching in Python

**Scalable Causal Inference, Propensity Score Matching (PSM), and Coarsened Exact Matching (CEM).**

`pymatchit-causal` is a Python port of the standard R package `MatchIt`. It allows data scientists to preprocess data for causal inference by balancing covariates between treated and control groups using state-of-the-art matching methods.

## Why use pymatchit?
If you are looking for **Propensity Score Matching** in Python, this library provides a robust, "R-style" workflow including:
* **Propensity Score Estimation:** Logistic Regression (GLM), Random Forest, GBM, Neural Networks.
* **Matching Algorithms:** Nearest Neighbor (Greedy), Exact, Subclassification, and Coarsened Exact Matching (CEM).
* **Diagnostics:** Publication-ready Love Plots (Covariate Balance), Propensity Density Plots, and ECDF plots.

## Features
* **Matching Methods:** Nearest Neighbor, Exact, Coarsened Exact Matching (CEM), Subclassification.
* **Distance Metrics:** Logistic Regression (GLM), Mahalanobis, Random Forest, GBM, Neural Networks, etc.
* **Diagnostics:** Love Plots, ECDF Plots, Propensity Score Density Plots, and Summary Tables (SMD, Variance Ratios).
* **Parity:** Designed to mirror the R `MatchIt` API (`matchit(formula, data, method=...)`).

## Table of Contents
1. [Installation](#installation)
2. [Example Workflow (Excel Data)](#example-workflow-excel-data)
3. [API Reference](#api-reference)
    - [The MatchIt Class](#the-matchit-class)
    - [Methods: fit, summary, plot, matches](#class-methods)
4. [Matching Methods Details](#matching-methods-details)
5. [Distance Measures](#distance-measures)

---

## Installation

```bash
pip install pymatchit-causal
```

Dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `scikit-learn`, `seaborn`, `patsy`.

---

## Example Workflow (Excel Data)

This example demonstrates a full analysis pipeline: loading data from Excel, configuring matching, assessing balance, and extracting the matched dataset.

**Scenario**: You have an Excel file `healthcare_data.xlsx` with a binary treatment variable `took_drug`, an outcome `recovery_time`, and confounders like `age`, `severity`, and `income`.

### 1. Load Data
```python
import pandas as pd
from pymatchit import MatchIt

# Load your dataset
df = pd.read_excel("healthcare_data.xlsx")

# Preview data
# Columns: [patient_id, took_drug, recovery_time, age, severity, income, gender]
print(df.head())
```

### 2. Initialize and Match
We will use **Nearest Neighbor** matching using a **Random Forest** to estimate the propensity score, applying a **caliper** to ensure good matches.

```python
# Initialize the matching model
m = MatchIt(
    data=df,
    method='nearest',           # 1:1 Nearest Neighbor matching
    distance='randomforest',    # Use Random Forest for Propensity Scores
    distance_options={'n_estimators': 500}, # Pass kwargs to sklearn
    caliper=0.2,                # Drop matches > 0.2 std devs apart
    replace=False,              # Match without replacement
    random_state=42             # Reproducibility
)

# Fit the model using an R-style formula
# Format: treatment_variable ~ covariate1 + covariate2 + ...
m.fit("took_drug ~ age + severity + income + gender")
```

### 3. Assess Balance (Diagnostics)
Before analyzing the outcome, verify that the treatment and control groups are balanced.

```python
# 1. Statistical Summary
# Check Standardized Mean Differences (SMD) and Variance Ratios
summary = m.summary()

# 2. Visual Inspection: Love Plot
# Displays covariate balance before (red) and after (blue) matching.
# Ideally, all blue dots should be close to 0.
m.plot(type='balance', threshold=0.1)

# 3. Visual Inspection: Propensity Overlap
# Check if treated and control groups share common support
m.plot(type='propensity')

# 4. Visual Inspection: ECDF Plot
# Check distributional balance for continuous variables (e.g., age)
m.plot(type='ecdf', variable='age')
```

### 4. Extract Matched Data
If balance is satisfactory, extract the data for analysis.

```python
# Get the final dataset containing only matched units (with weights)
matched_df = m.matches(format='long') 

# Merge back with original data to get outcomes if needed, 
# or use m.matched_data which retains the original columns + weights.
final_analysis_set = m.matched_data

print(final_analysis_set.head())
# Now you can perform a weighted regression or T-test on 'recovery_time'
```

---

## API Reference

### The `MatchIt` Class

```python
class MatchIt(
    data: pd.DataFrame,
    method: str = "nearest",
    distance: str = "glm",
    link: str = "logit",
    replace: bool = False,
    caliper: float = None,
    ratio: int = 1,
    estimand: str = "ATT",
    exact: Union[str, List[str]] = None,
    subclass: int = 6,
    discard: str = "none",
    cutpoints: Dict = None,
    distance_options: Dict = None,
    random_state: int = None
)
```

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`data`** | `pd.DataFrame` | *Required* | The input dataset containing treatment, outcome, and covariates. |
| **`method`** | `str` | `"nearest"` | The matching algorithm to use. <br>• **`nearest`**: Nearest Neighbor (Greedy) matching. <br>• **`exact`**: Exact matching on all covariates. <br>• **`subclass`**: Subclassification (Stratification). <br>• **`cem`**: Coarsened Exact Matching. |
| **`distance`** | `str` | `"glm"` | The method used to estimate propensity scores or distance. <br>• **`glm`**: Logistic Regression (standard PSM). <br>• **`mahalanobis`**: Mahalanobis distance (no PS estimation). <br>• **ML Methods**: `randomforest`, `gbm`, `neuralnet`, `decisiontree`, `adaboost`, `lasso`, `ridge`, `elasticnet`. |
| **`link`** | `str` | `"logit"` | The link function for the distance measure. <br>• **`logit`**: Log-odds (linear logit). Recommended for PSM. <br>• **`linear.logit`**: Same as logit. <br>• **`probit`**: Probit regression (only for GLM). <br>• **`linear`**: Raw probabilities (0-1). |
| **`replace`** | `bool` | `False` | Whether to match with replacement. If `True`, control units can be matched to multiple treated units. |
| **`caliper`** | `float` | `None` | The maximum allowed distance between matches, expressed in **standard deviations** of the distance measure. Matches exceeding this are dropped. |
| **`ratio`** | `int` | `1` | The number of control units to match to each treated unit (e.g., 2 for 1:2 matching). |
| **`estimand`** | `str` | `"ATT"` | The target causal estimand. <br>• **`ATT`**: Average Treatment Effect on the Treated. <br>• **`ATE`**: Average Treatment Effect (entire population). |
| **`exact`** | `list` | `None` | A list of column names (e.g. `['gender']`) to enforce **exact** matching on. Matching will occur within strata defined by these variables. |
| **`subclass`** | `int` | `6` | Number of subclasses to create when using `method='subclass'`. |
| **`discard`** | `str` | `"none"` | Logic to discard units outside common support. <br>• **`none`**: Keep all units. <br>• **`treated`**: Drop treated units outside control range. <br>• **`control`**: Drop control units outside treated range. <br>• **`both`**: Drop units from both groups outside the intersection. |
| **`cutpoints`**| `dict` | `None` | For `method='cem'`. Defines cutpoints for continuous variables. Example: `{'age': 4, 'income': [0, 20k, 50k, 100k]}`. |
| **`distance_options`** | `dict` | `None` | Keyword arguments passed directly to the underlying `scikit-learn` estimator (e.g., `{'n_estimators': 100}` for Random Forest). |

---

### Class Methods

#### `fit(formula: str)`
Executes the matching process.
* **`formula`**: A string in the format `treatment ~ cov1 + cov2 + ...`. The variable on the left is treated as the binary treatment indicator. Variables on the right are the covariates used for matching.

#### `summary(print_output: bool = True)`
Calculates balance statistics (Means, Standardized Mean Difference, Variance Ratios) for both matched and unmatched samples.
* **Returns**: A dictionary containing `unmatched` stats dataframe, `matched` stats dataframe, and `sample_sizes`.

#### `plot(type: str, ...)`
Visualizes the matching results.
* **`type='balance'`**: Draws a "Love Plot" of Standardized Mean Differences.
    * *Optional*: `threshold` (vertical line, e.g., 0.1), `var_names` (dict to rename vars for display), `colors`.
* **`type='propensity'`**: Plots Kernel Density Estimates (KDE) of propensity scores for treated vs. control, before and after matching.
* **`type='ecdf'`**: Plots Empirical Cumulative Distribution Functions for a specific continuous variable.
    * *Required*: `variable` (name of the column to plot).

#### `matches(format: str = 'long')`
Retrieves the map of matched units.
* **`format='long'`**: Returns a DataFrame with `treated_index` and `control_index` (one row per match).
* **`format='wide'`**: Returns a DataFrame where each row is a treated unit and columns `control_1`, `control_2`... contain indices of matched controls.

---

## Matching Methods Details

1.  **Nearest Neighbor (`method='nearest'`)**:
    * Greedy matching. For each treated unit, selects the closest control unit based on the distance measure.
    * Supports `caliper` to prune bad matches.
    * Supports `exact` argument for stratified matching (e.g., match nearest neighbor, but *only* within the same 'Gender' group).

2.  **Exact Matching (`method='exact'`)**:
    * Matches units that have identical values for *all* covariates in the formula.
    * Often results in many discarded units if covariates are continuous.

3.  **Subclassification (`method='subclass'`)**:
    * Divides the sample into subclasses (bins) based on propensity score quantiles.
    * Weights are calculated to balance the subclasses. Robust for estimating ATE.

4.  **Coarsened Exact Matching (`method='cem'`)**:
    * Coarsens continuous variables into bins (defined by `cutpoints`) and matches exactly on these coarsened bins.
    * Very fast and reduces model dependence.

---

## Distance Measures

The `distance` parameter controls how similarity is calculated.

* **`glm`**: Default. Fits a Logistic Regression (or Probit if `link='probit'`) to estimate propensity scores.
* **`mahalanobis`**: Calculates the Mahalanobis distance between units based on covariates. **Note**: Does not produce a propensity score property, so propensity plots will not be available unless a separate score is estimated.
* **Machine Learning Estimators**:
    * `randomforest`: Uses `RandomForestClassifier`.
    * `gbm`: Uses `GradientBoostingClassifier`.
    * `decisiontree`: Uses `DecisionTreeClassifier`.
    * `neuralnet`: Uses `MLPClassifier`.
    * `lasso` / `ridge` / `elasticnet`: Uses `LogisticRegression` with penalties.
    * `adaboost`: Uses `AdaBoostClassifier`.

## Citation

If you use `pymatchit-causal` in your research, please cite it.

**Until the accompanying paper is published, please cite the software directly:**

> Tünnermann, J. (2025). pymatchit: Propensity Score Matching and Causal Inference in Python (Version 0.1.0) [Computer software]. https://github.com/jtuenner/pymatchit

**BibTeX:**
```bibtex
@software{pymatchit-causal,
  author = {Tünnermann, Jonas},
  title = {pymatchit: Propensity Score Matching and Causal Inference in Python},
  url = {[https://github.com/jtuenner/pymatchit](https://github.com/jtuenner/pymatchit)},
  version = {0.1.0},
  year = {2025}
}
```