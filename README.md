# pymatchit: Propensity Score Matching in Python

**Scalable Causal Inference, Propensity Score Matching (PSM), and Coarsened Exact Matching (CEM).**

`pymatchit` is a Python port of the standard R package `MatchIt`. It allows data scientists to preprocess data for causal inference by balancing covariates between treated and control groups using state-of-the-art matching methods.

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

## Installation

```bash
pip install pymatchit
````

## Quick Start

```python
from pymatchit import MatchIt, load_lalonde

# 1. Load Data
df = load_lalonde()

# 2. Match (Nearest Neighbor with Caliper)
m = MatchIt(df, method='nearest', caliper=0.2)
m.fit("treat ~ age + educ + race + married + nodegree + re74 + re75")

# 3. Assess Balance
m.summary()
m.plot(type='balance')

# 4. Get Matched Data for Analysis
matched_data = m.matches()
```

## Citation

If you use `pymatchit` in your research, please cite it.

**Until the accompanying paper is published, please cite the software directly:**

> Tünnermann, J. (2025). pymatchit: Nonparametric Preprocessing for Parametric Causal Inference (Version 0.1.0) [Computer software]. https://github.com/jtuenner/pymatchit

**BibTeX:**
```bibtex
@software{pymatchit,
  author = {Tünnermann, Jonas},
  title = {pymatchit: Nonparametric Preprocessing for Parametric Causal Inference},
  url = {[https://github.com/jtuenner/pymatchit](https://github.com/jtuenner/pymatchit)},
  version = {0.1.0},
  year = {2025}
}
```