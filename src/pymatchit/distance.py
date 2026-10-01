# File: src/pymatchit/distance.py

import warnings
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.special import logit
from scipy.optimize import least_squares
from typing import Tuple, Optional, Dict, Any, Union

# Scikit-learn imports
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Link functions available for distance='glm' (statsmodels class names)
_GLM_LINKS = {
    "logit": "Logit",
    "probit": "Probit",
    "cloglog": "CLogLog",
    "cauchit": "Cauchy",
}
# Estimators that only return predicted probabilities. As in R MatchIt, no
# linear predictor is available for them (tree-based probabilities can be
# exactly 0 or 1, where the logit is undefined).
_PROBABILITY_ONLY_METHODS = ("randomforest", "decisiontree", "neuralnet")


def _parse_link(method: str, link: str) -> Tuple[str, bool]:
    """
    Splits `link` into the link function and whether the linear predictor is
    requested ('linear.' prefix; 'linear' alone means 'linear.logit').
    Raises for links the estimator cannot honour.
    """
    linear = link == "linear" or link.startswith("linear.")
    if link == "linear":
        base = "logit"
    elif linear:
        base = link[len("linear.") :]
    else:
        base = link

    if method == "glm":
        if base not in _GLM_LINKS:
            raise ValueError(
                f"link='{link}' is not recognized. Use one of {sorted(_GLM_LINKS)}, "
                "optionally prefixed with 'linear.'."
            )
    elif base != "logit":
        raise ValueError(
            f"link='{link}' is only available for distance='glm', not '{method}'. "
            "Use 'logit' or 'linear.logit'."
        )
    elif linear and method in _PROBABILITY_ONLY_METHODS:
        raise ValueError(
            f"link='{link}' is not available for distance='{method}', which only "
            "provides predicted probabilities. Use link='logit'."
        )
    return base, linear


def estimate_distance(
    data: pd.DataFrame,
    formula: str,
    method: str = "glm",
    link: str = "logit",
    distance_options: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
    estimand: str = "ATT",
) -> Tuple[pd.Series, pd.Series]:
    """
    Estimates propensity scores using GLM, CBPS, or Machine Learning methods.

    Args:
        data: The dataset.
        formula: R-style formula.
        method: 'glm', 'cbps', 'randomforest', 'decisiontree', 'neuralnet', 'gbm',
                'adaboost', 'lasso', 'ridge', 'elasticnet'.
        link: As in R MatchIt, plain links match on the predicted probability;
              'linear.'-prefixed links match on the linear predictor.
              GLM accepts 'logit', 'probit', 'cloglog' and 'cauchit'. Other
              estimators accept 'logit' and, except for random forests,
              decision trees and neural networks, 'linear.logit'.
        distance_options: kwargs passed to the sklearn estimator (e.g. {'n_estimators': 100}).
        random_state: Seed for reproducibility.
        estimand: 'ATT', 'ATC' or 'ATE'. Used by CBPS, whose balance
                  conditions depend on the target population.

    Returns:
        propensity_scores: Raw probabilities (0-1).
        distance_measure: Value used for matching (probability, or linear predictor
                          for 'linear.'-prefixed links).
    """
    if distance_options is None:
        distance_options = {}

    base_link, linear = _parse_link(method, link)

    # --- 1. GLM (Statsmodels) ---
    if method == "glm":
        # Define Family/Link (older statsmodels versions use lowercase names)
        links = sm.families.links
        link_name = _GLM_LINKS[base_link]
        link_cls = getattr(links, link_name, None) or getattr(links, link_name.lower())
        family = sm.families.Binomial(link=link_cls())

        try:
            model = smf.glm(formula=formula, data=data, family=family)
            result = model.fit()
        except Exception as e:
            raise RuntimeError(f"Failed to fit GLM Propensity Score model: {str(e)}")

        propensity_scores = result.fittedvalues

        # As in R MatchIt: plain links match on the predicted probability,
        # 'linear.'-prefixed links match on the linear predictor
        if linear:
            distance_measure = result.predict(which="linear")
        else:
            distance_measure = propensity_scores

    # --- 2. CBPS (Covariate Balancing Propensity Score) ---
    elif method == "cbps":
        propensity_scores, distance_measure = _estimate_cbps(
            data, formula, linear, random_state, estimand
        )

    # --- 3. Machine Learning (Scikit-Learn) ---
    else:
        # Prepare Data using Patsy (Handles categorical variables/dummies automatically)
        try:
            # return_type='dataframe' ensures we get pandas Index alignment
            y, X = patsy.dmatrices(formula, data, return_type="dataframe")
            y = y.iloc[:, 0]  # Flatten target to Series
        except Exception as e:
            raise ValueError(f"Error creating design matrices from formula: {str(e)}")

        # Select Model
        if method == "randomforest":
            model = RandomForestClassifier(
                random_state=random_state, **distance_options
            )
        elif method == "decisiontree":
            model = DecisionTreeClassifier(
                random_state=random_state, **distance_options
            )
        elif method == "neuralnet":
            model = MLPClassifier(random_state=random_state, **distance_options)
        elif method == "gbm":
            model = GradientBoostingClassifier(
                random_state=random_state, **distance_options
            )
        elif method == "adaboost":
            model = AdaBoostClassifier(random_state=random_state, **distance_options)
        elif method == "lasso":
            # Lasso is Logistic Regression with L1 penalty
            # Need liblinear or saga for l1
            opts = {
                "penalty": "l1",
                "solver": "liblinear",
                "random_state": random_state,
            }
            opts.update(distance_options)
            model = LogisticRegression(**opts)
        elif method == "ridge":
            # Ridge is Logistic Regression with L2 penalty
            opts = {"penalty": "l2", "random_state": random_state}
            opts.update(distance_options)
            model = LogisticRegression(**opts)
        elif method == "elasticnet":
            opts = {
                "penalty": "elasticnet",
                "solver": "saga",
                "l1_ratio": 0.5,
                "random_state": random_state,
            }
            opts.update(distance_options)
            model = LogisticRegression(**opts)
        else:
            raise NotImplementedError(f"Distance method '{method}' not implemented.")

        # Fit Model
        try:
            model.fit(X, y)
        except Exception as e:
            raise RuntimeError(f"Failed to fit {method} model: {str(e)}")

        # Predict Probabilities (Propensity Scores)
        # sklearn returns [prob_class_0, prob_class_1]
        scores = model.predict_proba(X)[:, 1]
        propensity_scores = pd.Series(scores, index=data.index)

        # Plain links match on the probability; only 'linear.logit' applies
        # the logit transform for ML methods
        if linear:
            # Clip probabilities to avoid inf/nan in logit
            eps = 1e-9
            clipped_scores = np.clip(propensity_scores, eps, 1 - eps)
            distance_measure = pd.Series(logit(clipped_scores), index=data.index)
        else:
            distance_measure = propensity_scores

    # Ensure return types
    if not isinstance(propensity_scores, pd.Series):
        propensity_scores = pd.Series(propensity_scores, index=data.index)
    else:
        propensity_scores.index = data.index

    if not isinstance(distance_measure, pd.Series):
        distance_measure = pd.Series(distance_measure, index=data.index)
    else:
        distance_measure.index = data.index

    return propensity_scores, distance_measure


def _estimate_cbps(
    data: pd.DataFrame,
    formula: str,
    linear: bool = False,
    random_state: Optional[int] = None,
    estimand: str = "ATT",
) -> Tuple[pd.Series, pd.Series]:
    """
    Covariate Balancing Propensity Score (CBPS) estimation: the
    just-identified estimator of Imai & Ratkovic (2014) 'Covariate
    Balancing Propensity Score'.

    A logistic propensity score model is fit by solving the covariate
    balance conditions sum_i w_i(beta) x_i = 0 instead of the likelihood
    score equations, so that the weights implied by the score balance the
    covariate means exactly. The weights depend on the estimand, as in the
    paper and R's CBPS package:

    - ATE: w = (T - ps) / (ps (1 - ps)); both groups weighted to the full sample
    - ATT: w = (T - ps) / (1 - ps); controls weighted to the treated group
    - ATC: the mirror image of the ATT
    """
    try:
        y, X = patsy.dmatrices(formula, data, return_type="dataframe")
        y_arr = y.iloc[:, 0].values.astype(float)
        X_arr = X.values.astype(float)
    except Exception as e:
        raise ValueError(f"Error creating design matrices from formula: {str(e)}")

    n, p = X_arr.shape

    # Standardize the non-constant columns. This only reparametrizes the
    # model (the fitted scores are unchanged) but puts the balance
    # conditions on a common scale.
    col_sd = X_arr.std(axis=0)
    is_const = col_sd < 1e-12
    center = np.where(is_const, 0.0, X_arr.mean(axis=0))
    scale = np.where(is_const, 1.0, col_sd)
    Xs = (X_arr - center) / scale

    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _balance_conditions(beta):
        """Mean of w_i * x_i: zero when the weighted groups are balanced."""
        ps = np.clip(_sigmoid(Xs @ beta), 1e-9, 1 - 1e-9)
        if estimand == "ATT":
            w = (y_arr - ps) / (1 - ps)
        elif estimand == "ATC":
            w = (y_arr - ps) / ps
        else:
            w = (y_arr - ps) / (ps * (1 - ps))
        return Xs.T @ w / n

    # Start from the logistic regression coefficients. Xs already contains
    # the patsy intercept column, so no separate sklearn intercept is fit.
    try:
        from sklearn.linear_model import LogisticRegression as LR

        init_model = LR(
            random_state=random_state,
            max_iter=1000,
            penalty=None,
            solver="lbfgs",
            fit_intercept=False,
        )
        init_model.fit(Xs, y_arr)
        beta_init = init_model.coef_.flatten()
        if len(beta_init) != p:
            beta_init = np.zeros(p)
    except Exception:
        beta_init = np.zeros(p)

    result = least_squares(_balance_conditions, beta_init, xtol=1e-12, ftol=1e-12)

    if np.max(np.abs(result.fun)) > 1e-6:
        warnings.warn(
            "CBPS could not satisfy the covariate balance conditions exactly "
            f"(largest remaining imbalance: {np.max(np.abs(result.fun)):.2g}). "
            "This usually means the groups barely overlap on some covariate."
        )

    ps = _sigmoid(Xs @ result.x)

    propensity_scores = pd.Series(ps, index=data.index)

    if linear:
        eps = 1e-9
        clipped = np.clip(ps, eps, 1 - eps)
        distance_measure = pd.Series(logit(clipped), index=data.index)
    else:
        distance_measure = propensity_scores.copy()

    return propensity_scores, distance_measure
