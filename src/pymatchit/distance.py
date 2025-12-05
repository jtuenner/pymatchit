# File: src/pymatchit/distance.py

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.special import logit
from typing import Tuple, Optional, Dict, Any

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

def estimate_distance(
    data: pd.DataFrame,
    formula: str,
    method: str = "glm",
    link: str = "logit",
    distance_options: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Estimates propensity scores using GLM or Machine Learning methods.

    Args:
        data: The dataset.
        formula: R-style formula.
        method: 'glm', 'randomforest', 'decisiontree', 'neuralnet', 'gbm', 'adaboost', 'lasso', 'ridge', 'elasticnet'.
        link: 'logit', 'linear.logit', 'probit' (only for GLM), or 'linear'. 
              For ML methods, 'logit'/'linear.logit' transforms probabilities to logits.
        distance_options: kwargs passed to the sklearn estimator (e.g. {'n_estimators': 100}).
        random_state: Seed for reproducibility.

    Returns:
        propensity_scores: Raw probabilities (0-1).
        distance_measure: Value used for matching (Linear Logit or Probability).
    """
    if distance_options is None:
        distance_options = {}

    # --- 1. GLM (Statsmodels) ---
    if method == "glm":
        # Define Family/Link
        family = sm.families.Binomial()
        if link == 'probit':
            family = sm.families.Binomial(link=sm.families.links.Probit())
        elif link in ['logit', 'linear.logit']:
            family = sm.families.Binomial(link=sm.families.links.Logit())

        try:
            model = smf.glm(formula=formula, data=data, family=family)
            result = model.fit()
        except Exception as e:
            raise RuntimeError(f"Failed to fit GLM Propensity Score model: {str(e)}")

        propensity_scores = result.fittedvalues
        
        # Calculate Distance Measure
        if link in ['logit', 'linear.logit', 'probit']:
            distance_measure = result.predict(which="linear")
        else:
            distance_measure = propensity_scores

    # --- 2. Machine Learning (Scikit-Learn) ---
    else:
        # Prepare Data using Patsy (Handles categorical variables/dummies automatically)
        try:
            # return_type='dataframe' ensures we get pandas Index alignment
            y, X = patsy.dmatrices(formula, data, return_type='dataframe')
            y = y.iloc[:, 0] # Flatten target to Series
        except Exception as e:
            raise ValueError(f"Error creating design matrices from formula: {str(e)}")

        # Select Model
        if method == 'randomforest':
            model = RandomForestClassifier(random_state=random_state, **distance_options)
        elif method == 'decisiontree':
            model = DecisionTreeClassifier(random_state=random_state, **distance_options)
        elif method == 'neuralnet':
            model = MLPClassifier(random_state=random_state, **distance_options)
        elif method == 'gbm':
            model = GradientBoostingClassifier(random_state=random_state, **distance_options)
        elif method == 'adaboost':
            model = AdaBoostClassifier(random_state=random_state, **distance_options)
        elif method == 'lasso':
            # Lasso is Logistic Regression with L1 penalty
            # Need liblinear or saga for l1
            opts = {'penalty': 'l1', 'solver': 'liblinear', 'random_state': random_state}
            opts.update(distance_options)
            model = LogisticRegression(**opts)
        elif method == 'ridge':
            # Ridge is Logistic Regression with L2 penalty
            opts = {'penalty': 'l2', 'random_state': random_state}
            opts.update(distance_options)
            model = LogisticRegression(**opts)
        elif method == 'elasticnet':
            opts = {'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': 0.5, 'random_state': random_state}
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

        # Calculate Distance Measure (Logit transformation if requested)
        if link in ['logit', 'linear.logit']:
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