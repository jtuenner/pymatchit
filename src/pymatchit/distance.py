# File: src/pymatchit/distance.py

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from typing import Tuple, Optional

def estimate_distance(
    data: pd.DataFrame,
    formula: str,
    method: str = "glm",
    link: str = "logit"
) -> Tuple[pd.Series, pd.Series]:
    """
    Estimates propensity scores and the distance measure (linear predictor).

    Args:
        data (pd.DataFrame): The dataset.
        formula (str): R-style formula (e.g., "treat ~ age + educ").
        method (str): Estimation method. Currently only 'glm' is supported.
        link (str): The link function. Default 'logit'. 
                    If 'logit', the distance returned is the linear logit.
                    If 'linear.logit', same as above.
                    If 'probit', uses probit link.

    Returns:
        Tuple[pd.Series, pd.Series]: 
            1. Propensity Scores (raw probabilities)
            2. Distance Measure (values used for matching, e.g., logits)
    """
    if method != "glm":
        raise NotImplementedError(f"Distance method '{method}' not implemented yet. Use 'glm'.")

    # 1. Define the Family and Link function
    # In R, family=binomial(link="logit") is the default.
    family = sm.families.Binomial()
    
    if link == 'probit':
        family = sm.families.Binomial(link=sm.families.links.Probit())
    elif link == 'logit' or link == 'linear.logit':
        # FIXED: Use Logit class instead of deprecated function alias
        family = sm.families.Binomial(link=sm.families.links.Logit())

    # 2. Fit the Model (GLM)
    try:
        model = smf.glm(formula=formula, data=data, family=family)
        result = model.fit()
    except Exception as e:
        raise RuntimeError(f"Failed to fit Propensity Score model: {str(e)}")

    # 3. Extract Values
    # .fittedvalues in statsmodels GLM is the predicted probability (mu)
    propensity_scores = result.fittedvalues

    # 4. Calculate the Distance Measure (Linear Predictor)
    # R matches on the linear predictor (X * beta), not the probability.
    if link in ['logit', 'linear.logit', 'probit']:
        # FIXED: Updated API from linear=True to which="linear"
        # Also ensures we get a numpy array or series, handled below
        distance_measure = result.predict(which="linear")
    else:
        # Fallback: specific numeric distance or raw scores
        distance_measure = propensity_scores

    # 5. Ensure return types are pandas Series with correct Index
    # This prevents "AttributeError: 'numpy.ndarray' object has no attribute 'index'"
    if not isinstance(propensity_scores, pd.Series):
        propensity_scores = pd.Series(propensity_scores, index=data.index)
    else:
        propensity_scores.index = data.index

    if not isinstance(distance_measure, pd.Series):
        distance_measure = pd.Series(distance_measure, index=data.index)
    else:
        distance_measure.index = data.index

    return propensity_scores, distance_measure