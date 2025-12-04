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
        family = sm.families.Binomial(link=sm.families.links.probit())
    elif link == 'logit' or link == 'linear.logit':
        family = sm.families.Binomial(link=sm.families.links.logit())

    # 2. Fit the Model (GLM)
    # Statsmodels formula API handles the 'treat ~ x1 + x2' parsing automatically.
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
    # We can ask statsmodels to predict with linear=True to get the log-odds.
    if link in ['logit', 'linear.logit', 'probit']:
        # This returns X @ params
        distance_measure = result.predict(linear=True)
    else:
        # Fallback: specific numeric distance or raw scores
        distance_measure = propensity_scores

    # Ensure indices align with the original data
    propensity_scores.index = data.index
    distance_measure.index = data.index

    return propensity_scores, distance_measure