import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Union, List, Dict
from .distance import estimate_distance
from .matchers import NearestNeighborMatcher
from .diagnostics import create_summary_table

class MatchIt:
    """
    MatchIt: Nonparametric Preprocessing for Parametric Causal Inference.
    
    A Python port of the R MatchIt package.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        method: str = "nearest",
        distance: str = "glm",
        link: str = "logit",
        replace: bool = False,
        caliper: Optional[float] = None,
        ratio: int = 1,
        random_state: Optional[int] = None
    ):
        """
        Initialize the MatchIt model configuration.

        Args:
            data (pd.DataFrame): The input dataset.
            method (str): Matching method. Currently only 'nearest' is implemented.
            distance (str): Method to estimate distance measure. Default 'glm' (Propensity Score).
            link (str): The link function for the distance measure. Default 'logit'.
            replace (bool): Whether to match with replacement. Default False.
            caliper (float): The caliper width (in standard deviations of the distance measure).
            ratio (int): How many control units to match to each treated unit (k:1 matching).
            random_state (int): Seed for reproducibility.
        """
        self.data = data.copy()
        self.method = method
        self.distance = distance
        self.link = link
        self.replace = replace
        self.caliper = caliper
        self.ratio = ratio
        self.random_state = random_state

        # Storage for results
        self.propensity_scores = None
        self.distance_measure = None  # The logit of the PS (usually used for matching)
        self.matched_data = None
        self.matched_indices = None # Dict mapping Treated Index -> [Control Indices]
        self.weights = None
        self._treatment_col = None

    def fit(self, formula: str):
        self._validate_inputs(formula)
        
        # UPDATE THIS BLOCK:
        # Delegate calculation to distance.py
        self.propensity_scores, self.distance_measure = estimate_distance(
            data=self.data,
            formula=formula,
            method=self.distance,
            link=self.link
        )
        
        # Assign to dataframe for user visibility
        self.data['propensity_score'] = self.propensity_scores
        self.data['distance_measure'] = self.distance_measure

        self._match()
        return self

    def _validate_inputs(self, formula: str):
        # Identify treatment column from formula
        # We use patsy (via statsmodels) logic: LHS is treatment
        if "~" not in formula:
            raise ValueError("Formula must contain '~' separating treatment and covariates.")
        
        lhs = formula.split("~")[0].strip()
        if lhs not in self.data.columns:
            raise ValueError(f"Treatment variable '{lhs}' not found in dataframe.")
        
        self._treatment_col = lhs
        
        # Ensure binary treatment (0/1)
        unique_vals = self.data[self._treatment_col].unique()
        if not set(unique_vals).issubset({0, 1}):
            # Basic check - in reality, we might need to map boolean/strings to 0/1
            raise ValueError("Treatment variable must be binary (0 and 1).")


    def _match(self):
        print(f"Performing {self.ratio}:1 {self.method} matching...")
        
        # Check for implemented methods
        if self.method == 'nearest':
            matcher = NearestNeighborMatcher(
                ratio=self.ratio, 
                replace=self.replace, 
                caliper=self.caliper,
                random_state=self.random_state
            )
        else:
            raise NotImplementedError(f"Method {self.method} not supported yet.")

        # Delegate to the matcher
        self.matched_indices, self.weights = matcher.match(
            treatment=self.data[self._treatment_col],
            distance_measure=self.distance_measure
        )
        
        # Assign weights to main dataframe
        self.data['weights'] = self.weights
        
        # Create the 'matched_data' view (subset where weights > 0)
        self.matched_data = self.data[self.data['weights'] > 0].copy()
        
        print(f"Matching complete. {len(self.matched_data)} observations in matched set.")
    

    def summary(self):
        """
        Computes and prints the balance summary.
        """
        if self.matched_data is None:
            raise ValueError("You must run .fit() before .summary()")

        # Extract Covariates from the formula
        # A rough extraction: get all columns used in formula minus the Treatment
        # (For a real production app, inspect the design matrix from patsy)
        
        # Simple heuristic for Tracer Bullet:
        # If formula is "treat ~ age + educ", split by "+"
        rhs = self.formula.split("~")[1]
        covariates = [x.strip() for x in rhs.split("+")]
        
        print("\nSummary of Balance for All Data:")
        unmatched, matched = create_summary_table(
            original_data=self.data,
            matched_data=self.matched_data,
            covariates=covariates,
            treatment_col=self._treatment_col,
            weights=self.weights
        )
        print(unmatched[['Means Treated', 'Means Control', 'Std. Mean Diff.', 'Var Ratio']])

        print("\nSummary of Balance for Matched Data:")
        print(matched[['Means Treated', 'Means Control', 'Std. Mean Diff.', 'Var Ratio']])
        
        return matched