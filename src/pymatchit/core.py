import numpy as np
import pandas as pd
import patsy  # <--- NEW IMPORT
import statsmodels.formula.api as smf
import statsmodels.api as sm
from typing import Optional, Union, List, Dict

# Updated imports to use the new BaseMatcher architecture
from .distance import estimate_distance
from .matchers import BaseMatcher, NearestNeighborMatcher, ExactMatcher # Ensure ExactMatcher is imported
from .diagnostics import create_summary_table
from .plotting import love_plot

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
        self.formula = None
        self.propensity_scores = None
        self.distance_measure = None
        self.matched_data = None
        self.matched_indices = None 
        self.weights = None
        self._treatment_col = None

    def fit(self, formula: str):
        """
        Estimate propensity scores and perform matching.
        """
        self.formula = formula 

        # 1. Input Validation
        self._validate_inputs(formula)
        
        # 2. Distance Estimation
        self.propensity_scores, self.distance_measure = estimate_distance(
            data=self.data,
            formula=formula,
            method=self.distance,
            link=self.link
        )
        
        # Assign to dataframe for user visibility
        self.data['propensity_score'] = self.propensity_scores
        self.data['distance_measure'] = self.distance_measure

        # 3. Perform Matching
        self._match()
        return self

    def _validate_inputs(self, formula: str):
        if "~" not in formula:
            raise ValueError("Formula must contain '~' separating treatment and covariates.")
        
        lhs = formula.split("~")[0].strip()
        if lhs not in self.data.columns:
            raise ValueError(f"Treatment variable '{lhs}' not found in dataframe.")
        
        self._treatment_col = lhs
        
        unique_vals = self.data[self._treatment_col].unique()
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError("Treatment variable must be binary (0 and 1).")

    def _get_matcher(self) -> BaseMatcher:
        """
        Factory method to initialize the correct matcher strategy.
        """
        if self.method == 'nearest':
            return NearestNeighborMatcher(
                ratio=self.ratio, 
                replace=self.replace, 
                caliper=self.caliper,
                random_state=self.random_state
            )
        elif self.method == 'exact':
            return ExactMatcher(
                ratio=self.ratio,
                replace=self.replace,
                random_state=self.random_state
            )
        else:
            raise NotImplementedError(f"Method {self.method} not supported yet.")

    def _match(self):
        print(f"Performing {self.ratio}:1 {self.method} matching...")
        
        matcher = self._get_matcher()

        # EXTRACT COVARIATES:
        rhs_formula = self.formula.split('~')[1]
        
        # FIX: Use patsy directly, not sm.patsy
        X_data = pd.DataFrame(patsy.dmatrix(rhs_formula, self.data, return_type='dataframe'))
        
        if 'Intercept' in X_data.columns:
            X_data = X_data.drop(columns=['Intercept'])

        # Delegate the matching process
        self.matched_indices, self.weights = matcher.match(
            treatment=self.data[self._treatment_col],
            distance_measure=self.distance_measure,
            covariates=X_data 
        )
        
        # Assign results
        self.data['weights'] = self.weights
        self.matched_data = self.data[self.data['weights'] > 0].copy()
        
        print(f"Matching complete. {len(self.matched_data)} observations in matched set.")
    

    def summary(self, print_output: bool = True):
        """
        Computes the balance summary.
        """
        if self.matched_data is None:
            raise ValueError("You must run .fit() before .summary()")

        rhs = self.formula.split("~")[1]
        covariates = [x.strip() for x in rhs.split("+")]
        
        unmatched, matched = create_summary_table(
            original_data=self.data,
            matched_data=self.matched_data,
            covariates=covariates,
            treatment_col=self._treatment_col,
            weights=self.weights
        )

        if print_output:
            print("\nSummary of Balance for All Data:")
            print(unmatched[['Means Treated', 'Means Control', 'Std. Mean Diff.', 'Var Ratio']])
            print("\nSummary of Balance for Matched Data:")
            print(matched[['Means Treated', 'Means Control', 'Std. Mean Diff.', 'Var Ratio']])
        
        return {'unmatched': unmatched, 'matched': matched}

    def plot(self, 
             type: str = "balance", 
             threshold: float = 0.1, 
             var_names: Optional[dict] = None,
             colors: tuple = ("#e74c3c", "#3498db")
             ):
        """
        Generates diagnostic plots.
        """
        if self.matched_data is None:
            raise ValueError("Run .fit() before plotting.")
            
        if type == "balance":
            summary_stats = self.summary(print_output=False)
            love_plot(summary_stats, threshold=threshold, var_names=var_names, colors=colors)
        else:
            raise NotImplementedError("Only type='balance' is currently implemented.")