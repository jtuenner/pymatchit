# File: src/pymatchit/core.py

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import statsmodels.api as sm
from typing import Optional, Union, List, Dict

from .distance import estimate_distance
from .matchers import BaseMatcher, NearestNeighborMatcher, ExactMatcher, SubclassMatcher
from .diagnostics import create_summary_table
from .plotting import love_plot, propensity_plot, ecdf_plot

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
        estimand: str = "ATT",  # <--- NEW PARAMETER
        subclass: int = 6,      # <--- NEW PARAMETER
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
        self.estimand = estimand 
        self.subclass = subclass
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
        self.formula = formula 
        self._validate_inputs(formula)
        
        # Subclassification uses Propensity Scores (GLM), 
        # Mahalanobis skips it.
        if self.distance == "mahalanobis":
            self.propensity_scores = None
            self.distance_measure = None 
            print("Distance 'mahalanobis' selected. Skipping Propensity Score estimation.")
        else:
            self.propensity_scores, self.distance_measure = estimate_distance(
                data=self.data,
                formula=formula,
                method=self.distance,
                link=self.link
            )
            self.data['propensity_score'] = self.propensity_scores
            self.data['distance_measure'] = self.distance_measure

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
            is_mahalanobis = (self.distance == 'mahalanobis')
            return NearestNeighborMatcher(
                ratio=self.ratio, 
                replace=self.replace, 
                caliper=self.caliper,
                random_state=self.random_state,
                mahalanobis=is_mahalanobis
            )
        elif self.method == 'exact':
            return ExactMatcher(
                ratio=self.ratio,
                replace=self.replace,
                random_state=self.random_state
            )
        elif self.method == 'subclass':
            return SubclassMatcher(
                n_subclasses=self.subclass,
                random_state=self.random_state
            )
        else:
            raise NotImplementedError(f"Method {self.method} not supported yet.")

    def _match(self):
        print(f"Performing {self.method} matching ({self.estimand})...")
        
        matcher = self._get_matcher()

        rhs_formula = self.formula.split('~')[1]
        X_data = pd.DataFrame(patsy.dmatrix(rhs_formula, self.data, return_type='dataframe'))
        if 'Intercept' in X_data.columns:
            X_data = X_data.drop(columns=['Intercept'])

        # PASS ESTIMAND HERE
        self.matched_indices, self.weights = matcher.match(
            treatment=self.data[self._treatment_col],
            distance_measure=self.distance_measure,
            covariates=X_data,
            estimand=self.estimand 
        )
        
        self.data['weights'] = self.weights
        self.matched_data = self.data[self.data['weights'] > 0].copy()
        
        n_matched = len(self.matched_data)
        print(f"Matching complete. {n_matched} observations in matched set.")

        if n_matched == 0:
            import warnings
            warnings.warn(
                f"No matches were found! This often happens with '{self.method}' matching "
                "on continuous variables or if strict cutoffs exclude all units."
            )
    

    def summary(self, print_output: bool = True):
        if self.matched_data is None:
            raise ValueError("You must run .fit() before .summary()")

        rhs = self.formula.split("~")[1]
        covariates = [x.strip() for x in rhs.split("+")]
        
        # PASS ESTIMAND HERE
        unmatched, matched = create_summary_table(
            original_data=self.data,
            matched_data=self.matched_data,
            covariates=covariates,
            treatment_col=self._treatment_col,
            weights=self.weights,
            estimand=self.estimand
        )

        if print_output:
            print("\nSummary of Balance for All Data:")
            print(unmatched[['Means Treated', 'Means Control', 'Std. Mean Diff.', 'Var Ratio']])
            print("\nSummary of Balance for Matched Data:")
            print(matched[['Means Treated', 'Means Control', 'Std. Mean Diff.', 'Var Ratio']])
        
        return {'unmatched': unmatched, 'matched': matched}

    def plot(self, 
             type: str = "balance", 
             variable: Optional[str] = None, # New arg for ecdf
             threshold: float = 0.1, 
             var_names: Optional[dict] = None, 
             colors: tuple = ("#e74c3c", "#3498db")
             ):
        """
        Generates diagnostic plots.
        
        Args:
            type (str): 'balance' (Love Plot), 'propensity' (Histogram), or 'ecdf'.
            variable (str): The column name to plot for type='ecdf'.
        """
        if self.matched_data is None:
            raise ValueError("Run .fit() before plotting.")
            
        if type == "balance":
            summary_stats = self.summary(print_output=False)
            love_plot(summary_stats, threshold=threshold, var_names=var_names, colors=colors)
            
        elif type == "propensity" or type == "jitter":
            # 'jitter' is the R name, 'propensity' is clearer. We support both.
            if self.propensity_scores is None:
                raise ValueError("No propensity scores found (did you use Mahalanobis?). Cannot plot propensity.")
            
            propensity_plot(
                data=self.data, 
                treatment_col=self._treatment_col, 
                weights=self.weights
            )
            
        elif type == "ecdf":
            if variable is None:
                raise ValueError("You must specify 'variable=' for eCDF plots.")
            if variable not in self.data.columns:
                raise ValueError(f"Variable '{variable}' not found in data.")
                
            ecdf_plot(
                data=self.data,
                var_name=variable,
                treatment_col=self._treatment_col,
                weights=self.weights
            )
            
        else:
            raise NotImplementedError(f"Plot type '{type}' not supported. Try 'balance', 'propensity', or 'ecdf'.")