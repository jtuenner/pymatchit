# File: src/pymatchit/core.py

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import statsmodels.api as sm
from typing import Optional, Union, List, Dict

from .distance import estimate_distance
from .matchers import BaseMatcher, NearestNeighborMatcher, ExactMatcher, SubclassMatcher, CEMMatcher
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
        estimand: str = "ATT",
        subclass: int = 6,
        discard: str = "none",
        cutpoints: Optional[Dict] = None, 
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
        self.discard = discard
        self.cutpoints = cutpoints
        self.random_state = random_state

        # Storage for results
        self.formula = None
        self.propensity_scores = None
        self.distance_measure = None
        self.matched_data = None
        self.matched_indices = None 
        self.weights = None
        self._treatment_col = None
        
        self._mask_kept = None 

    def fit(self, formula: str):
        self.formula = formula 
        
        # 1. Robust Input Validation (Safety Check)
        self._validate_inputs(formula)
        
        # 2. Estimate Distance
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

        # 3. Apply Common Support / Discard Logic
        if self.discard != "none" and self.propensity_scores is not None:
             self._apply_discard_logic()
        else:
             self._mask_kept = pd.Series(True, index=self.data.index)

        # 4. Match
        self._match()
        return self

    def _validate_inputs(self, formula: str):
        """
        Performs rigorous checks on the input data and formula.
        """
        # A. Check Index Uniqueness
        # Matching relies on unique indices to map treated<->control.
        if not self.data.index.is_unique:
            raise ValueError("Input DataFrame index must be unique. Try running `df.reset_index(drop=True)` before passing it to MatchIt.")

        # B. Check Formula Syntax
        if "~" not in formula:
            raise ValueError("Formula must contain '~' separating treatment and covariates.")
        
        lhs = formula.split("~")[0].strip()
        rhs = formula.split("~")[1].strip()

        # C. Check Treatment Column
        if lhs not in self.data.columns:
            raise ValueError(f"Treatment variable '{lhs}' not found in dataframe.")
        self._treatment_col = lhs
        
        # D. Check Treatment Values (Strictly Binary)
        # We allow integers 0/1, floats 0.0/1.0, or booleans.
        t_vals = self.data[lhs].unique()
        # Filter out NaNs for a moment to check values (NaN check is next)
        t_vals_clean = t_vals[~pd.isnull(t_vals)]
        
        valid_set = {0, 1, 0.0, 1.0, False, True}
        is_binary = all(v in valid_set for v in t_vals_clean)
        
        if not is_binary:
            raise ValueError(f"Treatment variable must be binary (0 and 1). Found values: {t_vals_clean}")
            
        # E. Check Missing Values (NaNs) in Treatment
        if self.data[lhs].isnull().any():
            raise ValueError(f"Treatment variable '{lhs}' contains missing values (NaN). Please drop or impute them.")

        # F. Check Missing Values (NaNs) in Covariates
        # We use patsy to parse the RHS and check ONLY the columns involved in the model.
        try:
            # NA_action='raise' will trigger a patsy.PatsyError if NaNs are found in the model frame
            patsy.dmatrix(rhs, self.data, NA_action='raise', return_type='dataframe')
        except patsy.PatsyError as e:
            if "missing values" in str(e).lower():
                raise ValueError("Covariates contain missing values (NaN). pymatchit requires complete data. Please drop missing rows or impute data.") from e
            elif "factor" in str(e).lower() and "not found" in str(e).lower():
                 # Extracts the variable name from the error usually
                raise ValueError(f"Formula Error: {str(e)}") from e
            else:
                raise ValueError(f"Error parsing formula or data: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"Unexpected data validation error: {str(e)}") from e


    def _apply_discard_logic(self):
        treat_mask = (self.data[self._treatment_col] == 1)
        control_mask = (self.data[self._treatment_col] == 0)
        
        scores = self.propensity_scores
        
        t_min, t_max = scores[treat_mask].min(), scores[treat_mask].max()
        c_min, c_max = scores[control_mask].min(), scores[control_mask].max()
        
        keep_mask = pd.Series(True, index=self.data.index)
        
        if self.discard == "treated":
            cond_discard = treat_mask & ((scores < c_min) | (scores > c_max))
            keep_mask[cond_discard] = False
            
        elif self.discard == "control":
            cond_discard = control_mask & ((scores < t_min) | (scores > t_max))
            keep_mask[cond_discard] = False
            
        elif self.discard == "both":
            common_min = max(t_min, c_min)
            common_max = min(t_max, c_max)
            cond_discard = (scores < common_min) | (scores > common_max)
            keep_mask[cond_discard] = False
            
        else:
            raise ValueError(f"Discard option '{self.discard}' not recognized.")
            
        n_dropped = (~keep_mask).sum()
        if n_dropped > 0:
            print(f"Discarding {n_dropped} units outside common support ({self.discard}).")
            self._mask_kept = keep_mask
        else:
            self._mask_kept = keep_mask

    def _get_matcher(self) -> BaseMatcher:
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
        elif self.method == 'cem':
            return CEMMatcher(
                cutpoints=self.cutpoints,
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
        
        # Apply discard mask (common support)
        if self._mask_kept is not None:
            active_treat = self.data.loc[self._mask_kept, self._treatment_col]
            
            if self.distance_measure is not None:
                active_dist = self.distance_measure[self._mask_kept]
            else:
                active_dist = None
                
            active_covs = X_data.loc[self._mask_kept]
        else:
            active_treat = self.data[self._treatment_col]
            active_dist = self.distance_measure
            active_covs = X_data

        matches, sub_weights = matcher.match(
            treatment=active_treat,
            distance_measure=active_dist,
            covariates=active_covs,
            estimand=self.estimand 
        )
        
        self.matched_indices = matches
        
        # Reassemble weights for full dataset
        full_weights = pd.Series(0.0, index=self.data.index)
        full_weights.update(sub_weights)
        
        self.weights = full_weights
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

    def plot(self, type: str = "balance", variable: Optional[str] = None, threshold: float = 0.1, var_names: Optional[dict] = None, colors: tuple = ("#e74c3c", "#3498db")):
        if self.matched_data is None:
            raise ValueError("Run .fit() before plotting.")
            
        if type == "balance":
            summary_stats = self.summary(print_output=False)
            love_plot(summary_stats, threshold=threshold, var_names=var_names, colors=colors)
        elif type == "propensity" or type == "jitter":
            if self.propensity_scores is None:
                raise ValueError("No propensity scores found (did you use Mahalanobis?). Cannot plot propensity.")
            propensity_plot(data=self.data, treatment_col=self._treatment_col, weights=self.weights)
        elif type == "ecdf":
            if variable is None:
                raise ValueError("You must specify 'variable=' for eCDF plots.")
            if variable not in self.data.columns:
                raise ValueError(f"Variable '{variable}' not found in data.")
            ecdf_plot(data=self.data, var_name=variable, treatment_col=self._treatment_col, weights=self.weights)
        else:
            raise NotImplementedError(f"Plot type '{type}' not supported. Try 'balance', 'propensity', or 'ecdf'.")