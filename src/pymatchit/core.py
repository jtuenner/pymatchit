# File: src/pymatchit/core.py

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import statsmodels.api as sm
from typing import Optional, Union, List, Dict, Any

from .distance import estimate_distance
from .matchers import (
    BaseMatcher, NearestNeighborMatcher, OptimalMatcher, ExactMatcher,
    SubclassMatcher, CEMMatcher, FullMatcher, GeneticMatcher, CardinalityMatcher
)
from .diagnostics import create_summary_table, compute_sample_size_table
from .plotting import love_plot, propensity_plot, ecdf_plot, qq_plot, jitter_plot

class MatchIt:
    """
    MatchIt: Nonparametric Preprocessing for Parametric Causal Inference.
    A Python port of the R MatchIt package.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        method: str = "nearest",
        distance: Union[str, np.ndarray, pd.Series] = "glm",
        link: str = "logit",
        replace: bool = False,
        caliper: Optional[Union[float, Dict[str, float]]] = None,
        ratio: int = 1,
        estimand: str = "ATT",
        subclass: int = 6,
        discard: str = "none",
        exact: Optional[Union[List[str], str]] = None,
        antiexact: Optional[Union[List[str], str]] = None,
        m_order: str = "largest",
        mahvars: Optional[List[str]] = None,
        cutpoints: Optional[Dict] = None,
        # Cardinality matching options
        tols: Optional[Dict[str, float]] = None,
        std_tols: float = 0.1,
        # Genetic matching options
        pop_size: int = 100,
        max_generations: int = 50,
        # Full matching options
        min_controls_per_subclass: int = 1,
        max_controls_per_subclass: Optional[int] = None,
        distance_options: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None
    ):
        """
        Args:
            method (str): Matching algorithm ('nearest', 'optimal', 'exact', 
                          'subclass', 'cem', 'full', 'genetic', 'cardinality').
            distance (str or array): Distance metric or pre-computed distance vector/matrix.
                Strings: 'glm', 'cbps', 'mahalanobis', 'randomforest', 'gbm', etc.
                Arrays: numpy array or pandas Series of pre-computed propensity scores.
            caliper (float or dict): A float acting as a global threshold on the distance 
                                     measure (std devs), or a dict for covariate-specific 
                                     limits (e.g. {'distance': 0.1, 'age': 2}).
            estimand (str): 'ATT', 'ATE', or 'ATC'.
            antiexact (list): Variables where matched units must have DIFFERENT values.
            mahvars (list): Variables for Mahalanobis distance within PS caliper.
            tols (dict): Covariate-specific balance tolerances for cardinality matching.
            std_tols (float): Default SMD tolerance for cardinality matching.
            pop_size (int): Population size for genetic matching optimization.
            max_generations (int): Max generations for genetic matching.
            distance_options (dict): Options passed to the distance estimation model 
                                     (e.g. {'n_estimators': 500} for randomforest).
            m_order (str): The order matches are generated ('largest', 'smallest', 'random', 'data').
        """
        self.data = data.copy()

        # --- FIX FOR PATSY/PANDAS 2.0 COMPATIBILITY ---
        for col in self.data.select_dtypes(include=['string']).columns:
            self.data[col] = self.data[col].astype(object)
        
        self.method = method
        self.distance = distance
        self.link = link
        self.replace = replace
        self.caliper = caliper
        self.ratio = ratio
        self.estimand = estimand 
        self.subclass = subclass
        self.discard = discard
        self.exact = exact
        self.antiexact = antiexact
        self.m_order = m_order
        self.mahvars = mahvars
        self.cutpoints = cutpoints
        self.tols = tols
        self.std_tols = std_tols
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.min_controls_per_subclass = min_controls_per_subclass
        self.max_controls_per_subclass = max_controls_per_subclass
        self.distance_options = distance_options 
        self.random_state = random_state

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
        self._validate_inputs(formula)
        
        # Check if user supplied pre-computed distance
        user_supplied_distance = isinstance(self.distance, (np.ndarray, pd.Series))
        distance_method = self.distance if isinstance(self.distance, str) else "glm"
        
        # 1. Estimate Distance / Propensity Scores
        if user_supplied_distance:
            # User supplied pre-computed propensity scores
            if isinstance(self.distance, np.ndarray):
                ps = pd.Series(self.distance, index=self.data.index)
            else:
                ps = self.distance.copy()
                ps.index = self.data.index
            
            self.propensity_scores = ps
            
            # Apply link transformation
            if self.link in ['logit', 'linear.logit']:
                from scipy.special import logit
                eps = 1e-9
                clipped = np.clip(ps, eps, 1 - eps)
                self.distance_measure = pd.Series(logit(clipped), index=self.data.index)
            else:
                self.distance_measure = ps.copy()
        else:
            should_estimate_ps = (distance_method != "mahalanobis") or (self.discard != "none")
            
            if should_estimate_ps:
                estimation_method = distance_method
                if distance_method == "mahalanobis":
                    estimation_method = "glm" 
                
                ps_scores, ps_dist = estimate_distance(
                    data=self.data,
                    formula=formula,
                    method=estimation_method, 
                    link=self.link,
                    distance_options=self.distance_options, 
                    random_state=self.random_state
                )
                self.propensity_scores = ps_scores
                
                if distance_method == "mahalanobis":
                    self.distance_measure = None
                else:
                    self.distance_measure = ps_dist
            else:
                self.propensity_scores = None
                self.distance_measure = None 

        if self.propensity_scores is not None:
            self.data['propensity_score'] = self.propensity_scores
        if self.distance_measure is not None:
            self.data['distance_measure'] = self.distance_measure

        # 2. Apply Common Support / Discard Logic
        if self.discard != "none" and self.propensity_scores is not None:
             self._apply_discard_logic()
        else:
             self._mask_kept = pd.Series(True, index=self.data.index)

        # 3. Match
        self._match()
        return self

    def matches(self, format: str = "long") -> pd.DataFrame:
        if self.matched_indices is None:
            raise ValueError("You must run .fit() before retrieving matches.")

        if self.method in ('subclass', 'full', 'cardinality'):
            print(f"Note: {self.method.capitalize()} matching does not produce pairwise matches.")
            return pd.DataFrame()

        if format == "long":
            rows = []
            for t_idx, c_indices in self.matched_indices.items():
                for c_idx in c_indices:
                    rows.append({
                        'treated_index': t_idx,
                        'control_index': c_idx
                    })
            df = pd.DataFrame(rows)
            
        elif format == "wide":
            rows = []
            for t_idx, c_indices in self.matched_indices.items():
                row = {'treated_index': t_idx}
                for i, c_idx in enumerate(c_indices):
                    row[f'control_{i+1}'] = c_idx
                rows.append(row)
            df = pd.DataFrame(rows)
            
        else:
            raise ValueError("Format must be 'long' or 'wide'.")

        for col in df.columns:
            if "index" in col or "control_" in col:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")
        
        return df

    def _validate_inputs(self, formula: str):
        if not self.data.index.is_unique:
            raise ValueError("Input DataFrame index must be unique. Try running `df.reset_index(drop=True)` before passing it to MatchIt.")

        if "~" not in formula:
            raise ValueError("Formula must contain '~' separating treatment and covariates.")
        
        lhs = formula.split("~")[0].strip()
        rhs = formula.split("~")[1].strip()

        if not rhs:
            raise ValueError("Formula cannot be empty on the right side. Please specify covariates (e.g. 'treat ~ age + educ').")

        if lhs not in self.data.columns:
            raise ValueError(f"Treatment variable '{lhs}' not found in dataframe.")
        self._treatment_col = lhs
        
        t_vals = self.data[lhs].unique()
        t_vals_clean = t_vals[~pd.isnull(t_vals)]
        
        valid_set = {0, 1, 0.0, 1.0, False, True}
        is_binary = all(v in valid_set for v in t_vals_clean)
        
        if not is_binary:
            raise ValueError(f"Treatment variable must be binary (0 and 1). Found values: {t_vals_clean}")
            
        if self.data[lhs].isnull().any():
            raise ValueError(f"Treatment variable '{lhs}' contains missing values (NaN). Please drop or impute them.")

        covariates = [c.strip() for c in rhs.replace(":", "+").replace("*", "+").split("+")]
        missing_covs = [c for c in covariates if c in self.data.columns and self.data[c].isna().any()]
        if missing_covs:
            raise ValueError(f"Missing values (NaN) found in covariates: {missing_covs}. pymatchit requires complete data. Please impute or drop missing rows.")

        if self.exact is not None:
            if isinstance(self.exact, str):
                self.exact = [self.exact]
            for col in self.exact:
                if col not in self.data.columns:
                    raise ValueError(f"Exact match variable '{col}' not found in dataframe.")
                if self.data[col].isnull().any():
                    raise ValueError(f"Exact match variable '{col}' contains missing values.")

        if self.antiexact is not None:
            if isinstance(self.antiexact, str):
                self.antiexact = [self.antiexact]
            for col in self.antiexact:
                if col not in self.data.columns:
                    raise ValueError(f"Anti-exact variable '{col}' not found in dataframe.")

        if self.mahvars is not None:
            for col in self.mahvars:
                if col not in self.data.columns:
                    raise ValueError(f"Mahalanobis variable '{col}' not found in dataframe.")

        # Validate estimand
        valid_estimands = {"ATT", "ATE", "ATC"}
        if self.estimand not in valid_estimands:
            raise ValueError(f"Estimand must be one of {valid_estimands}, got '{self.estimand}'.")

        try:
            patsy.dmatrix(rhs, self.data, NA_action='raise', return_type='dataframe')
        except patsy.PatsyError as e:
            if "missing values" in str(e).lower():
                raise ValueError("Covariates contain missing values (NaN). pymatchit requires complete data. Please drop missing rows or impute data.") from e
            elif "factor" in str(e).lower() and "not found" in str(e).lower():
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
        distance_method = self.distance if isinstance(self.distance, str) else "user"
        is_mahalanobis = (distance_method == 'mahalanobis')
        
        if self.method == 'nearest':
            return NearestNeighborMatcher(
                ratio=self.ratio, 
                replace=self.replace, 
                caliper=self.caliper,
                m_order=self.m_order,
                random_state=self.random_state,
                mahalanobis=is_mahalanobis
            )
        elif self.method == 'optimal':
            return OptimalMatcher(
                ratio=self.ratio,
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
        elif self.method == 'full':
            return FullMatcher(
                caliper=self.caliper,
                min_controls_per_subclass=self.min_controls_per_subclass,
                max_controls_per_subclass=self.max_controls_per_subclass,
                random_state=self.random_state,
                mahalanobis=is_mahalanobis
            )
        elif self.method == 'genetic':
            return GeneticMatcher(
                ratio=self.ratio,
                replace=self.replace,
                caliper=self.caliper,
                pop_size=self.pop_size,
                max_generations=self.max_generations,
                random_state=self.random_state
            )
        elif self.method == 'cardinality':
            return CardinalityMatcher(
                tols=self.tols,
                std_tols=self.std_tols,
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
        
        if self._mask_kept is not None:
            active_treat = self.data.loc[self._mask_kept, self._treatment_col]
            if self.distance_measure is not None:
                active_dist = self.distance_measure[self._mask_kept]
            else:
                active_dist = None
            
            active_covs = X_data.loc[self._mask_kept]
            active_exact = self.data.loc[self._mask_kept, self.exact] if self.exact else None
        else:
            active_treat = self.data[self._treatment_col]
            active_dist = self.distance_measure
            
            active_covs = X_data
            active_exact = self.data[self.exact] if self.exact else None

        # Handle antiexact: filter out pairs where antiexact variables match
        # This is done post-hoc for methods that support it
        kwargs = {}
        if self.antiexact is not None:
            kwargs['antiexact'] = self.data[self.antiexact] if self._mask_kept is None else self.data.loc[self._mask_kept, self.antiexact]

        matches, sub_weights, subclasses = matcher.match(
            treatment=active_treat,
            distance_measure=active_dist,
            covariates=active_covs,
            estimand=self.estimand,
            exact=active_exact,
            **kwargs
        )
        
        self.matched_indices = matches
        
        full_weights = pd.Series(0.0, index=self.data.index)
        full_weights.update(sub_weights)
        
        full_subclasses = pd.Series(pd.NA, index=self.data.index)
        full_subclasses.update(subclasses)
        
        self.weights = full_weights
        self.data['weights'] = self.weights
        self.data['subclass'] = full_subclasses 
        
        self.matched_data = self.data[self.data['weights'] > 0].copy()
        self.matched_data['subclass'] = pd.to_numeric(self.matched_data['subclass'], errors='coerce').astype("Int64")
        
        n_matched = len(self.matched_data)
        print(f"Matching complete. {n_matched} observations in matched set.")

        if n_matched == 0:
            import warnings
            warnings.warn(
                f"No matches were found! This often happens with strict calipers or '{self.method}' matching "
                "on continuous variables."
            )

    def summary(self, print_output: bool = True):
        if self.matched_data is None:
            raise ValueError("You must run .fit() before .summary()")

        rhs = self.formula.split("~")[1]
        
        X_data = pd.DataFrame(patsy.dmatrix(rhs, self.data, return_type='dataframe'))
        if 'Intercept' in X_data.columns:
            X_data = X_data.drop(columns=['Intercept'])
            
        covariates = list(X_data.columns)
        
        summary_df = X_data.copy()
        summary_df[self._treatment_col] = self.data[self._treatment_col]
        
        unmatched, matched = create_summary_table(
            original_data=summary_df,
            matched_data=self.matched_data, 
            covariates=covariates,
            treatment_col=self._treatment_col,
            weights=self.weights,
            estimand=self.estimand
        )
        
        sample_sizes = compute_sample_size_table(
            data=self.data, 
            treatment_col=self._treatment_col,
            weights=self.weights,
            mask_kept=self._mask_kept
        )

        if print_output:
            print("\nSample Sizes:")
            print(sample_sizes)
            print("\nSummary of Balance for All Data:")
            print(unmatched[['Means Treated', 'Means Control', 'Std. Mean Diff.', 'Var Ratio']])
            print("\nSummary of Balance for Matched Data:")
            print(matched[['Means Treated', 'Means Control', 'Std. Mean Diff.', 'Var Ratio']])
        
        return {
            'unmatched': unmatched, 
            'matched': matched, 
            'sample_sizes': sample_sizes 
        }

    def plot(self, type: str = "balance", variable: Optional[str] = None, 
             save_fig: Optional[str] = None, **kwargs):
        """
        Visualizes the matching results.
        
        Args:
            type: Plot type ('balance', 'propensity', 'ecdf', 'qq', 'jitter').
            variable: Required for 'ecdf' and 'qq' plots.
            save_fig: If provided, saves the figure to this file path.
            **kwargs: Additional arguments passed to the plot function.
        """
        if self.matched_data is None:
            raise ValueError("Run .fit() before plotting.")
            
        ax = None
        
        if type == "balance":
            summary_stats = self.summary(print_output=False)
            ax = love_plot(summary_stats, **kwargs)
            
        elif type == "propensity":
            if self.propensity_scores is None:
                raise ValueError("No propensity scores found (did you use Mahalanobis?). Cannot plot propensity.")
            
            if 'propensity_score' not in self.data.columns:
                 self.data['propensity_score'] = self.propensity_scores
            
            ax = propensity_plot(data=self.data, treatment_col=self._treatment_col, weights=self.weights, **kwargs)
            
        elif type == "ecdf":
            if variable is None:
                raise ValueError("You must specify 'variable=' for eCDF plots.")
            if variable not in self.data.columns:
                raise ValueError(f"Variable '{variable}' not found in data.")
            
            if not pd.api.types.is_numeric_dtype(self.data[variable]):
                raise TypeError(f"eCDF plots are mathematically defined for continuous/numeric variables only. '{variable}' is of type {self.data[variable].dtype}.")
                
            ax = ecdf_plot(data=self.data, var_name=variable, treatment_col=self._treatment_col, weights=self.weights, **kwargs)
        
        elif type == "qq":
            if variable is None:
                raise ValueError("You must specify 'variable=' for QQ plots.")
            if variable not in self.data.columns:
                raise ValueError(f"Variable '{variable}' not found in data.")
            if not pd.api.types.is_numeric_dtype(self.data[variable]):
                raise TypeError(f"QQ plots require numeric variables. '{variable}' is of type {self.data[variable].dtype}.")
            
            ax = qq_plot(data=self.data, var_name=variable, treatment_col=self._treatment_col, weights=self.weights, **kwargs)
        
        elif type == "jitter":
            if self.propensity_scores is None:
                raise ValueError("No propensity scores available for jitter plot.")
            if 'propensity_score' not in self.data.columns:
                self.data['propensity_score'] = self.propensity_scores
            
            ax = jitter_plot(data=self.data, treatment_col=self._treatment_col, weights=self.weights, **kwargs)
            
        else:
            raise NotImplementedError(f"Plot type '{type}' not supported. Try 'balance', 'propensity', 'ecdf', 'qq', or 'jitter'.")
        
        # Save figure if requested
        if save_fig is not None and ax is not None:
            import matplotlib.pyplot as plt
            fig = ax.get_figure() if hasattr(ax, 'get_figure') else plt.gcf()
            fig.savefig(save_fig, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_fig}")
        
        return ax
    
    def __repr__(self):
        distance_str = self.distance if isinstance(self.distance, str) else "user-supplied"
        status = "fitted" if self.matched_data is not None else "not fitted"
        n_matched = len(self.matched_data) if self.matched_data is not None else 0
        return (
            f"MatchIt(method='{self.method}', distance='{distance_str}', "
            f"estimand='{self.estimand}', status='{status}', "
            f"n_matched={n_matched})"
        )