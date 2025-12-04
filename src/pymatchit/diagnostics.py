# File: src/pymatchit/diagnostics.py

import numpy as np
import pandas as pd
from typing import Dict, Optional

def compute_weighted_stats(x: np.ndarray, weights: np.ndarray) -> dict:
    """
    Computes weighted mean and variance.
    """
    if len(x) == 0 or np.sum(weights) == 0:
        return {'mean': np.nan, 'var': np.nan, 'std': np.nan}

    # Weighted Mean
    weighted_mean = np.average(x, weights=weights)

    # Weighted Variance (Reliability weights)
    # Formula: sum(w * (x - mean)^2) / (sum(w) - 1) 
    # This mimics R's cov.wt or var() with frequency weights
    numerator = np.sum(weights * (x - weighted_mean)**2)
    denominator = np.sum(weights) - 1
    
    if denominator <= 0:
        weighted_var = 0.0
    else:
        weighted_var = numerator / denominator

    return {
        'mean': weighted_mean,
        'var': weighted_var,
        'std': np.sqrt(weighted_var)
    }

def covariate_balance(
    data: pd.DataFrame, 
    covariates: list, 
    treatment_col: str, 
    weights: Optional[pd.Series] = None,
    estimand: str = "ATT"
) -> pd.DataFrame:
    """
    Calculates SMD and Variance Ratio for all covariates.
    
    Args:
        data: The dataset (can be matched or unmatched).
        covariates: List of column names to check balance for.
        treatment_col: Name of the treatment column.
        weights: Vector of weights. If None, assumes all 1.0 (unmatched).
        estimand: "ATT", "ATE", or "ATC". Determines the denominator for SMD.
                  Note: The denominator logic usually requires access to the ORIGINAL data.
                  If 'data' passed here is already the matched subset, we might need 
                  passed-in standard deviations. 
                  
                  *Refined Strategy used here:* This function calculates the stats for the 
                  *current* data state. The standardization factor should be passed in or 
                  calculated from a reference.
    """
    if weights is None:
        weights = pd.Series(1.0, index=data.index)

    rows = []
    
    treated_mask = (data[treatment_col] == 1)
    control_mask = (data[treatment_col] == 0)

    for cov in covariates:
        # Extract data
        x_treat = data.loc[treated_mask, cov].values
        w_treat = weights.loc[treated_mask].values
        
        x_ctrl = data.loc[control_mask, cov].values
        w_ctrl = weights.loc[control_mask].values

        # Compute Weighted Stats
        stats_t = compute_weighted_stats(x_treat, w_treat)
        stats_c = compute_weighted_stats(x_ctrl, w_ctrl)

        # Raw Difference
        mean_diff = stats_t['mean'] - stats_c['mean']
        
        # Variance Ratio
        var_ratio = stats_t['var'] / stats_c['var'] if stats_c['var'] > 1e-9 else np.nan

        rows.append({
            'Covariate': cov,
            'Means Treated': stats_t['mean'],
            'Means Control': stats_c['mean'],
            'Mean Diff': mean_diff,
            'Var Ratio': var_ratio,
            # We will fill SMD later because we need the ORIGINAL denominator
        })
        
    return pd.DataFrame(rows).set_index('Covariate')

def create_summary_table(
    original_data: pd.DataFrame,
    matched_data: pd.DataFrame,
    covariates: list,
    treatment_col: str,
    weights: pd.Series,
    estimand: str = "ATT"
) -> pd.DataFrame:
    """
    Generates the full 'summary(out)' table seen in R.
    """
    # 1. Calculate Unmatched Balance (All Data, weights=1)
    unmatched_balance = covariate_balance(
        original_data, covariates, treatment_col, weights=None, estimand=estimand
    )
    
    # 2. Calculate Matched Balance (Matched Data, weights=matched_weights)
    # We use the FULL original dataframe but with the calculated weights
    # (Rows with 0 weight are effectively ignored by the math)
    original_data_with_weights = original_data.copy()
    
    # Ensure weights align
    weights_aligned = weights.reindex(original_data.index).fillna(0)
    
    matched_balance = covariate_balance(
        original_data_with_weights, covariates, treatment_col, weights=weights_aligned, estimand=estimand
    )

    # 3. Calculate Standardization Factors (from ORIGINAL data)
    # For ATT: Standard Deviation of the Treated group in the ORIGINAL sample
    std_factors = {}
    treated_original = original_data[original_data[treatment_col] == 1]
    
    for cov in covariates:
        if estimand == "ATT":
            std_factors[cov] = treated_original[cov].std()
        else:
            # TODO: Implement ATE / ATC logic
            std_factors[cov] = treated_original[cov].std()

    # 4. Compute SMD
    # SMD = (Mean_T_weighted - Mean_C_weighted) / Original_Std_Dev
    
    # Add SMD to Unmatched
    unmatched_balance['Std. Mean Diff.'] = [
        row['Mean Diff'] / std_factors[idx] for idx, row in unmatched_balance.iterrows()
    ]
    
    # Add SMD to Matched
    matched_balance['Std. Mean Diff.'] = [
        row['Mean Diff'] / std_factors[idx] for idx, row in matched_balance.iterrows()
    ]

    # Return a dictionary or a combined MultiIndex dataframe
    # For now, let's return just the Matched Balance with the SMD column included
    return unmatched_balance, matched_balance