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
    weights: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Calculates raw stats (Means, Variance Ratios) for the provided data/weights.
    SMD is calculated separately in create_summary_table.
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
        })
        
    return pd.DataFrame(rows).set_index('Covariate')

def create_summary_table(
    original_data: pd.DataFrame,
    matched_data: pd.DataFrame,
    covariates: list,
    treatment_col: str,
    weights: pd.Series,
    estimand: str = "ATT"  # <--- NEW PARAMETER
) -> pd.DataFrame:
    """
    Generates the full 'summary(out)' table.
    """
    # 1. Calculate Unmatched Balance (All Data, weights=1)
    unmatched_balance = covariate_balance(
        original_data, covariates, treatment_col, weights=None
    )
    
    # 2. Calculate Matched Balance (Matched Data, weights=matched_weights)
    # Ensure weights align
    weights_aligned = weights.reindex(original_data.index).fillna(0)
    
    matched_balance = covariate_balance(
        original_data.copy(), covariates, treatment_col, weights=weights_aligned
    )

    # 3. Calculate Standardization Factors (from ORIGINAL data)
    std_factors = {}
    treated_original = original_data[original_data[treatment_col] == 1]
    control_original = original_data[original_data[treatment_col] == 0]
    
    for cov in covariates:
        if estimand == "ATT":
            # ATT: Standardize by Treated group SD
            std_factors[cov] = treated_original[cov].std()
        elif estimand == "ATE":
            # ATE: Standardize by Pooled SD
            # sqrt((var_t + var_c) / 2)
            var_t = treated_original[cov].var()
            var_c = control_original[cov].var()
            std_factors[cov] = np.sqrt((var_t + var_c) / 2)
        else:
            # Fallback to ATT
            std_factors[cov] = treated_original[cov].std()

    # 4. Compute SMD
    # SMD = (Mean_T_weighted - Mean_C_weighted) / Original_Std_Dev_Factor
    
    unmatched_balance['Std. Mean Diff.'] = [
        row['Mean Diff'] / std_factors[idx] if std_factors[idx] > 0 else np.nan
        for idx, row in unmatched_balance.iterrows()
    ]
    
    matched_balance['Std. Mean Diff.'] = [
        row['Mean Diff'] / std_factors[idx] if std_factors[idx] > 0 else np.nan
        for idx, row in matched_balance.iterrows()
    ]

    return unmatched_balance, matched_balance