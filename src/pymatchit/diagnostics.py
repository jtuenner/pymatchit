# File: src/pymatchit/diagnostics.py

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy.stats import ks_2samp

def compute_weighted_stats(x: np.ndarray, weights: np.ndarray) -> dict:
    """
    Computes weighted mean and variance.
    """
    if len(x) == 0 or np.sum(weights) == 0:
        return {'mean': np.nan, 'var': np.nan, 'std': np.nan}

    V1 = np.sum(weights)
    V2 = np.sum(weights**2)
    
    # Weighted Mean
    weighted_mean = np.average(x, weights=weights)

    # Unbiased weighted sample variance (Reliability weights)
    if V1**2 <= V2:
        weighted_var = 0.0
    else:
        numerator = np.sum(weights * (x - weighted_mean)**2)
        weighted_var = numerator / (V1 - (V2 / V1))

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
    estimand: str = "ATT"
) -> pd.DataFrame:
    """
    Generates the full 'summary(out)' table.
    """
    # 1. Calculate Unmatched Balance (All Data, weights=1)
    unmatched_balance = covariate_balance(
        original_data, covariates, treatment_col, weights=None
    )
    
    # 2. Calculate Matched Balance (Matched Data, weights=matched_weights)
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
            std_factors[cov] = treated_original[cov].std()
        elif estimand == "ATC":
            std_factors[cov] = control_original[cov].std()
        elif estimand == "ATE":
            var_t = treated_original[cov].var()
            var_c = control_original[cov].var()
            std_factors[cov] = np.sqrt((var_t + var_c) / 2)
        else:
            std_factors[cov] = treated_original[cov].std()

    # 4. Compute SMD
    unmatched_balance['Std. Mean Diff.'] = [
        row['Mean Diff'] / std_factors[idx] if std_factors[idx] > 0 else np.nan
        for idx, row in unmatched_balance.iterrows()
    ]
    
    matched_balance['Std. Mean Diff.'] = [
        row['Mean Diff'] / std_factors[idx] if std_factors[idx] > 0 else np.nan
        for idx, row in matched_balance.iterrows()
    ]

    return unmatched_balance, matched_balance

def compute_sample_size_table(
    data: pd.DataFrame, 
    treatment_col: str, 
    weights: pd.Series, 
    mask_kept: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Computes the 'Sample Sizes' table (All, Matched, Unmatched, Discarded).
    """
    counts = {
        'All': {'Control': 0, 'Treated': 0},
        'Matched': {'Control': 0, 'Treated': 0},
        'Unmatched': {'Control': 0, 'Treated': 0},
        'Discarded': {'Control': 0, 'Treated': 0},
    }
    
    treat_mask = (data[treatment_col] == 1)
    control_mask = (data[treatment_col] == 0)
    
    counts['All']['Treated'] = treat_mask.sum()
    counts['All']['Control'] = control_mask.sum()
    
    if mask_kept is not None:
        discarded_mask = ~mask_kept
        counts['Discarded']['Treated'] = (discarded_mask & treat_mask).sum()
        counts['Discarded']['Control'] = (discarded_mask & control_mask).sum()
    
    matched_mask = (weights > 0)
    counts['Matched']['Treated'] = (matched_mask & treat_mask).sum()
    counts['Matched']['Control'] = (matched_mask & control_mask).sum()
    
    counts['Unmatched']['Treated'] = counts['All']['Treated'] - counts['Discarded']['Treated'] - counts['Matched']['Treated']
    counts['Unmatched']['Control'] = counts['All']['Control'] - counts['Discarded']['Control'] - counts['Matched']['Control']
    
    return pd.DataFrame(counts).T


def compute_effective_sample_size(weights: pd.Series, treatment: pd.Series) -> Dict[str, float]:
    """
    Computes the Effective Sample Size (ESS) for weighted samples.
    
    ESS = (sum(w))^2 / sum(w^2)
    
    This measures the equivalent number of unweighted observations.
    A lower ESS relative to the actual sample size indicates that the
    weighting scheme reduces precision.
    
    Args:
        weights: Matching weights.
        treatment: Treatment indicator (0/1).
        
    Returns:
        Dictionary with ESS for treated, control, and total.
    """
    def _ess(w):
        w = w[w > 0]
        if len(w) == 0:
            return 0.0
        return (w.sum())**2 / (w**2).sum()
    
    treated_mask = treatment == 1
    control_mask = treatment == 0
    
    ess_treated = _ess(weights[treated_mask])
    ess_control = _ess(weights[control_mask])
    ess_total = _ess(weights)
    
    return {
        'Treated': round(ess_treated, 1),
        'Control': round(ess_control, 1),
        'Total': round(ess_total, 1)
    }


def compute_ks_statistics(
    data: pd.DataFrame,
    covariates: list,
    treatment_col: str,
    weights: pd.Series
) -> pd.DataFrame:
    """
    Computes the Kolmogorov-Smirnov (KS) statistic for each covariate
    between treated and control groups, before and after matching.
    
    The KS statistic measures the maximum vertical distance between the
    empirical CDFs of two samples. A smaller value indicates better balance.
    
    Args:
        data: The dataset.
        covariates: List of covariate column names.
        treatment_col: Treatment indicator column name.
        weights: Matching weights.
        
    Returns:
        DataFrame with KS statistics and p-values before and after matching.
    """
    rows = []
    treated_mask = data[treatment_col] == 1
    control_mask = data[treatment_col] == 0
    matched_mask = weights > 0
    
    for cov in covariates:
        if not pd.api.types.is_numeric_dtype(data[cov]):
            continue
            
        # Before matching (raw)
        x_t_raw = data.loc[treated_mask, cov].values
        x_c_raw = data.loc[control_mask, cov].values
        
        ks_raw, p_raw = ks_2samp(x_t_raw, x_c_raw)
        
        # After matching (weighted)
        # For KS test with weights, we resample based on weights
        x_t_matched = data.loc[treated_mask & matched_mask, cov].values
        x_c_matched = data.loc[control_mask & matched_mask, cov].values
        
        if len(x_t_matched) > 0 and len(x_c_matched) > 0:
            ks_matched, p_matched = ks_2samp(x_t_matched, x_c_matched)
        else:
            ks_matched, p_matched = np.nan, np.nan
        
        rows.append({
            'Covariate': cov,
            'KS (Raw)': round(ks_raw, 4),
            'p-value (Raw)': round(p_raw, 4),
            'KS (Matched)': round(ks_matched, 4) if not np.isnan(ks_matched) else np.nan,
            'p-value (Matched)': round(p_matched, 4) if not np.isnan(p_matched) else np.nan,
        })
    
    return pd.DataFrame(rows).set_index('Covariate')