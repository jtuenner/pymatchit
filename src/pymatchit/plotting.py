# File: src/pymatchit/plotting.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Optional, Dict, Tuple, List

def love_plot(
    summary_dict: dict, 
    threshold: float = 0.1, 
    colors: Tuple[str, str] = ("#e74c3c", "#3498db"),
    var_names: Optional[Dict[str, str]] = None,
    title: str = "Covariate Balance (Love Plot)"
):
    """
    Generates a publication-ready Love Plot (Standardized Mean Differences).
    """
    sns.set_theme(style="whitegrid", context="talk")
    
    unmatched_df = summary_dict['unmatched']
    matched_df = summary_dict['matched']
    
    # Create Plot Data
    plot_data = pd.DataFrame({
        'Unmatched': unmatched_df['Std. Mean Diff.'].abs(),
        'Matched': matched_df['Std. Mean Diff.'].abs()
    })
    
    if var_names:
        plot_data = plot_data.rename(index=var_names)
    
    plot_data = plot_data.sort_values(by='Unmatched', ascending=True)
    covariates = plot_data.index
    y_pos = range(len(covariates))

    fig, ax = plt.subplots(figsize=(10, len(covariates) * 0.8 + 2))

    color_unmatched, color_matched = colors

    # Dumbbell Lines
    for i, cov in enumerate(covariates):
        ax.hlines(y=i, 
                  xmin=plot_data.loc[cov, 'Matched'], 
                  xmax=plot_data.loc[cov, 'Unmatched'], 
                  color='grey', alpha=0.4, linewidth=2, zorder=1)

    # Points
    ax.scatter(plot_data['Unmatched'], y_pos, 
               color=color_unmatched, label='Unmatched', 
               s=150, edgecolor='white', linewidth=1.5, zorder=3)
    
    ax.scatter(plot_data['Matched'], y_pos, 
               color=color_matched, label='Matched', 
               s=150, edgecolor='white', linewidth=1.5, zorder=3)

    # Reference Lines
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, zorder=0)
    ax.axvline(x=threshold, color='grey', linestyle='--', linewidth=1.5, 
               label=f'Threshold ({threshold})', zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates, fontweight='medium')
    ax.set_xlabel("Absolute Standardized Mean Difference", fontweight='medium', labelpad=15)
    ax.set_title(title, fontweight='bold', y=1.02)
    
    max_val = max(plot_data['Unmatched'].max(), plot_data['Matched'].max())
    # Handle NaN case for perfect balance or empty plots
    if pd.isna(max_val): max_val = 0.5
    ax.set_xlim(left=-0.02, right=max_val * 1.05)
    
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='white')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()

def propensity_plot(
    data: pd.DataFrame,
    treatment_col: str,
    weights: pd.Series,
    title: str = "Propensity Score Distribution (Common Support)"
):
    """
    Plots the density of Propensity Scores for Treated vs Control,
    BEFORE and AFTER matching.
    """
    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    # 1. Raw (Unmatched) Data
    # We plot distinct curves for Treated (1) and Control (0)
    sns.kdeplot(
        data=data[data[treatment_col] == 1], x='propensity_score',
        fill=True, color="orange", alpha=0.3, label='Treated', ax=axes[0]
    )
    sns.kdeplot(
        data=data[data[treatment_col] == 0], x='propensity_score',
        fill=True, color="blue", alpha=0.3, label='Control', ax=axes[0]
    )
    axes[0].set_title("Before Matching", fontweight='bold')
    axes[0].set_xlabel("Propensity Score")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # 2. Matched Data (Weighted)
    # We use the 'weights' column to simulate the matched distribution
    # Note: KDE with weights is supported in newer seaborn/matplotlib versions.
    # If weights are integer (frequency), we can use repeat. For float weights, we rely on `weights=` param.
    
    # Check if 'weights' is in data, otherwise assume the passed series
    w = weights if weights is not None else data['weights']
    
    # Filter for positive weights only to show the "Matched" set
    matched_data = data[w > 0]
    matched_weights = w[w > 0]
    
    sns.kdeplot(
        data=matched_data[matched_data[treatment_col] == 1], x='propensity_score',
        weights=matched_weights[matched_data[treatment_col] == 1],
        fill=True, color="orange", alpha=0.3, label='Treated', ax=axes[1]
    )
    sns.kdeplot(
        data=matched_data[matched_data[treatment_col] == 0], x='propensity_score',
        weights=matched_weights[matched_data[treatment_col] == 0],
        fill=True, color="blue", alpha=0.3, label='Control', ax=axes[1]
    )
    axes[1].set_title("After Matching", fontweight='bold')
    axes[1].set_xlabel("Propensity Score")
    
    plt.suptitle(title, fontweight='bold', y=1.05)
    sns.despine()
    plt.tight_layout()
    plt.show()

def ecdf_plot(
    data: pd.DataFrame,
    var_name: str,
    treatment_col: str,
    weights: pd.Series,
    title: str = None
):
    """
    Plots Empirical Cumulative Distribution Function (eCDF) for a specific covariate.
    Visualizes how well the distributions align after matching.
    """
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if title is None:
        title = f"eCDF Balance: {var_name}"

    # Unmatched (Dashed Lines)
    sns.ecdfplot(data=data, x=var_name, hue=treatment_col, 
                 palette=["blue", "orange"], linestyle="--", alpha=0.5, linewidth=2, ax=ax, legend=False)

    # Matched (Solid Lines)
    # Seaborn ecdfplot supports 'weights'
    matched_data = data[weights > 0]
    matched_weights = weights[weights > 0]
    
    sns.ecdfplot(data=matched_data, x=var_name, hue=treatment_col, weights=matched_weights,
                 palette=["blue", "orange"], linestyle="-", linewidth=3, ax=ax)

    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='orange', lw=2, linestyle='--'),
        Line2D([0], [0], color='blue', lw=2, linestyle='--'),
        Line2D([0], [0], color='orange', lw=3, linestyle='-'),
        Line2D([0], [0], color='blue', lw=3, linestyle='-')
    ]
    ax.legend(custom_lines, ['Treated (Raw)', 'Control (Raw)', 'Treated (Matched)', 'Control (Matched)'])
    
    ax.set_title(title, fontweight='bold')
    sns.despine()
    plt.show()