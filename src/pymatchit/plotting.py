# File: src/pymatchit/plotting.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Optional, Dict, Tuple, List

def love_plot(
    summary_dict: dict, 
    threshold: float = 0.1, 
    var_names: Optional[Dict[str, str]] = None,
    **kwargs
):
    """
    Generates a publication-ready Love Plot (Standardized Mean Differences).
    Returns a matplotlib Axes object for further customization.
    """
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    
    unmatched_df = summary_dict['unmatched']
    matched_df = summary_dict['matched']
    
    plot_data = pd.DataFrame({
        'Unmatched': unmatched_df['Std. Mean Diff.'].abs(),
        'Matched': matched_df['Std. Mean Diff.'].abs()
    })
    
    if var_names:
        plot_data = plot_data.rename(index=var_names)
    
    plot_data = plot_data.dropna()
    plot_data = plot_data.sort_values(by='Unmatched', ascending=True)
    covariates = plot_data.index
    y_pos = range(len(covariates))

    # Kwargs extrahieren oder mit Defaults füllen
    colors = kwargs.pop('colors', ("#1f77b4", "#ff7f0e"))
    figsize = kwargs.pop('figsize', (8, max(4, len(covariates) * 0.5)))
    title = kwargs.pop('title', "Covariate Balance")

    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    color_unmatched, color_matched = colors

    for i, cov in enumerate(covariates):
        val_unmatched = plot_data.loc[cov, 'Unmatched']
        val_matched = plot_data.loc[cov, 'Matched']
        ax.hlines(y=i, 
                  xmin=min(val_matched, val_unmatched), 
                  xmax=max(val_matched, val_unmatched), 
                  color='#cccccc', linewidth=1.5, zorder=1)

    ax.scatter(plot_data['Unmatched'], y_pos, 
               color=color_unmatched, label='Unmatched', 
               marker='o', s=70, alpha=0.8, edgecolor='none', zorder=2)
    
    ax.scatter(plot_data['Matched'], y_pos, 
               color=color_matched, label='Matched', 
               marker='o', s=100, edgecolor='white', linewidth=0.8, zorder=3)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, zorder=0)
    ax.axvline(x=threshold, color='#555555', linestyle='--', linewidth=1.2, 
               label=f'Threshold ({threshold})', zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates, color='black')
    ax.set_xlabel("Absolute Standardized Mean Difference", color='black', labelpad=10)
    
    if title:
        ax.set_title(title, pad=15)
    
    max_val = max(plot_data['Unmatched'].max(), plot_data['Matched'].max())
    if pd.isna(max_val): max_val = 0.5
    ax.set_xlim(left=-0.02, right=max_val * 1.05)
    
    ax.xaxis.grid(True, linestyle='-', color='#eeeeee', zorder=0)
    ax.yaxis.grid(False)
    
    sns.despine(left=True, bottom=False, trim=True)
    ax.tick_params(axis='y', length=0)
    
    ax.legend(loc='lower right', frameon=False, title="")
    
    plt.tight_layout()
    return ax

def propensity_plot(
    data: pd.DataFrame,
    treatment_col: str,
    weights: pd.Series,
    **kwargs
):
    """
    Plots the density of Propensity Scores for Treated vs Control.
    Returns an array of matplotlib Axes objects.
    """
    sns.set_theme(style="white", context="talk")
    
    title = kwargs.pop('title', "Propensity Score Distribution (Common Support)")
    figsize = kwargs.pop('figsize', (16, 6))
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True, **kwargs)

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

    w = weights if weights is not None else data['weights']
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
    
    if title:
        plt.suptitle(title, fontweight='bold', y=1.05)
        
    sns.despine()
    plt.tight_layout()
    return axes

def ecdf_plot(
    data: pd.DataFrame,
    var_name: str,
    treatment_col: str,
    weights: pd.Series,
    **kwargs
):
    """
    Plots Empirical Cumulative Distribution Function (eCDF) for a specific covariate.
    Returns a matplotlib Axes object.
    """
    sns.set_theme(style="whitegrid", context="talk")
    
    title = kwargs.pop('title', f"eCDF Balance: {var_name}")
    figsize = kwargs.pop('figsize', (10, 6))
    
    fig, ax = plt.subplots(figsize=figsize, **kwargs)

    sns.ecdfplot(data=data, x=var_name, hue=treatment_col, 
                 palette=["blue", "orange"], linestyle="--", alpha=0.5, linewidth=2, ax=ax, legend=False)

    matched_data = data[weights > 0]
    matched_weights = weights[weights > 0]
    
    sns.ecdfplot(data=matched_data, x=var_name, hue=treatment_col, weights=matched_weights,
                 palette=["blue", "orange"], linestyle="-", linewidth=3, ax=ax)

    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='orange', lw=2, linestyle='--'),
        Line2D([0], [0], color='blue', lw=2, linestyle='--'),
        Line2D([0], [0], color='orange', lw=3, linestyle='-'),
        Line2D([0], [0], color='blue', lw=3, linestyle='-')
    ]
    ax.legend(custom_lines, ['Treated (Raw)', 'Control (Raw)', 'Treated (Matched)', 'Control (Matched)'])
    
    if title:
        ax.set_title(title, fontweight='bold')
        
    sns.despine()
    return ax