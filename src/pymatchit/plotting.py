# File: src/pymatchit/plotting.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Optional, Dict, Tuple

def love_plot(
    summary_dict: dict, 
    threshold: float = 0.1, 
    colors: Tuple[str, str] = ("#e74c3c", "#3498db"),
    var_names: Optional[Dict[str, str]] = None,
    title: str = "Covariate Balance (Love Plot)"
):
    """
    Generates a publication-ready Love Plot (Covariate Balance Plot).
    
    Args:
        summary_dict (dict): Output from MatchIt.summary().
        threshold (float): The threshold for good balance (default 0.1).
        colors (tuple): Tuple of two color hex codes (Unmatched, Matched).
        var_names (dict): Optional mapping of variable names (e.g., {'age': 'Age (Years)'}).
        title (str): Custom title for the plot.
    """
    # 1. Set Theme
    sns.set_theme(style="whitegrid", context="talk")
    
    unmatched_df = summary_dict['unmatched']
    matched_df = summary_dict['matched']
    
    # Create Plot Data
    plot_data = pd.DataFrame({
        'Unmatched': unmatched_df['Std. Mean Diff.'].abs(),
        'Matched': matched_df['Std. Mean Diff.'].abs()
    })
    
    # Apply Variable Renaming if provided
    if var_names:
        plot_data = plot_data.rename(index=var_names)
    
    # Sort by Unmatched values
    plot_data = plot_data.sort_values(by='Unmatched', ascending=True)
    
    covariates = plot_data.index
    y_pos = range(len(covariates))

    # 2. Create Figure
    fig, ax = plt.subplots(figsize=(10, len(covariates) * 0.8 + 2)) # Dynamic height

    color_unmatched, color_matched = colors

    # 3. Draw Dumbbell Lines
    for i, cov in enumerate(covariates):
        ax.hlines(y=i, 
                  xmin=plot_data.loc[cov, 'Matched'], 
                  xmax=plot_data.loc[cov, 'Unmatched'], 
                  color='grey', alpha=0.4, linewidth=2, zorder=1)

    # 4. Draw Points
    ax.scatter(plot_data['Unmatched'], y_pos, 
               color=color_unmatched, label='Unmatched', 
               s=150, edgecolor='white', linewidth=1.5, zorder=3)
    
    ax.scatter(plot_data['Matched'], y_pos, 
               color=color_matched, label='Matched', 
               s=150, edgecolor='white', linewidth=1.5, zorder=3)

    # 5. Reference Lines
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, zorder=0)
    ax.axvline(x=threshold, color='grey', linestyle='--', linewidth=1.5, 
               label=f'Threshold ({threshold})', zorder=0)

    # 6. Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates, fontweight='medium')
    ax.set_xlabel("Absolute Standardized Mean Difference", fontweight='medium', labelpad=15)
    ax.set_title(title, fontweight='bold', y=1.02)
    
    # Fix axes limits
    max_val = max(plot_data['Unmatched'].max(), plot_data['Matched'].max())
    ax.set_xlim(left=-0.02, right=max_val * 1.05)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='white')

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()