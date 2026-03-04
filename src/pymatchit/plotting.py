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
    sns.set_theme(style="white", context="talk")

    COLOR_TREATED = "#ff7f0e"
    COLOR_CONTROL = "#1f77b4"

    title = kwargs.pop('title', f"eCDF Balance: {var_name}")
    figsize = kwargs.pop('figsize', (10, 6))
    
    fig, ax = plt.subplots(figsize=figsize, **kwargs)

    matched_data = data[weights > 0]
    matched_weights = weights[weights > 0]

    # Draw matched lines first so raw (dashed) lines render on top and stay visible
    sns.ecdfplot(data=matched_data, x=var_name, hue=treatment_col, weights=matched_weights,
                 palette=[COLOR_CONTROL, COLOR_TREATED], linestyle="-", linewidth=2.5,
                 ax=ax, legend=False)

    sns.ecdfplot(data=data, x=var_name, hue=treatment_col,
                 palette=[COLOR_CONTROL, COLOR_TREATED], linestyle="--", alpha=0.7, linewidth=2,
                 ax=ax, legend=False)

    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color=COLOR_TREATED, lw=2, linestyle='--'),
        Line2D([0], [0], color=COLOR_CONTROL, lw=2, linestyle='--'),
        Line2D([0], [0], color=COLOR_TREATED, lw=2.5, linestyle='-'),
        Line2D([0], [0], color=COLOR_CONTROL, lw=2.5, linestyle='-')
    ]
    ax.legend(custom_lines, ['Treated (Raw)', 'Control (Raw)', 'Treated (Matched)', 'Control (Matched)'],
              frameon=False)

    ax.xaxis.grid(True, linestyle='-', color='#eeeeee', zorder=0)
    ax.yaxis.grid(False)

    if title:
        ax.set_title(title, pad=15)

    sns.despine()
    plt.tight_layout()
    return ax


def qq_plot(
    data: pd.DataFrame,
    var_name: str,
    treatment_col: str,
    weights: pd.Series,
    **kwargs
):
    """
    Plots a Quantile-Quantile (QQ) plot comparing treated vs control distributions.
    Shows both raw (before matching) and matched (after matching) distributions.

    If the points fall along the 45-degree line, the distributions are similar.
    Returns a matplotlib Axes object.
    """
    sns.set_theme(style="white", context="talk")

    COLOR_BEFORE = "#1f77b4"   # unmatched / before – same blue as love_plot unmatched
    COLOR_AFTER  = "#ff7f0e"   # matched / after  – same orange as love_plot matched

    title = kwargs.pop('title', f"QQ Plot: {var_name}")
    figsize = kwargs.pop('figsize', (10, 10))
    n_quantiles = kwargs.pop('n_quantiles', 100)

    fig, axes = plt.subplots(1, 2, figsize=figsize, **kwargs)

    quantiles = np.linspace(0, 1, n_quantiles + 1)

    # --- Before Matching (Raw) ---
    treated_vals = data.loc[data[treatment_col] == 1, var_name].values
    control_vals = data.loc[data[treatment_col] == 0, var_name].values

    q_treated = np.quantile(treated_vals, quantiles)
    q_control = np.quantile(control_vals, quantiles)

    all_vals = np.concatenate([q_treated, q_control])
    ax_min, ax_max = all_vals.min(), all_vals.max()
    margin = (ax_max - ax_min) * 0.05

    axes[0].scatter(q_control, q_treated, color=COLOR_BEFORE, alpha=0.6, s=30, edgecolor='none')
    axes[0].plot([ax_min - margin, ax_max + margin], [ax_min - margin, ax_max + margin],
                 color='#555555', linestyle='--', linewidth=1, alpha=0.7, label='45° line')
    axes[0].set_xlabel(f"Control Quantiles ({var_name})")
    axes[0].set_ylabel(f"Treated Quantiles ({var_name})")
    axes[0].set_title("Before Matching", fontweight='bold')
    axes[0].set_xlim(ax_min - margin, ax_max + margin)
    axes[0].set_ylim(ax_min - margin, ax_max + margin)
    axes[0].set_aspect('equal')
    axes[0].xaxis.grid(True, linestyle='-', color='#eeeeee', zorder=0)
    axes[0].yaxis.grid(True, linestyle='-', color='#eeeeee', zorder=0)
    axes[0].legend(frameon=False)

    # --- After Matching ---
    matched_mask = weights > 0
    matched_data = data[matched_mask]
    matched_w = weights[matched_mask]

    treated_matched = matched_data[matched_data[treatment_col] == 1]
    control_matched = matched_data[matched_data[treatment_col] == 0]

    if len(treated_matched) > 0 and len(control_matched) > 0:
        t_vals = treated_matched[var_name].values
        t_weights = matched_w.loc[treated_matched.index].values
        c_vals = control_matched[var_name].values
        c_weights = matched_w.loc[control_matched.index].values

        q_treated_m = _weighted_quantiles(t_vals, t_weights, quantiles)
        q_control_m = _weighted_quantiles(c_vals, c_weights, quantiles)

        axes[1].scatter(q_control_m, q_treated_m, color=COLOR_AFTER, alpha=0.7, s=30, edgecolor='none')
        axes[1].plot([ax_min - margin, ax_max + margin], [ax_min - margin, ax_max + margin],
                     color='#555555', linestyle='--', linewidth=1, alpha=0.7, label='45° line')

    axes[1].set_xlabel(f"Control Quantiles ({var_name})")
    axes[1].set_ylabel(f"Treated Quantiles ({var_name})")
    axes[1].set_title("After Matching", fontweight='bold')
    axes[1].set_xlim(ax_min - margin, ax_max + margin)
    axes[1].set_ylim(ax_min - margin, ax_max + margin)
    axes[1].set_aspect('equal')
    axes[1].xaxis.grid(True, linestyle='-', color='#eeeeee', zorder=0)
    axes[1].yaxis.grid(True, linestyle='-', color='#eeeeee', zorder=0)
    axes[1].legend(frameon=False)

    if title:
        fig.suptitle(title, y=1.02)

    sns.despine()
    plt.tight_layout()
    return axes


def jitter_plot(
    data: pd.DataFrame,
    treatment_col: str,
    weights: pd.Series,
    **kwargs
):
    """
    Creates a jitter plot showing propensity scores for all units,
    distinguishing matched from unmatched units.

    Returns a matplotlib Axes object.
    """
    sns.set_theme(style="white", context="talk")

    COLOR_TREATED = "#ff7f0e"
    COLOR_CONTROL = "#1f77b4"

    title = kwargs.pop('title', "Propensity Score Jitter Plot")
    figsize = kwargs.pop('figsize', (12, 6))

    fig, ax = plt.subplots(figsize=figsize, **kwargs)

    matched_mask = weights > 0

    # Treated units
    treated = data[data[treatment_col] == 1]
    treated_matched = treated[matched_mask.loc[treated.index]]
    treated_unmatched = treated[~matched_mask.loc[treated.index]]

    # Control units
    control = data[data[treatment_col] == 0]
    control_matched = control[matched_mask.loc[control.index]]
    control_unmatched = control[~matched_mask.loc[control.index]]

    # Add jitter to y-axis
    rng = np.random.RandomState(42)
    jitter_amount = 0.15

    # Plot unmatched (hollow, lower opacity)
    if len(treated_unmatched) > 0:
        y_jitter = 1.0 + rng.uniform(-jitter_amount, jitter_amount, len(treated_unmatched))
        ax.scatter(treated_unmatched['propensity_score'], y_jitter,
                   color='none', edgecolor=COLOR_TREATED, alpha=0.3, s=20,
                   linewidth=0.8, label='Treated (Unmatched)')

    if len(control_unmatched) > 0:
        y_jitter = 0.0 + rng.uniform(-jitter_amount, jitter_amount, len(control_unmatched))
        ax.scatter(control_unmatched['propensity_score'], y_jitter,
                   color='none', edgecolor=COLOR_CONTROL, alpha=0.3, s=20,
                   linewidth=0.8, label='Control (Unmatched)')

    # Plot matched (solid)
    if len(treated_matched) > 0:
        y_jitter = 1.0 + rng.uniform(-jitter_amount, jitter_amount, len(treated_matched))
        ax.scatter(treated_matched['propensity_score'], y_jitter,
                   color=COLOR_TREATED, alpha=0.8, s=40, edgecolor='white', linewidth=0.5,
                   label='Treated (Matched)')

    if len(control_matched) > 0:
        y_jitter = 0.0 + rng.uniform(-jitter_amount, jitter_amount, len(control_matched))
        ax.scatter(control_matched['propensity_score'], y_jitter,
                   color=COLOR_CONTROL, alpha=0.8, s=40, edgecolor='white', linewidth=0.5,
                   label='Control (Matched)')

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Control', 'Treated'])
    ax.set_xlabel("Propensity Score")
    ax.set_ylim(-0.5, 1.5)

    ax.xaxis.grid(True, linestyle='-', color='#eeeeee', zorder=0)
    ax.yaxis.grid(False)

    if title:
        ax.set_title(title, pad=15)

    ax.legend(loc='upper right', frameon=False, fontsize=9, ncol=2)

    sns.despine(left=True, bottom=False, trim=True)
    ax.tick_params(axis='y', length=0)
    plt.tight_layout()
    return ax


def _weighted_quantiles(values, weights, quantiles):
    """Compute weighted quantiles."""
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    
    cumulative_weights = np.cumsum(weights)
    cumulative_weights /= cumulative_weights[-1]
    
    return np.interp(quantiles, cumulative_weights, values)