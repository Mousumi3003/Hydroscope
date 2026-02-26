import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
 
 
class HydroPlotter:
    def __init__(self, font='Arial'):
        plt.style.use('default')
        plt.rcParams['font.family'] = font

    def tsplot_vars(self, data, variables=None, title=None, filename='timeseries.tiff',
                             ncols=2):
        """
        Plot time series of selected variables in a grid layout.

        Parameters:
        - data: DataFrame with time series data.
        - variables: list of variables to plot. If None, all columns are used.
        - title: Optional plot title.
        - figurepath: Optional path to save the figure.
        - filename: Name of the output file if saving.
        - ncols: Number of columns in the subplot grid.
        """
        if variables is None:
            variables = data.columns.tolist()

        n_vars = len(variables)
        nrows = math.ceil(n_vars / ncols)

        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.5 * nrows), sharex=True)
        axs = axs.flatten()  # Flatten in case of 2D array

        for i, var in enumerate(variables):
            axs[i].plot(data.index, data[var], color='gray', linewidth=0.8)
            axs[i].set_ylabel(var, fontsize=11)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)

        # Turn off empty subplots
        for j in range(i+1, len(axs)):
            axs[j].axis('off')

        axs[-1].set_xlabel('Date', fontsize=11)

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout(h_pad=2)
        if filename:
            plt.savefig(f"{filename}", dpi=300, bbox_inches='tight')
        plt.show()
        

    def compare_tsplot_vars(self, obs_data, model_data, variables=None, labels=('Observed', 'Modeled'),ylabels=False,
                            title=None, filename='comparison.tiff', ncols=3, colors=('black', 'red'),commonLegend=True,legendOn=False):
        """
        Overlay observed and modeled time series for given variables.

        Parameters:
        - obs_data: DataFrame with observed time series data.
        - model_data: DataFrame with modeled time series data.
        - variables: list of variables to plot. If None, uses common columns.
        - labels: Tuple of labels for legend (observed, modeled).
        - title: Optional plot title.
        - filename: Output filename (set to False to disable saving).
        - ncols: Number of columns in the subplot grid.
        - colors: Tuple of colors for observed and modeled lines.
        """
        import math

        if variables is None:
            variables = list(set(obs_data.columns) & set(model_data.columns))

        n_vars = len(variables)
        nrows = 1

        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.5 * nrows))
        axs = axs.flatten()

        for i, var in enumerate(variables):
            axs[i].plot(obs_data.index, obs_data[var], color=colors[0], label=labels[0], linewidth=1)
            axs[i].plot(model_data.index, model_data[var], color=colors[1], label=labels[1], linewidth=1)
            axs[i].set_ylabel(var, fontsize=11)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            if legendOn:
                axs[i].legend(fontsize=9)
            if  ylabels:
                axs[i].set_ylabel(ylabels[i], fontsize=11)

        for j in range(i+1, len(axs)):
            axs[j].axis('off')
            
        

        axs[-1].set_xlabel('Date', fontsize=11)

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        if commonLegend:
            axs[i].legend(fontsize=9)
            

        plt.tight_layout(h_pad=2)
        if filename:
            plt.savefig(f"{filename}")
        plt.show()

    def compare_scatterplot_vars(self, obs_data, model_data, variables=None,
                             labels=('Observed', 'Modeled'), title=None,titles=False,
                             filename='scatter_comparison.tiff', ncols=3,
                             colors=('dodgerblue',), alpha=0.6, add_1to1=True,
                             show_trendline=True):
        """
        Scatter plot comparing observed vs modeled data for selected variables,
        with optional 1:1 line, trendline, and R² annotation.

        Parameters:
        - obs_data: DataFrame with observed values.
        - model_data: DataFrame with modeled values.
        - variables: List of variable names. If None, uses common columns.
        - labels: Tuple with axis labels (x=obs, y=mod).
        - title: Optional plot title.
        - filename: Output filename (set to False to disable saving).
        - ncols: Number of columns in the subplot grid.
        - colors: Tuple of colors for scatter points.
        - alpha: Transparency of scatter points.
        - add_1to1: Whether to add a 1:1 reference line.
        - show_trendline: Whether to add a linear regression line and R².
        """
        import math

        import numpy as np

        if variables is None:
            variables = list(set(obs_data.columns) & set(model_data.columns))

        n_vars = len(variables)
        nrows = math.ceil(n_vars / ncols)

        fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows))
        axs = axs.flatten()

        for i, var in enumerate(variables):
            x = obs_data[var]
            y = model_data[var]

            # Drop NaNs
            valid = x.notna() & y.notna()
            x = x[valid].values.reshape(-1, 1)
            y = y[valid].values.reshape(-1, 1)

            axs[i].scatter(x, y, color=colors[0], alpha=alpha, s=15, edgecolor='k', linewidth=0.2)
            r2 = np.corrcoef(x.flatten(),y.flatten())[0,1]
        
            if add_1to1:
                min_val = min(x.min(), y.min())
                max_val = max(x.max(), y.max())
                axs[i].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
               
                axs[i].text(0.05, 0.90, f"$R^2$ = {r2:.2f}",
                            transform=axs[i].transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle="round", fc="w", ec="gray", alpha=0.6))
                
            axs[i].set_xlabel(f"{labels[0]}", fontsize=11)
            axs[i].set_ylabel(f"{labels[1]}", fontsize=11)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
          
            if titles:
                axs[i].set_title(titles[i])

        for j in range(i+1, len(axs)):
            axs[j].axis('off')
            

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout(h_pad=2)
        if filename:
            plt.savefig(f"{filename}")
        plt.show()


    def heatmap(self, data, variables=None, title=None, filename='corrplot.tiff'):
        """
        Plot correlation heatmap of selected variables.
        """
        if variables is None:
            variables = data.columns.tolist()

        corr = data[variables].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=(0.8 * len(variables) + 2, 0.8 * len(variables) + 2))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    mask=mask, xticklabels=variables, yticklabels=variables)

        if title:
            plt.title(title, fontsize=13, pad=10)

        plt.tight_layout()
        if filename:
            plt.savefig(f"{filename}", dpi=300, bbox_inches='tight')
        plt.show()
        

    def lineplot(self,
        merged_results,
        primary_metric,
        secondary_metrics,
        variables,
        param_col="epsilon",
        figsize=(6, 4),
        markers=True,
        colors=['b','r']
    ):
        """
        Plots selected primary and secondary metrics for given variables.
        
        Parameters
        ----------
        merged_results : pd.DataFrame
            DataFrame containing 'variable', param_col, primary and secondary metric columns.
        primary_metric : str
            Name of the metric to plot on the primary y-axis.
        secondary_metrics : list of str
            Names of metrics to plot on the secondary y-axis (max 2 allowed).
        variables : list of str
            Variables to include in the plot (from merged_results['variable']).
        param_col : str
            Column name for the x-axis parameter.
        figsize : tuple
            Size of each subplot (width, height per variable).
        markers : bool
            Whether to include markers on the lines.
        """
    
        if len(secondary_metrics) > 2:
            raise ValueError("secondary_metrics can have at most 2 items.")
    
        n_vars = len(variables)
        fig, axes = plt.subplots(
            nrows=n_vars, ncols=1,
            figsize=figsize,
            sharex=True
        )
    
        if n_vars == 1:
            axes = [axes]  # Make iterable if only one subplot
    
        for i, var in enumerate(variables):
            ax1 = axes[i]
            data_var = merged_results[merged_results['variable'] == var].sort_values(param_col)
    
            # Primary Y-axis plot
            ax1.set_ylabel(primary_metric)
            style = '-o' if markers else '-'
            ax1.plot(data_var[param_col], data_var[primary_metric],
                     style, color=colors[0], label=primary_metric)
            ax1.tick_params(axis='y')
    
            # Secondary Y-axis plot
            ax2 = ax1.twinx()
            c = colors[1:]
            for j, sec_metric in enumerate(secondary_metrics):
                ax2.plot(data_var[param_col], data_var[sec_metric],
                         style, color=c[j], label=sec_metric)
            ax2.set_ylabel(', '.join(secondary_metrics))
            
            # Legends and titles
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, ncol=len(lines1+lines2))
            ax1.set_title(f"Variable: {var}")
    
            # Remove top/right spines
            ax1.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)
    
        axes[-1].set_xlabel(param_col)
        fig.tight_layout()
        plt.show()


    
    def plot_metric_heatmap(self,
        results_dict,
        source_vars,
        param_col='None',
        metric_col='Af',
        cmap='coolwarm',
        figsize=(12, 3),
        annot=False,
        center=0,
        fmt=".2f"
    ):
        """
        Plot hydrology performance metrics as heatmaps for each method.
    
        Parameters
        ----------
        results_dict : dict
            Dictionary of {method_name: DataFrame}.
            Each DataFrame must have columns: ['variable', param_col, metric_col]
        source_vars : list
            List of source variable names (without 'lagged_' prefix).
        param_col : str
            Column name for the parameter axis.
        metric_col : str
            Column name for the metric to color by.
        cmap : str
            Matplotlib/Seaborn colormap.
        figsize : tuple
            Size of each method's heatmap panel.
        annot : bool
            Whether to annotate each cell with numeric value.
        center : float or None
            Value to center colormap (useful for diverging color maps).
        fmt : str
            String formatting for annotations.
        """
        n_methods = len(results_dict)
        fig, axes = plt.subplots(1, n_methods, figsize=(figsize[0] * n_methods, figsize[1]), sharey=True)
    
        if n_methods == 1:
            axes = [axes]
    
        for ax, (method, df) in zip(axes, results_dict.items()):
            # Prepare pivot table
            df_clean = df.copy()
            df_clean['variable'] = df_clean['variable'].str.replace('lagged_', '', regex=False)
            df_pivot = df_clean.pivot_table(
                index='variable', columns=param_col, values=metric_col, aggfunc='mean'
            )
    
            sns.heatmap(
                df_pivot, cmap=cmap, annot=annot, fmt=fmt,
                center=center, cbar=True, ax=ax
            )
            ax.set_title(method)
            ax.set_xlabel(param_col)
            if ax == axes[0]:
                ax.set_ylabel('Variable')
    
        plt.tight_layout()
        plt.show()
    
    


    def plot_metric_overlay(self,
        results_dict,
        source_vars,
        variables=None,
        param_col='epsilon',
        metric_col='Af',
        method_colors=None,
        figsize=(10, 6),
        xlabel=None,
        ylabel=None,
        ref_line_x=None,
        ref_line_y=0,
        grid=True,
        markers=True
    ):
        """
        Plot hydrology performance metrics vs a parameter with methods overlaid.
        If multiple variables are selected, creates subplots for each variable.
    
        Parameters
        ----------
        results_dict : dict
            {method_name: DataFrame}, each with ['variable', param_col, metric_col].
        source_vars : list
            List of available source variable names (without 'lagged_' prefix).
        variables : list, str, or None
            Subset of variables (without 'lagged_' prefix). 
            If None or 'all', uses all source_vars.
        param_col : str
            Column for x-axis (e.g., 'epsilon').
        metric_col : str
            Column for metric to plot (e.g., 'Af').
        method_colors : dict or None
            {method_name: color}. If None, matplotlib picks colors.
        figsize : tuple
            Overall figure size.
        xlabel, ylabel : str or None
            Axis labels.
        ref_line_x, ref_line_y : float or None
            Reference line positions.
        grid : bool
            Show grid lines.
        markers : bool
            Show markers on lines.
        """
        # Handle default variables
        if variables is None or variables == "all":
            variables = source_vars
        else:
            variables = [v for v in variables if v in source_vars]
    
        n_vars = len(variables)
        
        # Prepare figure
        if n_vars > 1:
            fig, axes = plt.subplots(nrows=n_vars, ncols=1, figsize=figsize, sharex=True)
            if n_vars == 1:
                axes = [axes]
        else:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
    
        # Loop over variables
        for idx, var in enumerate(variables):
            ax = axes[idx]
            for method, df in results_dict.items():
                subset = df[df['variable'] == 'lagged_' + var].sort_values(param_col)
                if subset.empty:
                    continue
                style = '-' if not markers else '-o'
                ax.plot(
                    subset[param_col], subset[metric_col],
                    style,
                    label=method,
                    color=(method_colors.get(method) if method_colors and method in method_colors else None)
                )
    
            # Reference lines
            if ref_line_x is not None:
                ax.axvline(x=ref_line_x, color='k', linestyle='dashed', linewidth=1)
            if ref_line_y is not None:
                ax.axhline(y=ref_line_y, color='gray', linestyle='dotted', linewidth=1)
    
            ax.set_ylabel(ylabel if ylabel else metric_col)
            ax.set_title(f"{metric_col} vs {param_col} - {var}")
            if grid:
                ax.grid(True, linestyle='--', alpha=0.5)
    
            # Remove top/right spines for a clean look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
            ax.legend()
    
        # Shared x-axis label
        axes[-1].set_xlabel(xlabel if xlabel else param_col)
    
        fig.tight_layout()
        plt.show()
        
    def plot_ap_af_comparison(self, ap_series, af_dict, title=None, variable_labels=None, source=None,filename=None):
        """
        Plot grouped bar chart comparing Ap and Af metrics for each variable.
    
        Parameters:
        - ap_series: pd.Series with Ap values, index as variable names.
        - af_dict: dict of pd.Series, each representing a method in Af.
        - station_id: Optional string/ID for the title (e.g., USGS site).
        - variable_labels: Optional list of labels to override variable names on x-axis.
        - source: Optional iterable to determine figure width dynamically.
        """
       
    
        # --- Setup ---
        colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974']  # Muted elegant palette
        labels = variable_labels if variable_labels else ap_series.index.tolist()
        x = np.arange(len(labels))
        width = 0.2
    
        # Prepare data
        ap_series = ap_series.round(2).rename('Ap')
        af_df_rounded = pd.DataFrame({f"Af - {k}": v.round(2).values for k, v in af_dict.items()})
        plot_df = pd.concat([ap_series, af_df_rounded], axis=1)
        plot_df.columns.name = 'Metric'
        plot_df.index = labels
    
        # --- Plotting ---
        fig_width = len(source)*5 if source is not None else 7
        fig, ax = plt.subplots(figsize=(fig_width, 3), constrained_layout=True)
    
        def clean_spines(ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
        # Plot grouped bars
        n_metrics = len(plot_df.columns)
        for i, col in enumerate(plot_df.columns):
            offset = (i - n_metrics / 2) * width + width / 2
            vals = plot_df[col]
            ax.bar(x + offset, vals, width, label=col, color=colors[i % len(colors)])
            for j, val in enumerate(vals):
                y_offset = 0.03 if val >= 0 else -0.07
                ax.text(j + offset, val + y_offset, f"{val:.2f}", ha='center', fontsize=10)
    
        # Axes and styling
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        ymin = min(plot_df.min().min(), -0.5) - 0.1
        ymax = max(plot_df.max().max(), 1.1) + 0.1
        ax.set_ylim(ymin, ymax)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        clean_spines(ax)
        if filename:
            plt.savefig(f"{filename}", bbox_inches='tight')
        plt.show()

        
    def plot_performance_metrics(self, metrics_dict, variable_labels=None, title=None,filename=None):
        """
        Plots grouped bar chart of performance metrics (e.g., NSE, R², KGE) across variables.
    
        Parameters:
        - metrics_dict: dict of {metric_name: pd.Series}, each Series indexed by variable.
        - variable_labels: optional list to override x-axis labels (defaults to Series index).
        - title: optional plot title.
        """
        import numpy as np
        import matplotlib.pyplot as plt
    
        # --- Setup ---
        colors = ['#4C72B0', '#55A868', '#C44E52']  # 3-color palette
        metric_names = list(metrics_dict.keys())
        variables = metrics_dict[metric_names[0]].index.tolist()
        labels = variable_labels if variable_labels else variables
        x = np.arange(len(variables))
        width = 0.2
    
        # --- Build DataFrame ---
        plot_df = pd.DataFrame({metric: metrics_dict[metric].round(2) for metric in metric_names}, index=variables)
    
        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(max(6, len(variables) * 1.8), 4), constrained_layout=True)
    
        def clean_spines(ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
        n_metrics = len(metric_names)
        for i, metric in enumerate(metric_names):
            offset = (i - n_metrics / 2) * width + width / 2
            vals = plot_df[metric]
            ax.bar(x + offset, vals, width, label=metric.upper(), color=colors[i % len(colors)])
            for j, val in enumerate(vals):
                y_offset = 0.03 if val >= 0 else -0.07
                ax.text(j + offset, val + y_offset, f"{val:.2f}", ha='center', fontsize=10)
    
        # Axes and labels
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Metric Score', fontsize=12)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        ymin = min(plot_df.min().min(), -0.5) - 0.1
        ymax = max(plot_df.max().max(), 1.0) + 0.1
        ax.set_ylim(ymin, ymax)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(title="Metric", fontsize=10)
        clean_spines(ax)
        if filename:
            plt.savefig(f"{filename}", bbox_inches='tight')
        plt.show()

    
        
        
        
