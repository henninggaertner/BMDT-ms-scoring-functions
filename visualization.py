import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


class SpectrumVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare data for visualization."""
        # Add decoy identification
        self.df['side'] = self.df['protein_id'].str.contains('XXX_', na=False).map(
            {True: 'Decoys', False: 'Targets'})

        # Add log transformed scores
        self.df['log2_match_score'] = np.log2(self.df['match_score'].clip(lower=-0.0))

        # Clean up data
        self.df['log2_match_score'] = self.df['log2_match_score'].replace([-np.inf, np.inf], np.nan).dropna()

    def plot_score_distributions(self, title :str, custom_xlims=None, custom_ylims=None) -> None:
        """
        Plot score distributions using KDE plots with custom axis limits per scoring function.

        Parameters:
        -----------
        custom_xlims : dict, optional
            Dictionary mapping scoring function names to (min, max) tuples for x-axis limits.
            Example: {'XCorr': (-10, 10), 'HyperScore': (-5, 25)}
        custom_ylims : dict, optional
            Dictionary mapping scoring function names to (min, max) tuples for y-axis limits.
            Example: {'XCorr': (0, 0.5), 'HyperScore': (0, 0.3)}
        """
        # Create FacetGrid with sharex and sharey set to False
        sns.set_style("whitegrid")
        g = sns.FacetGrid(self.df, col='experiment_name', row='scoring_function', hue='side',
            palette={'Decoys': 'blue', 'Targets': 'red'}, height=4, aspect=1.5, sharex=False,
            # Allow different x-axis ranges
            sharey=False  # Allow different y-axis ranges
        )

        # Map the plot function
        g.map(sns.kdeplot, 'log2_match_score', bw_adjust=1, fill=True, alpha=0.5, linewidth=1.5)
        g.set_axis_labels('log2(Match Score)', 'Density', fontsize=12)
        g.set_titles(col_template='Experiment: {col_name}', row_template='Scoring: {row_name}', size=12)

        # Add title for the entire figure
        g.fig.suptitle(title, fontsize=16, y=1.02)

        # Add legend to the top right
        g.add_legend(title='Category', loc='upper right', bbox_to_anchor=(1.05, 0.5), frameon=True)

        # Calculate sensible limits for each scoring function if not provided
        if custom_xlims is None:
            custom_xlims = {}
            for func in self.df['scoring_function'].unique():
                subset = self.df[self.df['scoring_function'] == func]['log2_match_score']
                # Set limits to cover 99% of the data to avoid extreme outliers stretching the plot
                min_val = np.percentile(subset, 0.5)
                max_val = np.percentile(subset, 99.5)
                # Add some padding
                padding = (max_val - min_val) * 0.05
                custom_xlims[func] = (min_val - padding, max_val + padding)

        # Calculate y-axis limits for each scoring function if not provided
        if custom_ylims is None:
            custom_ylims = {}
            # First, we need to get the KDE values for each scoring function
            for func in self.df['scoring_function'].unique():
                # Split the data by scoring function and side
                max_density = 0

                # Loop through each experiment to find the maximum density across all experiments
                for exp in self.df['experiment_name'].unique():
                    # Get data for the current scoring function, experiment, and both sides
                    for side in ['Decoys', 'Targets']:
                        subset = self.df[(self.df['scoring_function'] == func) & (self.df['experiment_name'] == exp) & (
                                    self.df['side'] == side)]['log2_match_score']

                        if len(subset) > 0:
                            # Calculate the KDE
                            kde = sns.kdeplot(subset).get_lines()[0].get_ydata()
                            current_max = kde.max() if len(kde) > 0 else 0
                            max_density = max(max_density, current_max)

                # Add 10% padding to the top
                custom_ylims[func] = (0, max_density * 1.1)

        # Get the row values (scoring functions)
        row_values = g.row_names

        # Apply the custom axis limits
        for i, row_val in enumerate(row_values):
            for j in range(len(g.col_names)):
                ax = g.axes[i, j]
                # Set x-axis limits
                ax.set_xlim(custom_xlims.get(row_val, (None, None)))
                # Set y-axis limits
                ax.set_ylim(custom_ylims.get(row_val, (None, None)))

        plt.tight_layout()
        return g

    def plot_violin_distributions(self) -> None:
        """Plot score distributions using violin plots."""
        fig = px.violin(self.df, x='scoring_function', y='match_score', color='side', facet_col='experiment_name',
            box=True, points="outliers", title='Match Score Distribution: Decoys vs. Targets',
            category_orders={'side': ['Decoys', 'Targets'],
                'scoring_function': sorted(self.df['scoring_function'].unique()),
                'experiment_name': sorted(self.df['experiment_name'].unique())},
            labels={'match_score': 'Match Score', 'scoring_function': 'Scoring Function'}, log_y=True)

        fig.update_layout(violinmode='group', violingap=0, violingroupgap=0)
        return fig

    def plot_mass_score_scatter(self) -> None:
        """Plot peptide mass vs match score scatter plot."""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['peptide_mass'], self.df['match_score'], s=32, alpha=.8)
        plt.xlabel('Peptide Mass')
        plt.ylabel('Match Score')
        plt.gca().spines[['top', 'right']].set_visible(False)
        return plt.gcf()

    def plot_score_histogram(self) -> None:
        """Plot histogram of match scores."""
        plt.figure(figsize=(10, 6))
        self.df['match_score'].hist(bins=20)
        plt.xlabel('Match Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Match Scores')
        plt.gca().spines[['top', 'right']].set_visible(False)
        return plt.gcf()

    def get_confident_matches(self, threshold: float = 0.5) -> pd.DataFrame:
        """Get confidently matched spectra above threshold."""
        confident_matches = self.df[self.df['match_score'] > threshold].copy()
        confident_matches.sort_values(by=['match_score'], ascending=False, inplace=True)
        return confident_matches