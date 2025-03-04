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
        self.df['side'] = self.df['protein_id'].str.contains('XXX_', na=False).map({
            True: 'Decoys', 
            False: 'Non-decoys'
        })
        
        # Add log transformed scores
        self.df['log2_match_score'] = np.log2(self.df['match_score'].clip(lower=1e-9))
        
        # Clean up data
        self.df = self.df.replace([-np.inf, np.inf], np.nan).dropna(subset=['log2_match_score'])

    def plot_score_distributions(self) -> None:
        """Plot score distributions using KDE plots."""
        g = sns.FacetGrid(
            self.df, 
            col='experiment_name', 
            row='scoring_function', 
            hue='side',
            palette={'Decoys': 'blue', 'Non-decoys': 'red'}, 
            height=4, 
            aspect=1.5, 
            sharex=True, 
            sharey=True
        )
        
        g.map(sns.kdeplot, 'log2_match_score', fill=True, alpha=0.5, linewidth=1.5)
        g.set_axis_labels('log2(Match Score)', 'Density')
        g.set_titles(col_template='Experiment: {col_name}', row_template='Scoring: {row_name}')
        g.add_legend(title='Category')
        plt.tight_layout()
        return g

    def plot_violin_distributions(self) -> None:
        """Plot score distributions using violin plots."""
        fig = px.violin(
            self.df, 
            x='scoring_function', 
            y='match_score', 
            color='side', 
            facet_col='experiment_name', 
            box=True,
            points="outliers", 
            title='Match Score Distribution: Decoys vs. Non-decoys',
            category_orders={
                'side': ['Decoys', 'Non-decoys'],
                'scoring_function': sorted(self.df['scoring_function'].unique()),
                'experiment_name': sorted(self.df['experiment_name'].unique())
            },
            labels={
                'match_score': 'Match Score', 
                'scoring_function': 'Scoring Function'
            }, 
            log_y=True
        )
        
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
