import matplotlib.pyplot as plt
import seaborn as sns

class FeatureAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def encode_categorical(self):
        """Encodes categorical variables for correlation analysis."""
        # Simple binary encoding for Gender
        if 'Gender' in self.df.columns:
            self.df['Gender_Num'] = self.df['Gender'].map({'Male': 0, 'Female': 1})
        return self.df

    def compute_correlations(self, columns):
        """Computes correlation matrix for specified columns."""
        # Ensure columns exist
        available_cols = [c for c in columns if c in self.df.columns]
        corr_matrix = self.df[available_cols].corr()
        return corr_matrix

    def plot_heatmap(self, corr_matrix):
        """Plots the correlation heatmap."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')