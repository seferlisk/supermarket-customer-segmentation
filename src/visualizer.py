import matplotlib.pyplot as plt

class ClusterVisualizer:
    def plot_optimization(self, k_range, inertia, silhouette):
        """Plots the Elbow Curve and Silhouette Scores side-by-side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Elbow Plot
        ax1.plot(k_range, inertia, marker='o', linestyle='--')
        ax1.set_title('Elbow Method')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (WCSS)')
        ax1.grid(True)

        # Silhouette Plot
        ax2.plot(k_range, silhouette, marker='o', linestyle='--', color='orange')
        ax2.set_title('Silhouette Score')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('optimization_plots.png')
        plt.show()