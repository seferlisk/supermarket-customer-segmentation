import matplotlib.pyplot as plt
import numpy as np


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
        plt.show()
        # plt.savefig('optimization_plots.png')

    def plot_clusters_3d(self, X_original, labels, columns):
        """Visualizes clusters in 3D (using original values for readability)."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Columns: Age, Income, Score
        # We assume the order passed in 'columns'
        x_col, y_col, z_col = columns[0], columns[1], columns[2]

        # Use a color palette
        colors = plt.cm.viridis(np.linspace(0, 1, len(set(labels))))

        scatter = ax.scatter(X_original[x_col], X_original[y_col], X_original[z_col],
                             c=labels, cmap='viridis', s=60, alpha=0.6)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title('Customer Segments (3D)')

        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)

        plt.tight_layout()
        # plt.savefig('cluster_segmentation_3d.png')