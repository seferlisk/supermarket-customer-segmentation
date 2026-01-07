from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import itertools
import pandas as pd

class ClusterModeler:
    def __init__(self, X):
        self.X = X
        self.k_range = range(2, 11)
        self.model = None

    def find_optimal_k(self):
        """Calculates inertia and silhouette scores for k=2 to 10."""
        inertia = []
        sil_scores = []
        for k in self.k_range:
            km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            km.fit(self.X)
            inertia.append(km.inertia_)
            sil_scores.append(silhouette_score(self.X, km.labels_))
        return inertia, sil_scores

    def train_model(self, k, params=None):
        """Trains the final model, optionally accepting a dictionary of parameters."""
        if params:
            self.model = KMeans(n_clusters=k, random_state=42, **params)
        else:
            self.model = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init=10)

        return self.model.fit_predict(self.X)

    def tune_parameters(self, k, param_grid):
        """
        Runs a grid search to find the best hyperparameters for a specific k.
        Returns a DataFrame of results.
        """
        results = []
        keys = param_grid.keys()
        combinations = itertools.product(*param_grid.values())

        print(f"Tuning KMeans for K={k}...")

        for combo in combinations:
            params = dict(zip(keys, combo))

            # Train with specific params
            kmeans = KMeans(n_clusters=k, random_state=42, **params)
            labels = kmeans.fit_predict(self.X)
            score = silhouette_score(self.X, labels)

            # Store result
            result_entry = params.copy()
            result_entry['Silhouette Score'] = score
            results.append(result_entry)

        return pd.DataFrame(results).sort_values(by='Silhouette Score', ascending=False)