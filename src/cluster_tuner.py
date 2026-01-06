import pandas as pd
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ClusterTuner:
    def __init__(self, X, k):
        self.X = X
        self.k = k

    def tune_parameters(self, param_grid):
        """
        Iterates through hyperparameter combinations and calculates Silhouette Score.
        Returns a DataFrame of results sorted by performance.
        """
        results = []
        keys = param_grid.keys()
        combinations = itertools.product(*param_grid.values())

        print(f"Tuning KMeans for K={self.k}...")

        for combo in combinations:
            params = dict(zip(keys, combo))

            # Instantiate KMeans with specific params
            # Note: n_init='auto' is default in newer sklearn, but we tune it explicitly
            kmeans = KMeans(n_clusters=self.k, random_state=42, **params)
            labels = kmeans.fit_predict(self.X)

            score = silhouette_score(self.X, labels)

            # Store result
            result_entry = params.copy()
            result_entry['Silhouette Score'] = score
            results.append(result_entry)

        results_df = pd.DataFrame(results)
        return results_df.sort_values(by='Silhouette Score', ascending=False)