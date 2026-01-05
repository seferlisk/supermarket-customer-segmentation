from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class ClusterModeler:
    def __init__(self, X):
        self.X = X
        self.inertia_ = []
        self.silhouette_scores_ = []
        self.k_range = range(2, 11)
        self.models = {}

    def find_optimal_k(self):
        """Calculates inertia and silhouette scores for a range of k."""
        for k in self.k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            kmeans.fit(self.X)

            self.inertia_.append(kmeans.inertia_)
            self.silhouette_scores_.append(silhouette_score(self.X, kmeans.labels_))
            self.models[k] = kmeans

        return self.inertia_, self.silhouette_scores_

    def get_model(self, k):
        return self.models.get(k)