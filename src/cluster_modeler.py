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
        inertia = []
        sil_scores = []
        for k in self.k_range:
            km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            km.fit(self.X)
            inertia.append(km.inertia_)
            sil_scores.append(silhouette_score(self.X, km.labels_))
        return inertia, sil_scores

    def train_model(self, k):
        self.model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        labels = self.model.fit_predict(self.X)
        return labels