from enum import Enum
import numpy as np

class KMeansInit(Enum):
    RANDOM = "random"
    KMEANS_PLUS_PLUS = "kmeans++"

class KMeans:   
    def __init__(self, K: int=8, max_iters: int=300, tol: float=1e-4, init: KMeansInit=KMeansInit.RANDOM, random_state: int=None):
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.init_type = init
        self.centroids = None
        self.clusters = None
        self.inertia_history = []
        self.random_state = random_state

    @property
    def inertia_(self):
        if len(self.inertia_history) > 0:
            return self.inertia_history[-1]
        return None


    # K-means++ initialization
    def _initialize_kmeans_plus_plus(self):
        m, n = self.X.shape
        # Select first centroid randomly
        centroids = [self.X[np.random.randint(m)]]
        
        for _ in range(1, self.K):
            # Compute squared distances to nearest existing centroid
            dist_sq = np.array([min(np.sum((x - c)**2) for c in centroids) for x in self.X])
            probs = dist_sq / (np.sum(dist_sq) + 1e-10) # Stability handling 
            next_idx = np.random.choice(m, p=probs)
            centroids.append(self.X[next_idx])
        
        self.centroids = np.array(centroids)
    

    # random initialization
    def _initialize_random_centroids(self):
        m, n = self.X.shape
        # pick k random data points from X as the centroid
        indices = np.random.choice(m, self.K, replace=False) 
        # a centroid should be of shape (1, n), so the centroids array will be of shape (K, n)
        self.centroids = self.X[indices]

    
    def _closest_centroid(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)


    def _compute_means(self, cluster_idx):
        _, n = np.shape(self.X)
        for k in range(self.K):
            points = self.X[cluster_idx == k] # gather points for the cluster i
            if len(points) == 0:
                self.centroids[k] = self.X[np.random.randint(len(self.X))]
            else:
                self.centroids[k] = np.mean(points, axis=0) # use axis=0 to compute means across points


    def fit(self, data):
        self.X = data
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # initialize random centroids
        if self.init_type == KMeansInit.KMEANS_PLUS_PLUS:
            self._initialize_kmeans_plus_plus()
        else:
            self._initialize_random_centroids()

        # loop till max_iterations or convergance
        for _ in range(self.max_iters):

            # create clusters by assigning the samples to the closet centroids
            self.clusters = self._closest_centroid(self.X)
            old_centroids = self.centroids.copy()

            # compute means of the clusters and assign to centroids
            self._compute_means()

            # Record inertia (WCSS)
            inertia = np.sum((self.X - self.centroids[self.clusters])**2)
            self.inertia_history.append(inertia)

            # Convergence check: if the new_centroids are the same as the old centroids, return
            if np.linalg.norm(self.centroids - old_centroids) < self.tol:
                break
        return self


    def predict(self, X):
        return self._closest_centroid(X)
    
    


    
