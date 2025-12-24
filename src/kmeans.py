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
        self.init_centroids: callable = self._initialize_kmeans_plus_plus if init == KMeansInit.KMEANS_PLUS_PLUS else self._initialize_random_centroids
        self.centroids = None
        self.X = None
        self.clusters = None
        self.inertia_history = []
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)


    # K-means++ initialization
    def _initialize_kmeans_plus_plus(self):
        centroids = [self.X[np.random.randint(self.X.shape[0])]]

        for _ in range(1, self.K):
            distances = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in self.X])
            probabilities = distances / distances.sum()
            next_centroid = self.X[np.random.choice(len(self.X), p=probabilities)]
            centroids.append(next_centroid)

        self.centroids = np.array(centroids) 
    

    # random initialization
    def _initialize_random_centroids(self):
        m, n = np.shape(self.X)
        # a centroid should be of shape (1, n), so the centroids array will be of shape (K, n)
        self.centroids = np.empty((self.K, n))
        for i in range(self.K):
            # pick a random data point from X as the centroid
            self.centroids[i] =  self.X[np.random.choice(range(m))] 

    
    def _closest_centroid(self, x):
        distances = np.linalg.norm(self.centroids - x, axis=1)
        return np.argmin(distances) # return the index of the lowest distance
    

    def _create_clusters(self):
        m, _ = np.shape(self.X)
        cluster_idx = np.empty(m, dtype=int)
        for i in range(m):
            cluster_idx[i] = self._closest_centroid(self.X[i])
        return cluster_idx


    def _compute_means(self, cluster_idx):
        _, n = np.shape(self.X)
        for i in range(self.K):
            points = self.X[cluster_idx == i] # gather points for the cluster i
            if len(points) == 0:
                self.centroids[i] = self.X[np.random.randint(len(self.X))]
            else:
                self.centroids[i] = np.mean(points, axis=0) # use axis=0 to compute means across points

    def fit(self, data):
        self.X = data
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # initialize random centroids
        self.init_centroids()

        # loop till max_iterations or convergance
        for _ in range(self.max_iters):

            # create clusters by assigning the samples to the closet centroids
            self.clusters = self._create_clusters()
            previous_centroids = self.centroids                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
            # compute means of the clusters and assign to centroids
            self._compute_means(self.clusters)

            # track inertia history
            inertia = np.sum(np.linalg.norm(self.X - self.centroids[self.clusters], axis=1)**2)
            self.inertia_history.append(inertia)

            # if the new_centroids are the same as the old centroids, return
            shift = np.linalg.norm(previous_centroids - self.centroids)
            if shift < self.tol:
                break

    def predict(self, X_new):
        clusters = np.array([self._closest_centroid(x) for x in X_new])
        return clusters
    
    
    # def plot_2d(self, y_true=None):
    #     import matplotlib.pyplot as plt

    #     # Predicted clustering
    #     plt.figure()
    #     plt.scatter(self.X[:, 0], self.X[:, 1], c=self.clusters)
    #     plt.title("K-Means Clustering")
    #     plt.xlabel("Feature 1")
    #     plt.ylabel("Feature 2")
    #     plt.show()

    #     # Ground truth 
    #     if y_true is not None:
    #         plt.figure()
    #         plt.scatter(self.X[:, 0], self.X[:, 1], c=y_true)
    #         plt.title("Actual Clustering")
    #         plt.xlabel("Feature 1")
    #         plt.ylabel("Feature 2")
    #         plt.show()

    
