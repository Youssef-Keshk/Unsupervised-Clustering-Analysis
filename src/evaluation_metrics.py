import numpy as np
from gmm import Covariance

# INTERNAL VALIDATION METRICS

def silhouette_score(X, labels):
    """
    Compute silhouette score from scratch.
    X: (n, d)
    labels: (n,)
    """
    n = len(X)
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    # Precompute distances
    dist = np.linalg.norm(X[:, None] - X[None, :], axis=2)

    a = np.zeros(n)
    b = np.ones(n) * np.inf

    for cluster in unique_labels:
        idx_in = np.where(labels == cluster)[0]
        idx_out = np.where(labels != cluster)[0]

        # intra-cluster distance
        if len(idx_in) > 1:
            a[idx_in] = np.mean(dist[np.ix_(idx_in, idx_in)], axis=1)

        # nearest other cluster
        for other in unique_labels:
            if other == cluster:
                continue
            idx_other = np.where(labels == other)[0]
            mean_dist_other = np.mean(dist[np.ix_(idx_in, idx_other)], axis=1)
            b[idx_in] = np.minimum(b[idx_in], mean_dist_other)

    s = (b - a) / np.maximum(a, b)
    return np.mean(s)


def davies_bouldin_index(X, labels):
    """
    Davies-Bouldin Index implementation.
    Lower is better.
    """
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    centroids = np.array([X[labels == c].mean(axis=0) for c in unique_labels])
    S = np.array([
        np.mean(np.linalg.norm(X[labels == c] - centroids[i], axis=1))
        for i, c in enumerate(unique_labels)
    ])

    # pairwise centroid distances
    M = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2)

    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                R[i, j] = (S[i] + S[j]) / (M[i, j] + 1e-10)

    D = np.max(R, axis=1)
    return np.mean(D)


def calinski_harabasz_index(X, labels):
    """
    CH Index implementation.
    Higher is better.
    """
    n, d = X.shape
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    overall_mean = np.mean(X, axis=0)

    # Between-cluster dispersion
    B = 0
    # Within-cluster dispersion
    W = 0

    for c in unique_labels:
        Xc = X[labels == c]
        nc = len(Xc)
        mean_c = np.mean(Xc, axis=0)

        B += nc * np.sum((mean_c - overall_mean)**2)
        W += np.sum((Xc - mean_c)**2)

    return (B * (n - k)) / (W * (k - 1) + 1e-10)


def wcss(X, labels):
    """
    Within-Cluster Sum of Squares
    """
    unique_labels = np.unique(labels)

    total = 0
    for c in unique_labels:
        Xc = X[labels == c]
        centroid = Xc.mean(axis=0)
        total += np.sum((Xc - centroid) ** 2)
    return total


# ------------------------------------------------------------------------------------------------------------------------


# EXTERNAL VALIDATION METRICS

def adjusted_rand_index(y_true, y_pred):
    """
    ARI implementation from scratch.
    """
    from math import comb

    n = len(y_true)
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)

    # contingency table
    M = np.zeros((len(classes), len(clusters)))

    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            M[i, j] = np.sum((y_true == c) & (y_pred == k))

    sum_comb_c = sum(comb(int(n_ij), 2) for row in M for n_ij in row)
    sum_comb_rows = sum(comb(int(n_i), 2) for n_i in M.sum(axis=1))
    sum_comb_cols = sum(comb(int(n_j), 2) for n_j in M.sum(axis=0))

    expected_index = sum_comb_rows * sum_comb_cols / comb(n, 2)
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)

    return (sum_comb_c - expected_index) / (max_index - expected_index + 1e-10)


def normalized_mutual_information(y_true, y_pred):
    """
    NMI implementation.
    """
    eps = 1e-10
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)

    n = len(y_true)

    # contingency table
    M = np.zeros((len(classes), len(clusters)))
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            M[i, j] = np.sum((y_true == c) & (y_pred == k))

    # normalize
    M = M / n

    Pc = M.sum(axis=1)
    Pk = M.sum(axis=0)

    # mutual info
    MI = 0
    for i in range(len(classes)):
        for j in range(len(clusters)):
            if M[i, j] > 0:
                MI += M[i, j] * np.log(M[i, j] / (Pc[i] * Pk[j] + eps) + eps)

    # entropies
    Hc = -np.sum(Pc * np.log(Pc + eps))
    Hk = -np.sum(Pk * np.log(Pk + eps))

    return MI / np.sqrt(Hc * Hk + eps)


def purity_score(y_true, y_pred):
    """
    Purity implementation.
    """
    clusters = np.unique(y_pred)
    total = 0

    for c in clusters:
        labels_in_cluster = y_true[y_pred == c]
        most_common = np.bincount(labels_in_cluster).max()
        total += most_common
    return total / len(y_true)


def confusion_matrix(y_true, y_pred):
    """
    Simple confusion matrix.
    """
    classes = np.unique(y_true)
    k = len(classes)
    matrix = np.zeros((k, k), dtype=int)

    for i, c in enumerate(classes):
        for j, d in enumerate(classes):
            matrix[i, j] = np.sum((y_true == c) & (y_pred == d))
    return matrix


# ------------------------------------------------------------------------------------------------------------------------

# DIMENSIONALITY REDUCTION METRICS

def reconstruction_error(X, X_reconstructed):
    """
    MSE reconstruction error.
    """
    return np.mean((X - X_reconstructed) ** 2)


def explained_variance_ratio(eigenvalues):
    """
    PCA explained variance ratio.
    """
    total = np.sum(eigenvalues)
    return eigenvalues / (total + 1e-10)


# ------------------------------------------------------------------------------------------------------------------------

# GMM PROBABILISTIC METRICS

def gmm_log_likelihood(X, gmm):
    """
    Computes LL manually.
    """
    from scipy.stats import multivariate_normal

    n = len(X)
    k = gmm.k

    total = 0
    for i in range(n):
        s = 0
        for j in range(k):
            dist = multivariate_normal(mean=gmm.mu[j], cov=gmm.sigma[j], allow_singular=True)
            s += gmm.phi[j] * dist.pdf(X[i])
        total += np.log(s + 1e-10)
    return total


def gmm_num_params(d, k, covariance_type=Covariance.FULL):
    """
    Compute number of learnable parameters for AIC/BIC.
    """
    if covariance_type == Covariance.FULL:
        cov_params = k * (d * (d + 1) / 2)
    elif covariance_type == Covariance.DIAGONAL:
        cov_params = k * d
    elif covariance_type == Covariance.SPHERICAL:
        cov_params = k
    elif covariance_type == Covariance.TIED:
        cov_params = (d * (d + 1) / 2)
    else:
        raise ValueError("Invalid covariance type")

    mean_params = k * d
    weight_params = k - 1

    return int(cov_params + mean_params + weight_params)


def aic(log_likelihood, num_params):
    return 2 * num_params - 2 * log_likelihood


def bic(log_likelihood, num_params, n_samples):
    return np.log(n_samples) * num_params - 2 * log_likelihood