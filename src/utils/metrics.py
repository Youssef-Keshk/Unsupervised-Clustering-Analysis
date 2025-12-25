import numpy as np
from src.models.kmeans import KMeans, KMeansInit

def purity_score(y_true, y_pred):
    """Calculate the purity of clustering"""
    contingency_matrix = np.zeros((len(np.unique(y_true)), len(np.unique(y_pred))))
    for i, j in zip(y_true, y_pred):
        contingency_matrix[int(i), int(j)] += 1
    return np.sum(np.max(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def compute_silhouette_score(X, labels):
    """Calculate the Silhouette Score"""
    n_samples = X.shape[0]
    s_scores = np.zeros(n_samples)
    unique_labels = np.unique(labels)
    
    for i in range(n_samples):
        # a(i): Mean distance to points in same cluster
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) > 1:
            a_i = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))
        else:
            a_i = 0
            
        # b(i): Mean distance to points in nearest other cluster
        other_clusters = [l for l in unique_labels if l != labels[i]]
        b_i = np.inf
        for l in other_clusters:
            other_points = X[labels == l]
            mean_dist = np.mean(np.linalg.norm(other_points - X[i], axis=1))
            b_i = min(b_i, mean_dist)
            
        s_scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
    return np.mean(s_scores)


def compute_gap_statistic(X, k_range, n_references=5, random_state=42):
    """
    Computes the Gap Statistic for a range of k values.
    """
    if random_state:
        np.random.seed(random_state)
        
    gaps = np.zeros(len(k_range))
    results_std = np.zeros(len(k_range))
    
    for i, k in enumerate(k_range):
        # 1. Fit KMeans to actual data and get log(WCSS)
        model = KMeans(K=k, init=KMeansInit.KMEANS_PLUS_PLUS, random_state=random_state)
        model.fit(X)
        actual_wcss = model.inertia_history[-1]
        log_actual_wcss = np.log(actual_wcss + 1e-10)
        
        # 2. Generate reference datasets and get their log(WCSS)
        ref_log_wcss = []
        # Define bounds for uniform distribution based on the data
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)
        
        for _ in range(n_references):
            # Create random data with same shape and range as X
            random_data = np.random.uniform(min_vals, max_vals, X.shape)
            
            ref_model = KMeans(K=k, init=KMeansInit.KMEANS_PLUS_PLUS, random_state=random_state)
            ref_model.fit(random_data)
            ref_wcss = ref_model.inertia_history[-1]
            ref_log_wcss.append(np.log(ref_wcss + 1e-10))
        
        # 3. Calculate Gap(k) = E[log(W_ref)] - log(W_actual)
        gaps[i] = np.mean(ref_log_wcss) - log_actual_wcss
        results_std[i] = np.std(ref_log_wcss) * np.sqrt(1 + 1/n_references)
        
    return gaps, results_std


def compute_davies_bouldin(X, labels):
    """Davies-Bouldin Index implementation from scratch"""
    n_clusters = len(np.unique(labels))
    cluster_k = [X[labels == k] for k in range(n_clusters)]
    centroids = [np.mean(k, axis=0) for k in cluster_k]
    
    # Average distance of all points in cluster to its centroid
    S = [np.mean(np.linalg.norm(cluster_k[i] - centroids[i], axis=1)) for i in range(n_clusters)]
    
    R = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                dist_centroids = np.linalg.norm(centroids[i] - centroids[j])
                R[i, j] = (S[i] + S[j]) / (dist_centroids + 1e-10)
    
    return np.mean(np.max(R, axis=1))


def compute_calinski_harabasz(X, labels):
    """Calinski-Harabasz Index implementation from scratch"""
    n_samples = X.shape[0]
    n_clusters = len(np.unique(labels))
    extra_disp = 0  # Between-cluster dispersion
    intra_disp = 0  # Within-cluster dispersion
    
    mean_overall = np.mean(X, axis=0)
    for k in range(n_clusters):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * np.sum((mean_k - mean_overall)**2)
        intra_disp += np.sum((cluster_k - mean_k)**2)
        
    return (extra_disp / (n_clusters - 1)) / (intra_disp / (n_samples - n_clusters))


def compute_ari(y_true, y_pred):
    """
    Computes the Adjusted Rand Index.
    ARI = (Index - ExpectedIndex) / (MaxIndex - ExpectedIndex)
    """
    n = len(y_true)
    if n <= 1:
        return 0.0

    # 1. Create the contingency table (Confusion Matrix)
    classes, class_idx = np.unique(y_true, return_inverse=True)
    clusters, cluster_idx = np.unique(y_pred, return_inverse=True)
    n_classes = len(classes)
    n_clusters = len(clusters)
    
    contingency = np.zeros((n_classes, n_clusters))
    for i in range(n):
        contingency[class_idx[i], cluster_idx[i]] += 1

    # 2. Compute sums for combinations n_ijC2, aiC2, bjC2
    # Combination formula: n! / (2!(n-2)!) = n(n-1)/2
    sum_comb_nij = np.sum(contingency * (contingency - 1) / 2)
    
    sum_comb_ai = np.sum(np.sum(contingency, axis=1) * (np.sum(contingency, axis=1) - 1) / 2)
    sum_comb_bj = np.sum(np.sum(contingency, axis=0) * (np.sum(contingency, axis=0) - 1) / 2)
    
    # 3. Compute Expected Index and Max Index
    sum_comb_n = n * (n - 1) / 2
    expected_index = (sum_comb_ai * sum_comb_bj) / sum_comb_n
    max_index = 0.5 * (sum_comb_ai + sum_comb_bj)
    
    # 4. Final ARI calculation
    denominator = max_index - expected_index
    if denominator == 0:
        return 1.0 if sum_comb_nij == expected_index else 0.0
        
    return (sum_comb_nij - expected_index) / denominator



def compute_nmi(y_true, y_pred):
    """Normalized Mutual Information"""
    n = len(y_true)
    
    # Entropy of True Labels H(Y)
    _, counts_y = np.unique(y_true, return_counts=True)
    p_y = counts_y / n
    h_y = -np.sum(p_y * np.log(p_y + 1e-10))
    
    # Entropy of Predicted Clusters H(C)
    _, counts_c = np.unique(y_pred, return_counts=True)
    p_c = counts_c / n
    h_c = -np.sum(p_c * np.log(p_c + 1e-10))
    
    # Mutual Information I(Y;C)
    mi = 0
    unique_y = np.unique(y_true)
    unique_c = np.unique(y_pred)
    
    for y_val in unique_y:
        for c_val in unique_c:
            # Joint probability
            p_yc = np.sum((y_true == y_val) & (y_pred == c_val)) / n
            # Marginal probabilities
            py = np.sum(y_true == y_val) / n
            pc = np.sum(y_pred == c_val) / n
            
            if p_yc > 0:
                mi += p_yc * np.log(p_yc / (py * pc))
                
    # Normalization: 2 * MI / (H(Y) + H(C))
    return 2 * mi / (h_y + h_c + 1e-10)