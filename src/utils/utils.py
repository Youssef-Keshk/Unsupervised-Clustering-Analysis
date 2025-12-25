import numpy as np
import pandas as pd


def load_scale_data(data_path):
    # 1. Load data
    df = pd.read_csv(data_path)

    # 2. Basic cleaning (Handle missing values if any)
    if 'Unnamed: 32' in df.columns:
        df = df.drop(columns=['Unnamed: 32'])
    df = df.dropna()
        
    # 3. Separate features and labels
    X = df.drop(columns=['id', 'diagnosis']).values
    y_true = df['diagnosis'].map({'M': 1, 'B': 0}).values

    # 4. Standardize data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / (X_std + 1e-8)

    return X_scaled, y_true

def compute_confusion_matrix(y_true, y_pred, n_classes=2):
    """
    Computes a confusion matrix from scratch.
    Rows: Actual Labels, Columns: Predicted Clusters
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        cm[int(y_true[i]), int(y_pred[i])] += 1
    return cm

def align_clusters_with_labels(y_true, y_pred):
    """
    Since K-means doesn't know 'M' or 'B', we map clusters to 
    the most frequent label within that cluster.
    """
    mapped_predictions = np.zeros_like(y_pred)
    for cluster in np.unique(y_pred):
        # Find which true label is most common in this cluster
        labels_in_cluster = y_true[y_pred == cluster]
        if len(labels_in_cluster) > 0:
            most_frequent = np.bincount(labels_in_cluster).argmax()
            mapped_predictions[y_pred == cluster] = most_frequent
    return mapped_predictions