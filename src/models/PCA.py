import numpy as np
from typing import Optional

class PCA:
    """
    Principal Component Analysis Using eigenvalue decomposition approach
    """
    
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.eigenvalues_ = None
        
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit PCA on data X
        """
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        n_samples = X.shape[0]
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store eigenvalues
        self.eigenvalues_ = eigenvalues
        
        # Select number of components
        if self.n_components is None:
            self.n_components = X.shape[1]
        
        # Store principal components
        self.components_ = eigenvectors[:, :self.n_components]
        
        # Calculate explained variance
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal component space.
        """
        X_centered = X - self.mean_
        return X_centered @ self.components_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data.
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space (reconstruction).
        """
        return (X_transformed @ self.components_.T) + self.mean_
    
    def reconstruction_error(self, X: np.ndarray) -> float:
        """
        Compute reconstruction error (MSE).
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        return np.mean((X - X_reconstructed) ** 2)
    
    def get_cumulative_variance_ratio(self) -> np.ndarray:
        """
        Get cumulative explained variance ratio.
        """
        return np.cumsum(self.explained_variance_ratio_)
