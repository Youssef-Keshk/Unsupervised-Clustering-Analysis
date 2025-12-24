import numpy as np
from typing import Optional

class PCA:
    """
    Principal Component Analysis implementation from scratch.
    Uses eigenvalue decomposition approach.
    """
    
    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize PCA.
        
        Parameters:
        -----------
        n_components : int, optional
            Number of components to keep. If None, keep all components.
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.eigenvalues_ = None
        
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit PCA on data X.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : PCA
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
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Transformed data
        """
        X_centered = X - self.mean_
        return X_centered @ self.components_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Transformed data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space (reconstruction).
        
        Parameters:
        -----------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Transformed data
            
        Returns:
        --------
        X_reconstructed : np.ndarray of shape (n_samples, n_features)
            Reconstructed data
        """
        return (X_transformed @ self.components_.T) + self.mean_
    
    def reconstruction_error(self, X: np.ndarray) -> float:
        """
        Compute reconstruction error (MSE).
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Original data
            
        Returns:
        --------
        error : float
            Mean squared reconstruction error
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        return np.mean((X - X_reconstructed) ** 2)
    
    def get_cumulative_variance_ratio(self) -> np.ndarray:
        """
        Get cumulative explained variance ratio.
        
        Returns:
        --------
        cumulative_variance : np.ndarray
            Cumulative explained variance ratio
        """
        return np.cumsum(self.explained_variance_ratio_)


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 30)
    
    # Test PCA
    print("=" * 50)
    print("Testing PCA")
    print("=" * 50)
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)
    
    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_pca.shape}")
    print(f"\nExplained variance ratio (first 5 components):")
    print(pca.explained_variance_ratio_[:5])
    print(f"\nCumulative variance (first 5 components):")
    print(pca.get_cumulative_variance_ratio()[:5])
    print(f"\nReconstruction error: {pca.reconstruction_error(X):.6f}")
    
    # Test with different number of components
    print("\n" + "=" * 50)
    print("Testing with different number of components")
    print("=" * 50)
    for n_comp in [2, 5, 10, 15, 20]:
        pca_temp = PCA(n_components=n_comp)
        pca_temp.fit(X)
        error = pca_temp.reconstruction_error(X)
        cum_var = pca_temp.get_cumulative_variance_ratio()[-1]
        print(f"Components: {n_comp:2d} | Reconstruction Error: {error:.6f} | Cumulative Variance: {cum_var:.4f}")