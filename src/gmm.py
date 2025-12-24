from enum import Enum
import numpy as np
from scipy.stats import multivariate_normal 

class Covariance(Enum):
    FULL = 'full'
    TIED = 'tied'
    DIAGONAL = 'diagonal'
    SPHERICAL = 'spherical'

class GMM:
    def __init__(self, k, max_iter: int=5, tol: float=1e-4, covariance_type: Covariance=Covariance.FULL):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.log_likelihood_ = []
        self.covariance_type = covariance_type


    # theta =(mu1,sigma1,mu2,simga2......muk,sigmak)
    def _initialize(self, X):
        self.n, self.m = X.shape

        # initial weights given to each cluster are stored in phi or P(Zi=j)
        self.phi = np.full(self.k, 1/self.k)  # cluster weights
        
        # initial weights given to each data point wrt to each cluster or P(Xi|Zi=j) (responsibilities)
        self.weights = np.zeros((self.n, self.k))  

        # initial value of mean of k Gaussians
        indices = np.random.choice(self.n, self.k, replace=False)
        self.mu = X[indices]

        # initial value of covariance matrix of k Gaussians
        self.sigma = []
        base_cov = np.cov(X.T) + 1e-6*np.eye(self.m)
        for _ in range(self.k):
            self.sigma.append(base_cov.copy())
        if self.covariance_type == Covariance.TIED:
            self.tied_sigma = base_cov.copy()


    # E-Step: update weights and phi
    def _e_step(self, X):
        likelihood = np.zeros((self.n, self.k))

        for i in range(self.k):
            cov = self._get_covariance(i)
            dist = multivariate_normal(mean=self.mu[i], cov=cov, allow_singular=True)
            likelihood[:, i] = dist.pdf(X)

        # updated weights or P(Xi|Zi=j)
        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis] + 1e-10
        self.weights = numerator / denominator

        # mean of sum of probability of all data points wrt to one cluster is new updated probability of cluster k or (phi)k
        self.phi = self.weights.mean(axis=0)

        # Compute log-likelihood
        self.log_likelihood = np.sum(np.log(numerator.sum(axis=1) + 1e-10))
        self.log_likelihood_.append(self.log_likelihood)

    
    # M-Step: update meu and sigma
    def _m_step(self, X):
        for i in range(self.k):
            resp = self.weights[:, i]
            total_resp = resp.sum() + 1e-10  # avoid division by zero
            self.mu[i] = (resp[:, np.newaxis] * X).sum(axis=0) / total_resp
            diff = X - self.mu[i]

            if self.covariance_type == Covariance.FULL:
                cov = (resp[:, np.newaxis] * diff).T @ diff / total_resp
                cov += 1e-6*np.eye(self.m)
                self.sigma[i] = cov

            elif self.covariance_type == Covariance.DIAGONAL:
                var = ((resp[:, np.newaxis] * diff ** 2).sum(axis=0) / total_resp)
                self.sigma[i] = np.diag(var + 1e-6)

            elif self.covariance_type == Covariance.SPHERICAL:
                var = ((resp[:, np.newaxis] * diff ** 2).sum() / (total_resp * self.m))
                self.sigma[i] = np.eye(self.m) * (var + 1e-6)

        if self.covariance_type == Covariance.TIED:
            cov = np.zeros((self.m, self.m))

            for i in range(self.k):
                diff = X - self.mu[i]
                cov += (self.weights[:, i][:, np.newaxis] * diff).T @ diff
                
            cov /= self.n
            cov += 1e-6 * np.eye(self.m)
            self.tied_sigma = cov

            for i in range(self.k):
                self.sigma[i] = self.tied_sigma


    def _get_covariance(self, i):
        if self.covariance_type == Covariance.TIED:
            return self.tied_sigma
        return self.sigma[i]
    

    # clustering the data points correctly
    def fit(self, X):
        self._initialize(X)
        prev_log_likelihood = None
        
        for _ in range(self.max_iter):
            self._e_step(X) # iterate to update the value of P(Xi|Zi=j) and (phi)k
            self._m_step(X) # iterate to update the value of meu and sigma as the clusters shift
            
            if prev_log_likelihood is not None and abs(self.log_likelihood - prev_log_likelihood) < self.tol:
                break
            
            prev_log_likelihood = self.log_likelihood
    
    
    # predicts probability of each data point wrt each cluster
    def _predict_proba(self, X):
        # Creates a n*k matrix denoting probability of each point wrt each cluster 
        likelihood = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            cov = self._get_covariance(i)
            dist = multivariate_normal(mean=self.mu[i], cov=cov, allow_singular=True)
            # pdf
            likelihood[:, i] = dist.pdf(X)
        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis] + 1e-10
        return numerator / denominator


    def predict(self, X):
        weights = self._predict_proba(X)
        return np.argmax(weights, axis=1)

    
    def _num_parameters(self):
        if self.covariance_type == Covariance.FULL:
            return int(self.k * (self.m + self.m * (self.m + 1) / 2) + (self.k - 1))
        
        elif self.covariance_type == Covariance.DIAGONAL:
            return int(self.k * (2 * self.m) + (self.k - 1))
        
        elif self.covariance_type == Covariance.SPHERICAL:
            return int(self.k * (self.m + 1) + (self.k - 1))
        
        elif self.covariance_type == Covariance.TIED:
            return int(self.k * self.m + (self.m * (self.m + 1) / 2) + (self.k - 1))
    
    def bic(self, X):
        n = X.shape[0]
        p = self._num_parameters()
        return -2 * self.log_likelihood + p * np.log(n)


    def aic(self, X):
        p = self._num_parameters()
        return -2 * self.log_likelihood + 2 * p
