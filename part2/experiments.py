import numpy as np
from src.gmm import GMM
from src.PCA import PCA
from src.autoencoder import Autoencoder
from src.kmeans import KMeans, KMeansInit
from src.evaluation_metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_index,
    normalized_mutual_info,
    purity_score
)


def experiment1_kmeans_original(
    X: np.ndarray,
    y: np.ndarray,
    k_values: list,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: int = 42
):

    results = []

    for k in k_values:
        # 1. Initialize K-Means
        kmeans = KMeans(
            K=k,
            max_iters=max_iter,
            tol=tol,
            random_state=random_state
        )


        # 2. Fit model and get cluster assignments
        kmeans.fit(X)
        labels = kmeans.clusters

        # 3. Internal evaluation metrics
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        wcss = kmeans.inertia_history[-1]

        # 4. External evaluation metrics
        ari = adjusted_rand_index(y, labels)
        nmi = normalized_mutual_info(y, labels)
        purity = purity_score(y, labels)

        # 5. Store results
        results.append({
            "experiment": "Original + KMeans",
            "k": k,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "wcss": wcss,
            "ARI": ari,
            "NMI": nmi,
            "purity": purity
        })

    return results


def experiment2_gmm_original(
    X: np.ndarray,
    y: np.ndarray,
    component_values: list,
    covariance_types: list,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42
):
    results = []

    for k in component_values:
        for cov_type in covariance_types:
            gmm = GMM(
                k=k,
                max_iter=max_iter,
                tol=tol,
                covariance_type=cov_type,
                random_state=random_state
            )
            gmm.fit(X)
            labels = gmm.predict(X)

            # Internal metrics
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)

            # External metrics
            ari = adjusted_rand_index(y, labels)
            nmi = normalized_mutual_info(y, labels)
            purity = purity_score(y, labels)

            # Model metrics
            log_likelihood = gmm.log_likelihood_
            bic = gmm.bic(X)
            aic = gmm.aic(X)

            results.append({
                "experiment": "Original + GMM",
                "k": k,
                "covariance_type": cov_type,
                "silhouette": sil,
                "davies_bouldin": db,
                "calinski_harabasz": ch,
                "ARI": ari,
                "NMI": nmi,
                "purity": purity,
                "log_likelihood": log_likelihood,
                "BIC": bic,
                "AIC": aic
            })

    return results


def experiment3_pca_kmeans(
    X: np.ndarray,
    y: np.ndarray,
    pca_dims: list,
    k_values: list,
    max_iters: int = 300,
    tol: float = 1e-4,
    random_state: int = 42
):
    results = []

    for d in pca_dims:
        pca = PCA(n_components=d)
        X_pca = pca.fit_transform(X)

        reconstruction_error = pca.reconstruction_error(X)
        explained_variance = pca.get_cumulative_variance_ratio()[-1]

        for k in k_values:
            kmeans = KMeans(
                K=k,
                max_iters=max_iters,
                tol=tol,
                init=KMeansInit.KMEANS_PLUS_PLUS,
                random_state=random_state
            )
            kmeans.fit(X_pca)
            labels = kmeans.clusters

            sil = silhouette_score(X_pca, labels)
            db = davies_bouldin_score(X_pca, labels)
            ch = calinski_harabasz_score(X_pca, labels)

            ari = adjusted_rand_index(y, labels)
            nmi = normalized_mutual_info(y, labels)
            purity = purity_score(y, labels)

            inertia = kmeans.inertia_history[-1]

            results.append({
                "experiment": "PCA + KMeans",
                "pca_dim": d,
                "k": k,
                "explained_variance": explained_variance,
                "reconstruction_error": reconstruction_error,
                "silhouette": sil,
                "davies_bouldin": db,
                "calinski_harabasz": ch,
                "ARI": ari,
                "NMI": nmi,
                "purity": purity,
                "inertia": inertia
            })

    return results


def experiment4_pca_gmm(
    X: np.ndarray,
    y: np.ndarray,
    pca_dims: list,
    component_values: list,
    covariance_types: list,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42
):
    results = []

    for d in pca_dims:
        pca = PCA(n_components=d)
        X_pca = pca.fit_transform(X)

        reconstruction_error = pca.reconstruction_error(X)
        explained_variance = pca.get_cumulative_variance_ratio()[-1]

        for k in component_values:
            for cov_type in covariance_types:
                gmm = GMM(
                    k=k,
                    max_iter=max_iter,
                    tol=tol,
                    covariance_type=cov_type,
                    random_state=random_state
                )
                gmm.fit(X_pca)
                labels = gmm.predict(X_pca)

                sil = silhouette_score(X_pca, labels)
                db = davies_bouldin_score(X_pca, labels)
                ch = calinski_harabasz_score(X_pca, labels)

                ari = adjusted_rand_index(y, labels)
                nmi = normalized_mutual_info(y, labels)
                purity = purity_score(y, labels)

                log_likelihood = gmm.log_likelihood_
                bic = gmm.bic(X_pca)
                aic = gmm.aic(X_pca)

                results.append({
                    "experiment": "PCA + GMM",
                    "pca_dim": d,
                    "k": k,
                    "covariance_type": cov_type,
                    "explained_variance": explained_variance,
                    "reconstruction_error": reconstruction_error,
                    "silhouette": sil,
                    "davies_bouldin": db,
                    "calinski_harabasz": ch,
                    "ARI": ari,
                    "NMI": nmi,
                    "purity": purity,
                    "log_likelihood": log_likelihood,
                    "BIC": bic,
                    "AIC": aic
                })

    return results


def experiment5_autoencoder_kmeans(
    X: np.ndarray,
    y: np.ndarray,
    bottleneck_dims: list,
    k_values: list,
    encoding_dims: list,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    l2_lambda: float = 0.001,
    random_state: int = 42
):
    results = []

    for bottleneck in bottleneck_dims:
        ae = Autoencoder(
            input_dim=X.shape[1],
            encoding_dims=encoding_dims,
            bottleneck_dim=bottleneck,
            activation='relu',
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            l2_lambda=l2_lambda,
            random_state=random_state
        )

        ae.fit(X, verbose=False)

        X_encoded = ae.encode(X)
        reconstruction_error = ae.reconstruction_error(X)
        final_loss = ae.train_loss_history[-1]

        for k in k_values:
            kmeans = KMeans(
                K=k,
                init=KMeansInit.KMEANS_PLUS_PLUS,
                random_state=random_state
            )

            kmeans.fit(X_encoded)
            labels = kmeans.clusters

            sil = silhouette_score(X_encoded, labels)
            db = davies_bouldin_score(X_encoded, labels)
            ch = calinski_harabasz_score(X_encoded, labels)

            ari = adjusted_rand_index(y, labels)
            nmi = normalized_mutual_info(y, labels)
            purity = purity_score(y, labels)

            results.append({
                "experiment": "Autoencoder + K-Means",
                "bottleneck_dim": bottleneck,
                "k": k,
                "reconstruction_error": reconstruction_error,
                "final_train_loss": final_loss,
                "inertia": kmeans.inertia_history[-1],
                "silhouette": sil,
                "davies_bouldin": db,
                "calinski_harabasz": ch,
                "ARI": ari,
                "NMI": nmi,
                "purity": purity
            })

    return results


def experiment6_autoencoder_gmm(
    X: np.ndarray,
    y: np.ndarray,
    bottleneck_dims: list,
    component_values: list,
    covariance_types: list,
    encoding_dims: list,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    l2_lambda: float = 0.001,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42
):
    results = []

    for bottleneck in bottleneck_dims:
        ae = Autoencoder(
            input_dim=X.shape[1],
            encoding_dims=encoding_dims,
            bottleneck_dim=bottleneck,
            activation='relu',
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            l2_lambda=l2_lambda,
            random_state=random_state
        )

        ae.fit(X, verbose=False)

        X_encoded = ae.encode(X)
        reconstruction_error = ae.reconstruction_error(X)
        final_loss = ae.train_loss_history[-1]

        for k in component_values:
            for cov_type in covariance_types:
                gmm = GMM(
                    k=k,
                    max_iter=max_iter,
                    tol=tol,
                    covariance_type=cov_type,
                    random_state=random_state
                )

                gmm.fit(X_encoded)
                labels = gmm.predict(X_encoded)

                sil = silhouette_score(X_encoded, labels)
                db = davies_bouldin_score(X_encoded, labels)
                ch = calinski_harabasz_score(X_encoded, labels)

                ari = adjusted_rand_index(y, labels)
                nmi = normalized_mutual_info(y, labels)
                purity = purity_score(y, labels)

                log_likelihood = gmm.log_likelihood_
                bic = gmm.bic(X_encoded)
                aic = gmm.aic(X_encoded)

                results.append({
                    "experiment": "Autoencoder + GMM",
                    "bottleneck_dim": bottleneck,
                    "k": k,
                    "covariance_type": cov_type,
                    "reconstruction_error": reconstruction_error,
                    "final_train_loss": final_loss,
                    "silhouette": sil,
                    "davies_bouldin": db,
                    "calinski_harabasz": ch,
                    "ARI": ari,
                    "NMI": nmi,
                    "purity": purity,
                    "log_likelihood": log_likelihood,
                    "BIC": bic,
                    "AIC": aic
                })

    return results