import anndata
from scipy import sparse
import numpy as np  
import pandas as pd

def cluster_cells(
        adata, n_clusters = None, 
        n_comps = 30, init = "k-means++",
        n_init = 10, max_iter=300, 
        tol = 0.0001, include_unspliced = True, 
        standardize = True):
    """\
    Cluster cells into clusters, using K-means

    Parameters
    ----------
    adata
        The annotated data matrix.
    n_clusters
        number of clusters, default, cell number/10
    init, n_init, max_iter, tol
        K-means parameters
    include_unspliced
        Boolean, whether include unspliced count or not

    Returns
    -------
    `adata.obs[groups]`
        Array of dim (number of samples) that stores the subgroup id
        (`0`, `1`, ...) for each cell.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    if 'groups' in adata.obs:
        print("Already conducted clustering")
        return adata.obs["groups"].values.astype("int64")
    else:
        if n_clusters == None:
            n_clusters = int(adata.n_obs/10)
        kmeans = KMeans(n_clusters, init, n_init, max_iter, tol)

        X_spliced = np.log1p(adata.layers['spliced'].toarray())
        X_unspliced = np.log1p(adata.layers['unspliced'].toarray())

        if standardize:
            pca = Pipeline(
                [('scaling', StandardScaler(with_mean=True, with_std=True)), 
                ('pca', PCA(n_components=n_comps, svd_solver='arpack'))])
        else:
            pca = PCA(n_components=n_comps, svd_solver="arpack")

        # Include unspliced count for clustering, recalculate X_pca and X_concat_pca
        if include_unspliced:
            
            X_concat = np.concatenate((X_spliced,X_unspliced),axis=1)
            X_concat_pca = pca.fit_transform(X_concat)
            X_pca = pca.fit_transform(X_spliced)
            groups = kmeans.fit_predict(X_concat_pca)
            adata.obsm['X_pca'] = X_pca
                
        else:
            X_pca = pca.fit_transform(X_spliced)
            adata.obsm['X_pca'] = X_pca
            groups = kmeans.fit_predict(X_pca)
            
        velo_matrix = adata.layers["velocity"]
        # predict gene expression data
        X_pre = X_spliced + velo_matrix 
        # /np.linalg.norm(velo_matrix,axis=1)[:,None]
        adata.obsm['X_pre_pca'] = pca.transform(X_pre)
        adata.obsm['velocity_pca'] = adata.obsm['X_pre_pca'] - X_pca

        adata.obs['groups'] = groups.astype('int64')

        return groups.astype('int64')

def meta_cells(adata, kernel = 'rbf', alpha = 1, gamma = 0.3):
    """\
    estimate the expression and velocity of meta cells, using rbf kernel regression

    Parameters
    ----------
    adata
        The annotated data matrix.
    kernel
        kernel choice, default rbf kernel
    alpha 
        regularization
    gamma 
        para of rbf kernel

    Returns
    -------
    X_cluster
        meta cell expression data
    vel_cluster
        meta cell velocity data

    """
    
    if 'groups' not in adata.obs.columns or 'X_pca' not in adata.obsm:
        raise ValueError("please cluster cells first") 

    from sklearn.kernel_ridge import KernelRidge
    k = KernelRidge(alpha, kernel, gamma)
    groups = adata.obs['groups'].values
    n_clusters = int(np.max(groups) + 1)
    X_pca = adata.obsm['X_pca']
    X_cluster = np.zeros((n_clusters,X_pca.shape[1]))
    velo_cluster = np.zeros(X_cluster.shape)
    velo_pca = adata.obsm['velocity_pca']

    for c in range(n_clusters):
        indices = np.where(groups == c)[0]
        k.fit(X_pca[indices,:],velo_pca[indices,:])
        X_cluster[c,:] = np.mean(X_pca[indices,:],axis=0)
        velo_cluster[c,:] = k.predict(X_cluster[c,:][np.newaxis,:])
    
    return X_cluster, velo_cluster
    
        