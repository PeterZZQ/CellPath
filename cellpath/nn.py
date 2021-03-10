import numpy as np  
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


def NeighborhoodGraph(X, k_neighs, symm = True, pruning = True):
    # construct obj with sklearn, metric eculidean
    neighbor = NearestNeighbors(n_neighbors=k_neighs)
    # training state [n_samples(cells),n_features(genes)]
    neighbor.fit(X)
    # get test connectivity result 0-1 adj_matrix, mode = 'connectivity' by default
    adj_matrix = neighbor.kneighbors_graph(X).toarray()
    if symm:
        if pruning:
            adj_matrix *= adj_matrix.T
        else:
            adj_matrix += adj_matrix.T
        # change 2 back to 1
        adj_matrix = np.where(adj_matrix != 0, 1.0, 0.0)
        dist_matrix = pairwise_distances(X, metric = "euclidean")

    return adj_matrix, dist_matrix


def _assign_weights(connectivities, distances, X, velo_pca, scaling = 3, distance_scalar = 0.5, threshold = 0):
    """\
    assigning weights for the graph

    Arguments
    ---------
    connectivites
        input adjacency matrix, 0-1 connectivity matrix
    distances
        distance matrix
    X 
        input expression data, dimension reduction version
    velo_pca 
        input rna velocity data, dimension reduction version
    scaling
        Escalate the penalty
    distance_scalar
        Bias towards distance penalty 
    threshold
        Cut-off value for the angular difference
    Returns
    -------
    Returns adjacency matrix 
    """
    adj_matrix = np.zeros_like(connectivities)
    cells = adj_matrix.shape[0]
    distances = connectivities * distances
    # the resulted adj_matrix is quite sparse
    for i in range(cells):
        # the maximal distance between i and its neighbors
        dmax = np.max(distances[i,:])
        for j in range(cells):
            # cell i to cell j: 
            # if not zero, which means there's an connnection, notice the diagonal value is also one
            if connectivities[i,j] != 0 and i != j:
                # expression data for cell i
                express_i = X[i,:]
                # expression data for cell j
                express_j = X[j,:]
                # expression error
                velo_i = velo_pca[i,:]
                cos_theta = np.matmul((express_j - express_i),velo_i)\
                /(np.linalg.norm(express_j - express_i,2) * np.linalg.norm(velo_i,2))
                
                
                if cos_theta > threshold:
                    # distance penalty in [0,1]
                    # ld = np.linalg.norm(express_j-express_i,2)/dmax
                    ld = distances[i,j]/dmax
                    assert np.abs(np.linalg.norm(express_j-express_i,2) - distances[i,j]) < 1e-4
                    # direction penalty in [0,1]
                    ltheta = 1-cos_theta
                    # distance penalty and angular penalty, distance penalty is important for knn on sparse graph(around 500 nodes), product is not good for small angular but large distance
                    adj_matrix[i,j] = (scaling * (ltheta + distance_scalar * ld)) ** scaling
                
                else:
                    adj_matrix[i,j] = np.inf
            # same node can also have 0
            elif i == j:
                adj_matrix[i,j] = 0
            # disconnected nodes, assign it to infinity
            else:    
                adj_matrix[i,j] = np.inf
    
    
    return adj_matrix


def assign_weights(connectivities, distances, X, velo_pca, scaling = 3, distance_scalar = 0.5, threshold = 0):
    """\
    assigning weights for the graph

    Arguments
    ---------
    connectivites
        input adjacency matrix, 0-1 connectivity matrix
    distances
        distance matrix
    X 
        input expression data, dimension reduction version
    velo_pca 
        input rna velocity data, dimension reduction version
    scaling
        Escalate the penalty
    distance_scalar
        Bias towards distance penalty 
    threshold
        Cut-off value for the angular difference
    Returns
    -------
    Returns adjacency matrix 
    """

    cosine_sim = (-np.sum(X * velo_pca, axis = 1)[:, None] + np.matmul(velo_pca, X.T))/(1e-12 + distances * np.linalg.norm(velo_pca, 2, axis = 1)[:,None])
    # cosine_sim = (cosine_sim > threshold) * cosine_sim
    np.fill_diagonal(cosine_sim, val = 1)

    # assert np.min(cosine_sim) >= 0
    # assert np.max(cosine_sim) <= 1

    adj_matrix = (connectivities * (1 - cosine_sim) + distance_scalar * connectivities * distances/np.max(1e-12 + connectivities * distances, axis = 1)[:,None])
    adj_matrix = (scaling * adj_matrix) ** scaling
    adj_matrix = np.where((adj_matrix == 0)|(cosine_sim <= threshold), np.inf, adj_matrix)

    np.fill_diagonal(adj_matrix, val = 0)
    
    return adj_matrix

