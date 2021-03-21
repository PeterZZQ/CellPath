import numpy as np  
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


def NeighborhoodGraph(X,k_neighs, symm = True, pruning = True):
    # construct obj with sklearn, metric eculidean
    neighbor = NearestNeighbors(n_neighbors=k_neighs)
    # training state [n_samples(cells),n_features(genes)]
    neighbor.fit(X)
    # get test connectivity result 0-1 adj_matrix, mode = 'connectivity' by default
    adj_matrix = neighbor.kneighbors_graph(X)
    dist_matrix = neighbor.kneighbors_graph(X, mode='distance')
    # change from csr to ndarray
    adj_matrix = adj_matrix.toarray()
    dist_matrix = dist_matrix.toarray()
    if symm:
        print("make symmetric by ",end="")
        if pruning:
            print("pruning")
            adj_matrix *= adj_matrix.T
        else:
            print("adding")
            adj_matrix += adj_matrix.T
        # change 2 back to 1
        adj_matrix[adj_matrix.nonzero()[0],adj_matrix.nonzero()[1]] = 1
        dist_matrix = dist_matrix * adj_matrix
    return adj_matrix, dist_matrix


def RadiusNeighborhoodGraph(X, r):
    neighbor = NearestNeighbors(radius = r)
    neighbor.fit(X)
    adj_matrix = neighbor.radius_neighbors_graph(X)
    dist_matrix = neighbor.radius_neighbors_graph(X, mode='distance')
    
    # symmetric matrix
    adj_matrix = adj_matrix.toarray()
    dist_matrix = dist_matrix.toarray()

    return adj_matrix, dist_matrix


def assign_weights(connectivities, distances, X, velo_pca, scaling = 3, distance_scalar = 0.3, threshold = 0):
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

    Returns
    -------
    Returns adjacency matrix 
    """
    adj_matrix = connectivities.copy()
    cells = adj_matrix.shape[0]
    # the resulted adj_matrix is quite sparse
    for i in range(cells):
        # the maximal distance between i and its neighbors
        dmax = np.max(distances[i,:])
        for j in range(cells):
            # cell i to cell j: 
            # if not zero, which means there's an connnection, notice the diagonal value is also one
            if adj_matrix[i,j] != 0 and i != j:
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
                    ld = np.linalg.norm(express_j-express_i,2)/dmax
                    # direction penalty in [0,1]
                    ltheta = 1-cos_theta
                    
                    # considering the distance penalty and angular penalty
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



