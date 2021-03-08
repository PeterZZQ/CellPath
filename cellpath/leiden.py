import leidenalg as la
import anndata
from typing import Optional, Tuple, Sequence, Type, Union
from scipy import sparse
import numpy as np  
import pandas as pd

try:
    from leidenalg.VertexPartition import MutableVertexPartition
except ImportError:
    class MutableVertexPartition: pass
    MutableVertexPartition.__module__ = 'leidenalg.VertexPartition'

def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig
    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except:
        pass
    if g.vcount() != adjacency.shape[0]:
        print( 'Your adjacency matrix contained redundant nodes.' )
    return g


def cluster_cells_leiden(
    X, 
    n_clusters = 400,
    resolution = 30.0,
    random_state = 0,
    n_iterations: int = -1,
    log_trans = True,
    include_unspliced = True,
    k_neighs = 30,
    sigma = 1,
    **partition_kwargs):

    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import pdist, squareform

    try:
        import leidenalg
    except ImportError:
        raise ImportError(
            'Please install the leiden algorithm: `conda install -c conda-forge leidenalg` or `pip3 install leidenalg`.'
        )

    partition_kwargs = dict(partition_kwargs)


    neighbor = NearestNeighbors(n_neighbors=k_neighs)
    neighbor.fit(X)
    # get test connectivity result 0-1 adj_matrix, mode = 'connectivity' by default
    adj_matrix = neighbor.kneighbors_graph(X).toarray()
    dist_matrix = neighbor.kneighbors_graph(X, mode='distance').toarray()

    adj_matrix += adj_matrix.T
    # change 2 back to 1
    adj_matrix[adj_matrix.nonzero()[0],adj_matrix.nonzero()[1]] = 1

    affin = np.exp(- (dist_matrix - np.min(dist_matrix, axis = 1)[:,None]) ** 2/sigma)
    affin = adj_matrix * affin
    affin = (affin + affin.T)/2
        
    partition_type = leidenalg.RBConfigurationVertexPartition
    g = get_igraph_from_adjacency(affin, directed = True)

    partition_kwargs['n_iterations'] = n_iterations
    partition_kwargs['seed'] = random_state
    partition_kwargs['resolution_parameter'] = resolution

    partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)
    part = leidenalg.find_partition(g, partition_type, **partition_kwargs)
    groups = np.array(part.membership)

    return groups
