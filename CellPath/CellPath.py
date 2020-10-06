import cellpath.visual as visual
import cellpath.clustering as clust
import cellpath.path as path
import cellpath.nn as nn
import cellpath.princurve as pcurve
import cellpath.benchmark as bmk
import matplotlib.pyplot as plt

import numpy as np  
import pandas as pd


class CellPath():
    def __init__(self, adata):
        self.adata = adata
        self.metacell_graph = None
        self.paths = None
        self.greedy_order = None

    def meta_cell_construction(self, n_clusters = None, include_unspliced = True, standardize = True, **kwarg):
        """\
        Constructing meta cell

        Parameters
        ----------
        n_clusters
            number of meta cells, default cell number/10
        include_unspliced
            Boolean, whether include unspliced count or not
        standardize
            Standardize before pca, boolean
        """
        _kwargs = {
            "n_comps": 30, 
            "init": "k-means++",
            "n_init": 10, 
            "max_iter": 300, 
            "tol": 0.0001, 
            "kernel": "rbf",
            "alpha": 1,
            "gamma": 0.3,
            "verbose": True
        }
        _kwargs.update(kwarg)

        self.groups = clust.cluster_cells(self.adata, n_clusters = n_clusters,
                                          n_comps = _kwargs["n_comps"], init = _kwargs["init"],
                                          n_init = _kwargs["n_init"], max_iter = _kwargs["max_iter"],
                                          tol = _kwargs["tol"], include_unspliced = include_unspliced,
                                          standardize = standardize)

        self.X_clust, self.velo_clust = clust.meta_cells(self.adata, kernel = _kwargs["kernel"], 
                                                         alpha = _kwargs["alpha"], gamma = _kwargs["gamma"])

        if _kwargs["verbose"] == True:
            print("Meta-cell constructed")

    def meta_cell_graph(self, k_neighs = 10, **kwargs):
        """\
        meta-cell level graph construction

        Parameters
        ----------
        k_neighs
            number of neighbors for the neighborhood graph, default 10
        """        
        _kwargs = {
            "symm": True,
            "pruning": False,
            "scaling": 3,
            "distance_scalar": 0.5,
            "threshold": 0,
            "verbose": True
        }
        _kwargs.update(kwargs)

        _adj_matrix, _dist_matrix = nn.NeighborhoodGraph(self.X_clust, k_neighs = 10, symm = _kwargs["symm"], pruning=_kwargs["pruning"])
        self.adj_assigned = nn.assign_weights(connectivities = _adj_matrix, distances = _dist_matrix, X_pca = self.X_clust, 
                                                               velo_pca = self.velo_clust, scaling = _kwargs["scaling"], 
                                                               distance_scalar = _kwargs["distance_scalar"], threshold = _kwargs["threshold"])

        self.max_weight = (_kwargs["scaling"] * (1 + _kwargs["distance_scalar"]))**_kwargs["scaling"]
        if _kwargs["verbose"] == True:
            print("Meta-cell level neighborhood graph constructed")
    
    def meta_paths_finding(self, threshold = 0.5, cutoff_length = 5, length_bias = 0.7, **kwargs):
        """\
        meta-cell level trajectory finding

        Parameters
        ----------
        threshold
            Cut-off quality score, equals to threshold * max_weight
        cutoff_length
            The cutoff length (lower bound) of inferred trajectory
        length_bias
            The bias on the path length for greedy selection
        """ 
        _kwargs = {
            "max_trajs": None,
            "verbose": True,
            "root_cell_indeg":[0,1,2]
        }

        _kwargs.update(kwargs)
        self.paths, self.opt = path.dijkstra_paths(adj = self.adj_assigned.copy(), indeg = _kwargs["root_cell_indeg"])
        
        n_metacells = int(np.max(self.groups)+1)
        self.greedy_order, self.paths = path.greedy_selection(nodes = n_metacells, paths = self.paths,opt_value = self.opt, threshold = threshold, 
                                                              max_w=self.max_weight, cut_off=cutoff_length, 
                                                              verbose = _kwargs["verbose"], length_bias = length_bias, 
                                                              max_trajs = _kwargs["max_trajs"])
            
    def first_order_pt(self, num_trajs = None, verbose = True):
        """\
        cell level pseudo-time inference using first order approximation

        Parameters
        ----------
        num_trajs
            Number of trajectories
        verbose
            Output result
        """ 
        if num_trajs == None:
            num_trajs = len(self.greedy_order)
        elif num_trajs > len(self.greedy_order):
            print(len(self.greedy_order))
            raise ValueError("number of trajectory to be selected larger than maximum number")
        self.pseudo_order = pd.DataFrame(data = np.nan, index = self.adata.obs.index, columns = ["traj_" + str(x) for x in range(num_trajs)]) 

        for i in range(num_trajs):
            traj = np.array([])

            # the traj is already sorted by enumerating process
            for index in self.paths[self.greedy_order[i]]:
                # find the cells corresponding to the cluster in greedy paths
                group_i = np.where(self.groups == index)[0]
                # ordering the cells
                diff = self.adata.obsm['X_pca'][group_i,:] - self.X_clust[index,:] 
                group_i = group_i[np.argsort(np.dot(diff, self.velo_clust[index,:])/np.linalg.norm(self.velo_clust[index,:],2))]

                # traj store all the cells in the trajectory/greedy paths
                traj = np.append(traj, group_i)  
                
            traj = traj.astype('int')

            for pt, curr_cell in enumerate(traj):
                self.pseudo_order.loc["cell_"+str(curr_cell),"traj_"+str(i)] = pt

        if verbose:
            print("Cell-level pseudo-time inferred")

    def all_in_one(self, num_metacells = None, n_neighs = 10, 
                   include_unspliced = True, standardize = True,
                   threshold = 0.5, cutoff_length = 5, 
                   length_bias = 0.7, num_trajs = None, **kwargs):
        """\
        run CellPath in one function

        Parameters
        ----------
        n_clusters
            number of meta cells, default cell number/10
        include_unspliced
            Boolean, whether include unspliced count or not
        standardize
            Standardize before pca, boolean
        k_neighs
            number of neighbors for the neighborhood graph, default 10
        threshold
            Cut-off quality score, equals to threshold * max_weight
        cutoff_length
            The cutoff length (lower bound) of inferred trajectory
        length_bias
            The bias on the path length for greedy selection
        num_trajs
            Number of trajectories
        """ 

        self.meta_cell_construction(n_clusters = num_metacells, include_unspliced = include_unspliced,
                                    standardize = standardize, **kwargs)
        self.meta_cell_graph(k_neighs = n_neighs, **kwargs)
        self.meta_paths_finding(threshold = threshold, cutoff_length = cutoff_length, length_bias = length_bias, **kwargs)
        self.first_order_pt(num_trajs = num_trajs)