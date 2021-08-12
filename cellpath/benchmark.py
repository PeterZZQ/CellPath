import numpy as np
import anndata
import pandas as pd


def f1_score(cellpath_obj = None, adata = None, method = "CellPath", slingshot_result = None, trajs = None):
    """\
    Description
        Calculate f1 score for trajectory detection    
    Parameters
    ----------
    cellpath_obj
        cellpath object
    trajs
        number of trajectories

    Returns
    -------
    F1 
        F1 score
    """
    if method == "CellPath":
        if cellpath_obj is None:
            raise ValueError("cellpath object must be provided for CellPath")

        adata = cellpath_obj.adata
        paths = cellpath_obj.paths
        greedy_paths = cellpath_obj.greedy_order
        groups = cellpath_obj.groups

        # number of trajectory
        if trajs == None:
            trajs = len(greedy_paths)

    elif method == "Slingshot":
        if slingshot_result is None:
            raise ValueError("No Slingshot result provided")

        if adata is None:
            raise ValueError("adata must be provided for Slingshot")

        trajs = len(slingshot_result["pseudotime"].columns)

    else:
        raise ValueError("only provide benchmark between Slingshot and CellPath")

    # load ground truth
    if adata.obs['simulation_i'].cat.categories.dtype != 'int64':
        cell_belonging = adata.obs['simulation_i'].astype('int64').astype('category')
    else:
        cell_belonging = adata.obs['simulation_i']
        
    recovery = 0

    print("recovery\n")
    for i in range(trajs):
        if method == "CellPath":
            traj = np.array([])
            for index in paths[greedy_paths[i]]:
                group_i = np.where(groups == index)[0]
                traj = np.append(traj, group_i)       
            traj = traj.astype('int')
        elif method == "Slingshot":
            traj = np.where(~np.isnan(slingshot_result["pseudotime"][i].values))[0]

        infer_pool = set([x for x in traj])

        max_jaccard = 0

        for clust in cell_belonging.cat.categories:
            # original trajectory/cluster
            cells = cell_belonging[cell_belonging == clust].index
            ori_pool = set([eval(x.split("_")[1]) for x in cells])
            jaccard = len(infer_pool.intersection(ori_pool))/len(infer_pool.union(ori_pool))
            max_jaccard = max([jaccard, max_jaccard])

        print("inferred trajectory:", i, "jaccard:", max_jaccard)
        recovery += max_jaccard

    recovery = recovery / trajs 

    print("recovery value:", recovery)

    relevence = 0

    print("relevence\n")
    for clust in cell_belonging.cat.categories:
        # original trajectory/cluster
        cells = cell_belonging[cell_belonging == clust].index
        ori_pool = set([eval(x.split("_")[1]) for x in cells])

        max_jaccard = 0

        for i in range(trajs):
            if method == "CellPath":
                traj = np.array([])
                for index in paths[greedy_paths[i]]:
                    group_i = np.where(groups == index)[0]
                    traj = np.append(traj, group_i)       
                traj = traj.astype('int')
            elif method == "Slingshot":
                traj = np.where(~np.isnan(slingshot_result["pseudotime"][i].values))[0]

            infer_pool = set([x for x in traj])
            jaccard = len(infer_pool.intersection(ori_pool))/len(infer_pool.union(ori_pool))
            max_jaccard = max([jaccard, max_jaccard])

        print("ori trajectory:", clust, "jaccard:", max_jaccard)
        relevence += max_jaccard

    relevence = relevence / len(cell_belonging.cat.categories)

    print("relevence value:", relevence)

    F1 = 2/(1/recovery + 1/relevence)
    return F1


def purity_count(cellpath_obj = None, adata = None, method = "CellPath", paths = None, greedy_paths = None, slingshot_result = None, trajs = None):
    """\
    Description
        Calculated the purity count
    
    Parameters
    ----------
    adata
        gene expression data structure
    trajs
        number of trajectories
    groups
        list that store the belonging of each cell

    Returns
    -------
    bmk_belongings 
        the data frame for the purity count
    """  
    if method == "CellPath": 
        groups = cellpath_obj.groups

        if cellpath_obj is None:
            raise ValueError("Cellpath object must be provided") 

        adata = cellpath_obj.adata
        paths = cellpath_obj.paths
        greedy_paths = cellpath_obj.greedy_order

        if trajs is None:
            trajs = len(greedy_paths)

    elif method == "Slingshot":
        if adata is None:
            raise ValueError("adata must be provided")
        if slingshot_result is None:
            raise ValueError("No Slingshot result provided") 

        trajs = len(slingshot_result.columns)

    else:
        raise ValueError("Choose CellPath or Slingshot")


    if adata.obs['simulation_i'].cat.categories.dtype != 'int64':
        cell_belonging = adata.obs['simulation_i'].astype('int64').astype('category')
    else:
        cell_belonging = adata.obs['simulation_i']

    cols = ["ori_traj_" + str(x) for x in list(cell_belonging.cat.categories)]
    rows = ["reconst_"+ str(x) for x in range(1,trajs + 1)]
    bmk_belongings = pd.DataFrame(columns = cols, index = rows, data = 0)

    for i in range(trajs):
        if method == "CellPath":
            traj = np.array([])
            for index in paths[greedy_paths[i]]:
                group_i = np.where(groups == index)[0]
                traj = np.append(traj, group_i)       
            traj = traj.astype('int')

        elif method == "Slingshot":
            traj = np.where(~np.isnan(slingshot_result[i].values))[0]

        belongings = adata[traj,:].obs['simulation_i']
        idx = belongings.value_counts().index
        idx = ["ori_traj_" + str(x) for x in idx.tolist()]   
        bmk_belongings.loc['reconst_'+ str(i+1),idx] = belongings.value_counts().values
    
    return bmk_belongings


def average_entropy(bmk_belongings):
    """\
    Description
        Calculate average entropy, benchmark the purity of the inference result
    """
    aver_entropy = 0
    count = 0
    for traj in bmk_belongings.index:
        prop = bmk_belongings.loc[traj,:].values
        if np.sum(prop) == 0:
            continue 
        else:
            prop = prop/np.sum(prop) 
        entropy_traj = - np.sum(prop * np.log(prop + 1e-6))
        aver_entropy += entropy_traj
        count += 1
    
    aver_entropy = aver_entropy/count
    max_entropy = np.ones([1, bmk_belongings.shape[1]]) * 1/bmk_belongings.shape[1] 
    max_entropy = - np.sum(max_entropy * np.log(max_entropy))

    aver_entropy = 1 - aver_entropy / max_entropy

    return aver_entropy
    

def meta_cell_comp(adata, groups = None):
    """\
    Description
        Calculated the purity of meta cell clusters
    
    Parameters
    ----------
    adata
        gene expression data structure
    groups
        list that store the belonging of each cell
    Returns
    -------
    bmk_belongings 
        the data frame for the purity count
    """
    if groups == None:    
        groups = adata.obs['groups']

    # make sure the datatype is int    
    if adata.obs['simulation_i'].cat.categories.dtype != 'int64':
        df = adata.obs['simulation_i'].astype('int64').astype('category')
    else:
        df = adata.obs['simulation_i']
    clusters = int(np.max(groups) + 1)
    cols = ["simulation_" + str(x) for x in list(df.cat.categories)]
    rows = ["group_"+ str(x) for x in range(clusters)]
    components = pd.DataFrame(columns = cols, index = rows, data = 0)  
    for c in range(clusters):
        idx = np.where(groups == c)[0]
        components.loc["group_"+ str(c),:] = df[idx].value_counts().values #/df[idx].count()
    return components


def kendalltau(pt_pred, pt_true):
    """\
    Description
        kendall tau correlationship
    
    Parameters
    ----------
    pt_pred
        inferred pseudo-time
    pt_true
        ground truth pseudo-time
    Returns
    -------
    tau
        returned score
    """
    from scipy.stats import kendalltau
    pt_true = pt_true.squeeze()
    pt_pred = pt_pred.squeeze()
    tau, p_val = kendalltau(pt_pred, pt_true)
    return tau


def cellpath_kt(cellpath_obj):
    """\
    Description
        kendall tau correlationship for CellPath
    
    Parameters
    ----------
    cellpath_obj
        Cellpath object
    Returns
    -------
    kt
        returned score    
    """
    if "sim_time" not in cellpath_obj.adata.obs.columns:
        raise ValueError("ground truth value not provided")

    pseudo_order = cellpath_obj.pseudo_order
    non_zeros = {}
    pt_pred = {}
    pt_true = {}
    kt = {}
    for icol, col in enumerate(pseudo_order.columns):
        non_zeros[col] = np.where(~np.isnan(pseudo_order[col].values.squeeze()))[0]
        pt_pred[col] = pseudo_order.iloc[non_zeros[col], icol].values.squeeze()
        pt_true[col] = cellpath_obj.adata.obs["sim_time"].iloc[non_zeros[col]].values
        kt[col] = kendalltau(pt_pred[col], pt_true[col])
    return kt