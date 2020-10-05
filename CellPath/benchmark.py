import numpy as np
import anndata
import pandas as pd


def f1_score(adata, method = "CellPaths", paths = None, greedy_paths = None, slingshot_result = None, trajs = None):
    """\
    Description
        Calculate f1 score for trajectory detection    
    Parameters
    ----------
    adata
        gene expression data structure
    trajs
        number of trajectories

    Returns
    -------
    F1 
        F1 score
    """   

    # load ground truth
    if adata.obs['simulation_i'].cat.categories.dtype != 'int64':
        cell_belonging = adata.obs['simulation_i'].astype('int64').astype('category')
    else:
        cell_belonging = adata.obs['simulation_i']

    # benchmark all trajectories
    if method == "CellPaths":
        groups = adata.obs['groups'] 

        if trajs == None:
            trajs = len(greedy_paths)

        if paths == None or greedy_paths == None:
            raise ValueError("No CellPaths result provided")

    elif method == "Slingshot":
        trajs = len(slingshot_result["pseudotime"].columns)

        if slingshot_result == None:
            raise ValueError("No Slingshot result provided")

    recovery = 0

    print("recovery\n")
    for i in range(trajs):
        if method == "CellPaths":
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
            if method == "CellPaths":
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



def purity_count(adata, method = "CellPaths", paths = None, greedy_paths = None, slingshot_result = None, trajs = None):
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

    if adata.obs['simulation_i'].cat.categories.dtype != 'int64':
        cell_belonging = adata.obs['simulation_i'].astype('int64').astype('category')
    else:
        cell_belonging = adata.obs['simulation_i']


    # benchmark all trajectories
    if method == "CellPaths":
        groups = adata.obs['groups']

        if trajs == None:
            trajs = len(greedy_paths)

        if paths == None or greedy_paths == None:
            raise ValueError("No CellPaths result provided")

    elif method == "Slingshot":
        trajs = len(slingshot_result["pseudotime"].columns)

        if slingshot_result == None:
            raise ValueError("No Slingshot result provided") 

    else:
        raise ValueError("Choose CellPaths or Slingshot")


    cols = ["ori_traj_" + str(x) for x in list(cell_belonging.cat.categories)]
    rows = ["reconst_"+ str(x) for x in range(1,trajs + 1)]
    bmk_belongings = pd.DataFrame(columns = cols, index = rows, data = 0)

    for i in range(trajs):
        if method == "CellPaths":
            traj = np.array([])
            for index in paths[greedy_paths[i]]:
                group_i = np.where(groups == index)[0]
                traj = np.append(traj, group_i)       
            traj = traj.astype('int')

        elif method == "Slingshot":
            traj = np.where(~np.isnan(slingshot_result["pseudotime"][i].values))[0]

        belongings = adata[traj,:].obs['simulation_i']
        idx = belongings.value_counts().index
        idx = ["ori_traj_" + str(x) for x in idx.tolist()]   
        bmk_belongings.loc['reconst_'+ str(i+1),idx] = belongings.value_counts().values
    
    return bmk_belongings



def average_entropy(adata, bmk_belongings):
    """\
    Description
        Calculate average entropy, benchmark the purity of the inference result
    """
    aver_entropy = 0
    for traj in bmk_belongings.index:
        prop = bmk_belongings.loc[traj,:].values
        prop = prop/np.sum(prop)
        entropy_traj = - np.sum(prop * np.log(prop))
        aver_entropy += entropy_traj
    
    aver_entropy = aver_entropy/bmk_belongings.shape[0]
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
