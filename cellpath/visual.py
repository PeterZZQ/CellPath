import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import cellpath.benchmark as bmk

def plot_data(cellpath_obj, basis = "umap", figsize = (15,7), save_as = None, title = None, **kwargs):
    """\
    Description
        Plot original dataset
    
    Parameters
    ----------
    cellpath_obj
        cellpath object
    basis
        the basis used for visualization
    figsize
        Figure size, tuple
    save_as
        Name of the saved file
    """

    _kwargs = {
        "axis": False,
        "legend_pos": "upper left",
        "colormap": "tab20",
        "s": 10,
        "add_arrow": False,
        "markerscale": 1.0
    }
    _kwargs.update(kwargs)

    X = cellpath_obj.adata.obsm["X_" + basis]
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot()
    if _kwargs["axis"]:
        ax.tick_params(axis = "both", direction = "out", labelsize = 16)
        ax.set_xlabel(basis + " 1", fontsize = 19)
        ax.set_ylabel(basis + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    else:
        ax.axis("off")

    if "clusters" in cellpath_obj.adata.obs.columns:
        cluster_anno = [x for x in cellpath_obj.adata.obs["clusters"].values]
        cluster_uni = [x for x in np.unique(cellpath_obj.adata.obs["clusters"].values)]

        colormap = plt.cm.get_cmap(_kwargs["colormap"], len(cluster_uni))

        for count, clust in enumerate(cluster_uni):
            idx = np.where(np.array(cluster_anno) == clust)[0]
            ax.scatter(X[idx,0], X[idx,1], color = colormap(count), alpha = 0.7, label = clust, s = _kwargs["s"])    
        ax.legend(loc=_kwargs["legend_pos"], prop={'size': 15}, frameon = False, ncol = 1, markerscale=_kwargs["markerscale"])
    
    elif "sim_time" in cellpath_obj.adata.obs.columns:
        X_ordered = X[np.argsort(cellpath_obj.adata.obs["sim_time"].values),:]
        if _kwargs["add_arrow"]:
            for i in range(X_ordered.shape[0]-1):
                line = ax.plot(X_ordered[i:(i+2), 0], X_ordered[i:(i+2), 1], 'gray', '-', alpha = 1)
                add_arrow(line[0], size = 5)

        pic = ax.scatter(X_ordered[:,0], X_ordered[:,1], cmap = "gnuplot", c = np.arange(X_ordered.shape[0]), alpha = 1, s = _kwargs["s"])
        cbar = fig.colorbar(pic, fraction=0.046, pad=0.04, ax = ax)
        cbar.ax.tick_params(labelsize = 20)
        

    else:
        ax.scatter(X[:,0], X[:,1], color = "red", alpha = 0.7, s = _kwargs["s"])

    if title is not None:
        ax.set_title(title, fontsize = 25)
    if save_as != None:
        fig.savefig(save_as, bbox_inches = 'tight')


def add_arrow(line, position = None, direction='right', **kwargs):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    arrow_size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    _kwargs = {
        "arrow_size": 15,
        "linewidth": 2,
        "alpha": 0.7,
        "color": None,
        "arrow_style": "->"
    }
    _kwargs.update(kwargs)

    if _kwargs["color"] is None:
        color = line.get_color()
    else:
        color = _kwargs["color"]

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    
    if direction == 'right':
        start_ind = 0
        end_ind = 1
    else:
        end_ind = 0
        start_ind = 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle = _kwargs["arrow_style"], color=color, linewidth = _kwargs["linewidth"], alpha = _kwargs["alpha"]),
        size=_kwargs["arrow_size"]
    )


def first_order_approx_pt(cellpath_obj, basis = "pca", trajs = 4, figsize= (20,20), save_as = None, title = None, axis = True):
    """\
    Description        
        Plot the first order approximation estimation of pseudo-time

    Parameters
    ----------
    cellpath_obj
        Cellpath object
    basis
        the basis used for visualization
    trajs
        The number of trajectories plotted, note that trajs cannot be too much, or there may exists trajs contains too few cells, may even less than the neighs parameter
    figsize
        Figure size, tuple
    save_as
        Name of the saved file
    title
        The title name of the plot
    axis
        Show axis or not, boolean value
    """
    if trajs >= 2:
        nrows = np.ceil(trajs/2).astype('int32')
        ncols = 2

    elif trajs == 1:
        nrows = 1
        ncols = 1        
        
    else:
        raise ValueError("invalid trajectory numbers")

    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)

    if title:
        fig.suptitle("first order approximate pseudo-time", fontsize = 18)

    adata = cellpath_obj.adata
    basis = 'X_' + basis
    if basis not in adata.obsm.keys():
        raise ValueError("basis incorrect")


    for i in range(min(trajs, cellpath_obj.pseudo_order.shape[1])):
        sorted_pt = cellpath_obj.pseudo_order["traj_"+str(i)].dropna(axis = 0).sort_values()
        # traj = [int(x.split("_")[1]) for x in sorted_pt.index]
        # X_traj = adata.obsm[basis][traj,:]
        X_traj = adata[sorted_pt.index,:].obsm[basis]
 
        if nrows != 1:
            # multiple >2 plots
            if not axis:
                axs[i%nrows, i//nrows].axis("off")
            axs[i%nrows, i//nrows].scatter(adata.obsm[basis][:,0],adata.obsm[basis][:,1], color = 'gray', alpha = 0.1)
            

            pseudo_visual = axs[i%nrows, i//nrows].scatter(X_traj[:,0],X_traj[:,1],c = np.arange(X_traj.shape[0]), cmap=plt.get_cmap('gnuplot'), alpha = 0.7)

            axs[i%nrows, i//nrows].set_title("CellPath: Path " + str(i), fontsize = 25)
            axs[i%nrows, i//nrows].set_xlabel(basis.split("_")[1] + " 1", fontsize = 19)
            axs[i%nrows, i//nrows].set_ylabel(basis.split("_")[1] + " 2", fontsize = 19)
            axs[i%nrows, i//nrows].set_xticks([])
            axs[i%nrows, i//nrows].set_yticks([])
            axs[i%nrows, i//nrows].spines['right'].set_visible(False)
            axs[i%nrows, i//nrows].spines['top'].set_visible(False)

            cbar = fig.colorbar(pseudo_visual,fraction=0.046, pad=0.04, ax = axs[i%nrows, i//nrows])
            cbar.ax.tick_params(labelsize = 20)

        elif nrows == 1 and ncols == 1:
            # one plot
            if not axis:
                axs.axis("off")            
            axs.scatter(adata.obsm[basis][:,0],adata.obsm[basis][:,1], color = 'gray', alpha = 0.1)

            pseudo_visual = axs.scatter(X_traj[:,0],X_traj[:,1],c = np.arange(X_traj.shape[0]), cmap=plt.get_cmap('gnuplot'),alpha = 0.7)

            axs.set_title("CellPath: Path " + str(i), fontsize = 25)
            axs.set_xlabel(basis.split("_")[1] + " 1", fontsize = 19)
            axs.set_ylabel(basis.split("_")[1] + " 2", fontsize = 19)
            axs.set_xticks([])
            axs.set_yticks([])
            axs.spines['right'].set_visible(False)
            axs.spines['top'].set_visible(False)   

            cbar = fig.colorbar(pseudo_visual,fraction=0.046, pad=0.04, ax = axs)
            cbar.ax.tick_params(labelsize = 20)

        else:
            # two plots
            if not axis:
                axs[i].axis("off")
            axs[i].scatter(adata.obsm[basis][:,0],adata.obsm[basis][:,1], color = 'gray', alpha = 0.1)
            pseudo_visual = axs[i].scatter(X_traj[:,0],X_traj[:,1],c = np.arange(X_traj.shape[0]), cmap=plt.get_cmap('gnuplot'),alpha = 0.7)
            
            axs[i].set_title("CellPath: Path " + str(i), fontsize = 25)
            axs[i].set_xlabel(basis.split("_")[1] + " 1", fontsize = 19)
            axs[i].set_ylabel(basis.split("_")[1] + " 2", fontsize = 19)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)

            cbar = fig.colorbar(pseudo_visual,fraction=0.046, pad=0.04, ax = axs[i])
            cbar.ax.tick_params(labelsize = 20)

 
    
    if save_as!= None:
        fig.savefig(save_as, bbox_inches = 'tight')
    
    plt.show()     


def meta_traj_visual(cellpath_obj, basis = "pca", trajs = 4, 
                     figsize = (20,10), save_as = None, title = None, **kwargs):
    """\
    Description        
        Meta-cell-level trajectory visualization

    Parameters
    ----------
    cellpath_obj
        Cellpath object
    basis
        the basis used for visualization
    trajs
        The number of trajectories plotted, note that trajs cannot be too much, or there may exists trajs contains too few cells, may even less than the neighs parameter
    figsize
        Figure size, tuple
    save_as
        Name of the saved file
    title
        The title name of the plot
    """
    _kwargs = {
        "arrow_size": 30,
        "linewidth": 3,
        "alpha": 1,
        "color": None,
        "arrow_style": "->",
        "legend_pos": "upper right", 
        "axis": False,
        "colormap": "tab20b",
        "bbox_to_anchor":(1.05, 1),
        "markerscale": 1.0    
    }

    _kwargs.update(kwargs)

    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot()

    if _kwargs["axis"]:
        ax.tick_params(axis = "both", direction = "out", labelsize = 16)
        ax.set_xlabel(basis + " 1", fontsize = 19)
        ax.set_ylabel(basis + " 2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    else:
        ax.axis('off')

    # get cmap
    colormap = plt.cm.get_cmap(_kwargs["colormap"], trajs)

    # path_color = get_cmap(trajs, 'tab20b')
    if "X_" + basis not in cellpath_obj.adata.obsm:
        raise ValueError("please calculate " + basis + " first")
    else:
        X_cell = cellpath_obj.adata.obsm["X_"+basis].copy()
        groups = cellpath_obj.groups
        X_clust = np.zeros((int(np.max(groups) + 1), X_cell.shape[1]))
        for c in np.unique(groups):
            indices = np.where(groups == c)[0]
            X_clust[c,:] = np.mean(X_cell[indices,:], axis = 0)

    # draw clusters 
    ax.scatter(X_clust[0:-1:1,0], X_clust[0:-1:1,1], color = 'gray', alpha = 0.5)

    # draw meta-cell trajectories
    for i in range(trajs):
        start_point = True
        for index in cellpath_obj.paths[cellpath_obj.greedy_order[i]]:
            if start_point:
                coord_pri = X_clust[index,0:2]
                start_point = False
            else:
                coord_pri = coord_x.copy()
            # new coord
            coord_x = X_clust[index,0:2]
            line = plt.plot([coord_pri[0],coord_x[0]],[coord_pri[1],coord_x[1]],'o-', 
                            linewidth = _kwargs["linewidth"], color = colormap(i), alpha = _kwargs["alpha"])
            add_arrow(line[0], **_kwargs)

        line[0].set_label('Path '+str(i))
        plt.legend(loc = _kwargs["legend_pos"], prop = {'size': 20}, frameon = False, ncol = 2, bbox_to_anchor=_kwargs["bbox_to_anchor"], markerscale = _kwargs["markerscale"])

    if title != None:
        plt.title(title)
    if save_as!= None:
        fig.savefig(save_as, bbox_inches = 'tight')


def traj_visual(cellpath_obj, trajs = 4, figsize = (15,10), save_as = None, title = None, axis = True, cmap = "tab20b"):
    """\
    Description        
        Plot the trajectories using principal curve method, visualized using principal curve. Note that the principal curve may not be plotted very well with complex lineage structure.

    Parameters
    ----------
    cellpath_obj
        Cellpath object
    trajs
        The number of trajectories plotted, note that trajs cannot be too much, or there may exists trajs contains too few cells, may even less than the neighs parameter
    figsize
        Figure size, tuple
    save_as
        Name of the saved file
    title
        The title name of the plot
    axis
        Show axis or not, boolean value
    cmap
        Colormap
    """
    try:
        import cellpath.princurve as pcurve
    except:
        raise MissingDependency("Please install rpy2 for principal curve visualization")

    groups = cellpath_obj.groups
    adata = cellpath_obj.adata

    fig = plt.figure(figsize = figsize)
    
    ax = fig.add_subplot()
    if not axis:
        ax.axis('off')

    if 'simulation_i' in adata.obs.columns:
        if adata.obs['simulation_i'].cat.categories.dtype != 'int64': 
            adata.obs['simulation_i'] = adata.obs['simulation_i'].astype('int64').astype('category')
        
        ori_trajs = np.max([x for x in adata.obs['simulation_i'].cat.categories])

        # get cmap
        if cmap != None:
            colormap = plt.cm.get_cmap(cmap, ori_trajs)
        else:
            colormap = lambda i: np.random.rand(3,ori_trajs)[:,i]
        
        for i in range(ori_trajs):
            origin_traj = adata[adata.obs['simulation_i']==(i+1)].obsm['X_pca']
            ax.scatter(origin_traj[:,0],origin_traj[:,1],color = colormap(i), alpha = 0.5)
        ax.legend(["original traj "+str(i) for i in np.arange(ori_trajs)], loc = "upper left", prop = {"size": 20}, frameon = False, ncol = 2)
    else:
        ax.scatter(adata.obsm['X_pca'][:,0],adata.obsm['X_pca'][:,1],color = 'red', alpha = 0.5)

    for i in range(trajs):
        traj = np.array([])
        for index in cellpath_obj.paths[cellpath_obj.greedy_order[i]]:
            group_i = np.where(groups == index)[0]
            traj = np.append(traj, group_i)       
        traj = traj.astype('int')
        X_traj = adata.obsm['X_pca'][traj,:]
        results = pcurve.principalcurve(X_traj)
        projs = results['projections']
        orders = results['order']-1
        projs = projs[orders,:]

        # pseudo_order['reconst_'+str(i+1)] = adata[traj,:][orders,:].obs['sim_time']
        ax.plot(projs[:,0],projs[:,1],'b-',linewidth = 3, alpha = 0.7)

        ax.tick_params(axis = "both", direction = "out", labelsize = 16)
        ax.set_xlabel("PC1", fontsize = 19)
        ax.set_ylabel("PC2", fontsize = 19)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    if title != None:
        plt.title(title)        
    if save_as != None:
        fig.savefig(save_as, bbox_inches = 'tight')


def slingshot_visual(adata, results, basis = "umap", figsize = (20,10), save_as = None, title = None, axis = True, use_pcurve = False):
    """\
    Description        
        Infer pseudo-time using Slingshot, and return kendall-tau coefficient as accuracy value

    Parameters
    ----------
    adata
        AnnData
    results
        Slingshot result
    basis
        the basis used for visualization
    figsize
        Figure size, tuple
    save_as
        Name of the saved file
    title
        The title name of the plot
    axis
        Show axis or not, boolean value
    use_pcurve:
        Use pcurve or not

    Return
    ----------
    kt
        Kendall-tau coefficient          
    """
    if "X_" + basis not in adata.obsm:
        raise ValueError("please calculate " + basis + " first")
    else:
        X = adata.obsm['X_' + basis]

    trajs = results['pseudotime'].shape[1]
    nrows = np.ceil(trajs).astype('int32')
    ncols = 1
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
    if title:
        fig.suptitle("Slingshot pseudo-time", fontsize = 18)

    kt = {}

    for i in range(trajs):
        if nrows == 1 or ncols == 1:
            if nrows == 1 and ncols == 1:
                ax = axs
            else:
                ax = axs[i]
        else:
            ax = axs[i%nrows, i//nrows]

        ax.scatter(X[:,0],X[:,1], color = 'gray', alpha = 0.1)
        if use_pcurve:
            ax.plot(results['curves'][i,:,0],results['curves'][i,:,1],color = 'black', linewidth = 4)

        # kendall-tau
        pt_i = results['pseudotime'].iloc[:,i]
        pt_index = [x for x in pt_i.index if not np.isnan(pt_i[x])]
        ordering = [eval(x) for x in pt_i[pt_index].sort_values().index]
        pt_i = pt_i[pt_index]
        true_i = adata.obs['sim_time'].iloc[pt_index].values
        kt["traj_"+str(i)] = bmk.kendalltau(pt_i, true_i)

        X_traj = X[ordering,:]

        pseudo_visual = ax.scatter(X_traj[:,0],X_traj[:,1],c = np.arange(X_traj.shape[0]), cmap=plt.get_cmap('gnuplot'),alpha = 0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(basis + " 1", fontsize = 19)
        ax.set_ylabel(basis + " 2", fontsize = 19)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        cbar = fig.colorbar(pseudo_visual,fraction=0.046, pad=0.04, ax = ax)
        cbar.ax.tick_params(labelsize = 20)
        ax.set_title("Slingshot: Path " + str(i), fontsize = 25)

    if save_as != None:
        fig.savefig(save_as,bbox_inches = 'tight')
    plt.show()    
    return kt


def purity_bar(bmk_belongings, figsize = (20,20), save_as = None):
    """\
    Description        
        Plot the bar plot that shows the purity of trajectories separation, run purity_count first to get bmk_belongings

    Parameters
    ----------
    bmk_belongings
        data frame that store the purity results
    figsize
        figure size, tuple
    save_as
        saved file name
    """
    trajs = bmk_belongings.shape[0]

    if trajs >= 2:
        nrows = np.ceil(trajs/2).astype('int32')
        ncols = 2

    elif trajs == 1:
        nrows = 1
        ncols = 1        
        
    else:
        raise ValueError("invalid trajectory numbers")

    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)

    if nrows != 1:
        for reconst in range(trajs):
            axs[reconst%nrows, reconst//nrows].bar(x = np.arange(bmk_belongings.shape[1]), height = bmk_belongings.iloc[reconst].to_numpy(),alpha = 0.5,width = 0.5, align = "center")
            axs[reconst%nrows, reconst//nrows].set_xticks(np.arange(bmk_belongings.shape[1]))
            axs[reconst%nrows, reconst//nrows].set_xticklabels(list(bmk_belongings.columns))
            axs[reconst%nrows, reconst//nrows].set_title(bmk_belongings.index[reconst])
    
    elif nrows == 1 and ncols == 1:    
        for reconst in range(trajs):
            axs.bar(x = np.arange(bmk_belongings.shape[1]), height = bmk_belongings.iloc[reconst].to_numpy(),alpha = 0.5,width = 0.5, align = "center")
            axs.set_xticks(np.arange(bmk_belongings.shape[1]))
            axs.set_xticklabels(list(bmk_belongings.columns))
            axs.set_title(bmk_belongings.index[reconst])

    else:
         for reconst in range(trajs):
            axs[reconst].bar(x = np.arange(bmk_belongings.shape[1]), height = bmk_belongings.iloc[reconst].to_numpy(),alpha = 0.5,width = 0.5, align = "center")
            axs[reconst].set_xticks(np.arange(bmk_belongings.shape[1]))
            axs[reconst].set_xticklabels(list(bmk_belongings.columns))
            axs[reconst].set_title(bmk_belongings.index[reconst])   

    if save_as!= None:
        fig.savefig(save_as, bbox_inches = 'tight')
    plt.show()


def radius_hist(cellpath_obj, resolution = None):
    """\
    Description        
        Plot the distribution of the cluster radius 

    Parameters
    ----------
    cellpath_obj
        Cellpath object
    resolution
        resolution of the histogram

    Returns
    -------
    """    
    groups = cellpath_obj.groups    
    clusters = int(np.max(groups) + 1)

    # had to recalculate...
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline   
    pca = Pipeline(
    [('scaling', StandardScaler(with_mean=True, with_std=True)), 
    ('pca', PCA(n_components=30, svd_solver='arpack'))])
    X_spliced = np.log1p(cellpath_obj.adata.layers['spliced'].toarray())
    X_unspliced = np.log1p(cellpath_obj.adata.layers['unspliced'].toarray())
    X_concat = np.concatenate((X_spliced,X_unspliced),axis=1)
    X_concat_pca = pca.fit_transform(X_concat)

    radius = np.zeros(clusters)
    print("The clusters with radius close to 0 has components number:")
    for c in range(clusters):
        indices = np.where(groups == c)[0]
        for idx in indices:
            diff = np.linalg.norm(X_concat_pca[idx,:] - X_concat_pca[indices,:],ord=2,axis=0)
            if radius[c] < np.max(diff):
                radius[c] = np.max(diff)
        # check the reason for radius close to 0
        radius[c] = radius[c]/2
        if np.isclose(radius[c],0,rtol=1):
            print(indices.shape[0], end =", ")
    
    # plot
    _, (ax1, ax2) = plt.subplots(1,2,figsize = (20,7))
    if resolution == None:
        _ = ax1.hist(radius,bins=int(np.max(radius)))
    else:
        _ = ax1.hist(radius,bins=resolution)
    ax1.set_title('Radius Histogram')
    max_group = np.argmax(radius)
    # ax = plt.axes(projection = '3d')
    ax2.scatter(X_concat_pca[:,0],X_concat_pca[:,1],color = 'red',alpha = 0.3)
    max_indices = np.where(groups == max_group)[0]
    ax2.scatter(X_concat_pca[max_indices,0],X_concat_pca[max_indices,1],color = 'blue',alpha = 0.3)
    ax2.set_title('largest cluster')
    return max_group


def weight_histogram(cellpath_obj, resolution = 100):
    """\
    Description        
        Plot the histogram of the adjacency matrix

    Parameters
    ----------
    cellpath_obj
        Cellpath object
    resolution
        The resolution of the histogram
    """
    adj = cellpath_obj.adj_assigned
    count = np.where(adj.flatten() == np.inf, 0, adj.flatten())
    count = count[np.where(count!=0)[0]]
    _ = plt.hist(count, bins= resolution, range=(0,cellpath_obj.max_weight))

