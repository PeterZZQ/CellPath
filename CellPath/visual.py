import numpy as np 
import matplotlib.pyplot as plt
import cellpath.princurve as pcurve
import pandas as pd
import cellpath.benchmark as bmk

def add_arrow(line, position=None, arrow_line = True, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

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

    if arrow_line:
        line.axes.annotate('',
            xytext=(xdata[start_ind], ydata[start_ind]),
            xy=(xdata[end_ind], ydata[end_ind]),
            arrowprops=dict(arrowstyle="->", color=color), # , linewidth = 4, alpha = 0.7
            size=size
        )
    else:
        line.axes.annotate('',
            xytext=(xdata[start_ind], ydata[start_ind]),
            xy=(xdata[end_ind], ydata[end_ind]),
            arrowprops=dict(arrowstyle="-", color=color),
            size=size
        )



def weight_histogram(cellpath_obj, scaling = 100):
    """\
    plot the histogram of the adjacency matrix

    Parameters
    ----------
    cellpath_obj
        cellpath object

    Returns
    -------
    """
    adj = cellpath_obj.adj_assigned
    count = np.where(adj.flatten() == np.inf, 0, adj.flatten())
    count = count[np.where(count!=0)[0]]
    _ = plt.hist(count, bins= scaling, range=(0,cellpath_obj.max_weight))


def first_order_approx_pt(cellpath_obj, basis = 'umap', trajs = 8, figsize= (20,20), save_as = None, title = None, axis = True):
    """\
    plot the first order approximation estimation of pseudo-time

    Parameters
    ----------
    cellpath_obj
        cellpath object
    basis
        the basis used for visualization
    trajs
        the number of trajectories plotted
    figsize
        figure size, tuple
    save_as
        saved file name
    title
        The title of the plot
    axis
        Whether show axis or not

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


    for i in range(trajs):
        sorted_pt = cellpath_obj.pseudo_order["traj_"+str(i)].dropna(axis = 0).sort_values()
        traj = [int(x.split("_")[1]) for x in sorted_pt.index]
        X_traj = adata.obsm[basis][traj,:]
 
        if nrows != 1:
            # multiple >2 plots
            if not axis:
                axs[i%nrows, i//nrows].axis("off")
            axs[i%nrows, i//nrows].scatter(adata.obsm[basis][:,0],adata.obsm[basis][:,1], color = 'gray', alpha = 0.1)
            

            pseudo_visual = axs[i%nrows, i//nrows].scatter(X_traj[:,0],X_traj[:,1],c = np.arange(X_traj.shape[0]), cmap=plt.get_cmap('gnuplot'), alpha = 0.7)

            axs[i%nrows, i//nrows].set_title("CellPaths: Path " + str(i), fontsize = 25)
            axs[i%nrows, i//nrows].set_xlabel("PC1", fontsize = 19)
            axs[i%nrows, i//nrows].set_ylabel("PC2", fontsize = 19)
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

            # for idx in range(X_traj.shape[0]-1):
            #     pseudo_visual = axs.plot([X_traj[idx,0], X_traj[idx + 1,0]],[X_traj[idx, 1], X_traj[idx + 1, 1]], color = "gray", alpha = 0.7)
            #     add_arrow(pseudo_visual[0])
            pseudo_visual = axs.scatter(X_traj[:,0],X_traj[:,1],c = np.arange(X_traj.shape[0]), cmap=plt.get_cmap('gnuplot'),alpha = 0.7)

            axs.set_title("CellPaths: Path " + str(i), fontsize = 25)
            axs.set_xlabel("PC1", fontsize = 19)
            axs.set_ylabel("PC2", fontsize = 19)
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
            
            axs[i].set_title("CellPaths: Path " + str(i), fontsize = 25)
            axs[i].set_xlabel("PC1", fontsize = 19)
            axs[i].set_ylabel("PC2", fontsize = 19)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)

            cbar = fig.colorbar(pseudo_visual,fraction=0.046, pad=0.04, ax = axs[i])
            cbar.ax.tick_params(labelsize = 20)

 
    
    if save_as!= None:
        fig.savefig(save_as, bbox_inches = 'tight')
    
    plt.show()     


def meta_traj_visual(X_cluster, paths, greedy_paths, basis = None, trajs = 4, title = None, figsize = (20,10), save_as = None, cmap = 'tab20b', axis = True):
    """\
    Meta-cell-level trajectory visualization

    Parameters
    ----------
    X_cluster
        meta-cell-level expression data
    paths
        paths generated by path finding algorithm
    greedy_paths
        paths index ordered greedily
    basis
        the basis used for visualization
    trajs
        the number of trajectories plotted
    title
        the title of the figure
    figsize
        the size of the figure
    save_as
        file name
    cmap
        color map
    axis
        Whether show axis or not        

    Returns
    -------

    """

    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot()
    if not axis:
        ax.axis('off')

    # get cmap
    if cmap != None:
        colormap = plt.cm.get_cmap(cmap, trajs)
    else:
        colormap = lambda i: np.random.rand(3,trajs)[:,i]
    # path_color = get_cmap(trajs, 'tab20b')

    ax.scatter(X_cluster[0:-1:1,0], X_cluster[0:-1:1,1], color = 'gray', alpha = 0.5)

    ax.set_xlabel("PC1", fontsize = 19)
    ax.set_ylabel("PC2", fontsize = 19)

    ax.tick_params(axis = "both", direction = "out", labelsize = 16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for i in range(trajs):
        start_point = True
        for count, index in enumerate(paths[greedy_paths[i]]):
            if start_point:
                coord_pri = X_cluster[index,0:2]
                start_point = False
            else:
                coord_pri = coord_x.copy()
            # new coord
            coord_x = X_cluster[index,0:2]
            line = plt.plot([coord_pri[0],coord_x[0]],[coord_pri[1],coord_x[1]],'o-', linewidth = 4, color = colormap(i), alpha = 0.7)
            add_arrow(line[0], arrow_line = True, size=30, color=colormap(i))

            # line = plt.plot([coord_pri[0],coord_x[0]],[coord_pri[1],coord_x[1]],'o-',color = path_color(i))
            # add_arrow(line[0], arrow_line = True, size=15, color=path_color(i))
 
            # line = plt.plot([coord_pri[0],coord_x[0]],[coord_pri[1],coord_x[1]],'o-',color = path_color[:,i])
            # add_arrow(line[0], arrow_line = True, size=15, color=path_color[:,i])
        line[0].set_label('Path '+str(i))
        plt.legend(loc='upper right', prop={'size': 20}, frameon = False, ncol = 2)

    if basis != None:
        plt.xlabel(basis + str(1))
        plt.ylabel(basis + str(2))
    if title != None:
        plt.title(title)
    if save_as!= None:
        fig.savefig(save_as, bbox_inches = 'tight')

def traj_visual(adata, paths, greedy_paths, groups = None, trajs = 4, projection = '3d', figsize = (15,10), elev = 30, azim = 10, neighs = 15, save_as = None, cmap = None, axis = True):
    """\
    plot the trajectories using principal curve method, note that the principal curve may not be plotted very well with complex lineage structure.

    Parameters
    ----------
    adata
        data structure that store the gene expression data
    paths
        paths generated by path finding algorithm
    greedy_paths
        paths index ordered greedily
    groups
        list that store the cluster belonging of each cell. If None, generate from adata.obs['groups']
    trajs
        the number of trajectories plotted, note that trajs cannot be too much, or there may exists trajs contains too few cells, may even less than the neighs parameter
    projection
        either 3d plot or 2d plot
    figsize
        figure size, tuple
    elev, azim
        visualizing parameters for 3d plot
    neighs
        number of neighbors, for rwpt calculation
    save_as
        name of the saved file
    cmap
        If cmap = None, use random color
    axis
        Whether show axis or not

    Returns
    -------
    """
    if groups == None:
        groups = adata.obs['groups']
    
    fig = plt.figure(figsize = figsize)
        
    """
    if projection == '3d':
        from mpl_toolkits.mplot3d import Axes3D
        # %matplotlib inline    
        ax = plt.axes(projection='3d')
        ax.axis('off')

        if 'simulation_i' in adata.obs.columns:
            if adata.obs['simulation_i'].cat.categories.dtype != 'int64': 
                adata.obs['simulation_i'] = adata.obs['simulation_i'].astype('int64').astype('category')
            
            ori_trajs = np.max([x for x in adata.obs['simulation_i'].cat.categories])

            # get cmap
            if cmap != None:
                colormap = plt.cm.get_cmap(cmap, ori_trajs)
            else:
                colormap = lambda i: np.random.rand(3,trajs)[:,i]

            for i in range(ori_trajs):
                origin_traj = adata[adata.obs['simulation_i']== (i+1)].obsm['X_pca']
                ax.scatter(origin_traj[:,0],origin_traj[:,1], origin_traj[:,2],color = colormap(i),alpha = 0.7)

        else:
            ax.scatter(adata.obsm['X_pca'][:,0],adata.obsm['X_pca'][:,1],adata.obsm['X_pca'][:,2],color = 'red', alpha = 0.7)


        ax.legend(["ori_traj_"+str(i) for i in np.arange(1,ori_trajs+1)])
        for i in range(trajs):
            traj = np.array([])
            for count, index in enumerate(paths[greedy_paths[i]]):
                group_i = np.where(groups == index)[0]
                traj = np.append(traj, group_i)       
            traj = traj.astype('int')
            X_traj = adata.obsm['X_pca'][traj,:]
            results = pcurve.principalcurve(X_traj)
            projs = results['projections']
            orders = results['order']-1
            projs = projs[orders,:]

            # pseudo_order['reconst_'+str(i+1)] = adata[traj,:][orders,:].obs['sim_time']
            ax.plot(projs[:,0],projs[:,1], projs[:,2],'k-', linewidth = 3, alpha = 1)

        ax.view_init(elev = elev, azim = azim)
        plt.draw()
    """
    
    if projection == '2d':
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
            for count, index in enumerate(paths[greedy_paths[i]]):
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
    else:
        raise ValueError('projection can only be 2d or 3d')
    
    if save_as != None:
        fig.savefig(save_as, bbox_inches = 'tight')


def slingshot_visual(adata, results, basis = "umap", trajs = None, show_pcurve = True, figsize = (20,10), save_as = None):
    X_pca = adata.obsm['X_' + basis]
    if trajs == None:
        trajs = results['pseudotime'].shape[1]
    nrows = np.ceil(trajs).astype('int32')
    ncols = 1
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)

    kt = {}

    for i in range(trajs):
        if nrows == 1 or ncols == 1:
            if nrows == 1 and ncols == 1:
                ax = axs
            else:
                ax = axs[i]
        else:
            ax = axs[i%nrows, i//nrows]
        ax.scatter(X_pca[:,0],X_pca[:,1], color = 'gray', alpha = 0.1)

        if show_pcurve:
            ax.plot(results['curves'][i,:,0],results['curves'][i,:,1],color = 'black')

        pt_i = results['pseudotime'].iloc[:,i]

        pt_index = [x for x in pt_i.index if not np.isnan(pt_i[x])]
        ordering = [eval(x) for x in pt_i[pt_index].sort_values().index]

        pt_i = pt_i[pt_index]
        true_i = adata.obs['sim_time'].iloc[pt_index].values

        kt["traj_"+str(i)] = bmk.kendalltau(pt_i, true_i)
        X_traj = X_pca[ordering,:]

        pseudo_visual = ax.scatter(X_traj[:,0],X_traj[:,1],c = np.arange(X_traj.shape[0]), cmap=plt.get_cmap('gnuplot'),alpha = 0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("PC1", fontsize = 19)
        ax.set_ylabel("PC2", fontsize = 19)

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
    plot the bar plot that shows the purity of trajectories separation, run purity_count first to get bmk_belongings

    Parameters
    ----------
    bmk_belongings
        data frame that store the purity results
    separate
        Boolean, two different modes of plotting.
    trajs
        the number of trajectories plotted
    figsize
        figure size, tuple
    save_as
        saved file name

    Returns
    -------
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

def radius_hist(adata, groups = None, scaling = None):
    """\
    plot the distribution of the cluster radius 

    Parameters
    ----------
    adata
        data structure that store the gene expression data
    groups
        list that store the cluster belonging of each cell. If None, generate from adata.obs['groups']
    scaling
        resolution of the histogram

    Returns
    -------
    """    
    if groups == None:
        groups = adata.obs['groups']
    
    clusters = int(np.max(groups) + 1)

    # had to recalculate...
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline   
    pca = Pipeline(
    [('scaling', StandardScaler(with_mean=True, with_std=True)), 
    ('pca', PCA(n_components=30, svd_solver='arpack'))])
    X_spliced = np.log1p(adata.layers['spliced'].toarray())
    X_unspliced = np.log1p(adata.layers['unspliced'].toarray())
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
    fig, (ax1, ax2) = plt.subplots(1,2,figsize = (20,7))
    if scaling == None:
        _ = ax1.hist(radius,bins=int(np.max(radius)))
    else:
        _ = ax1.hist(radius,bins=scaling)
    ax1.set_title('Radius Histogram')
    max_group = np.argmax(radius)
    # ax = plt.axes(projection = '3d')
    ax2.scatter(X_concat_pca[:,0],X_concat_pca[:,1],color = 'red',alpha = 0.3)
    max_indices = np.where(groups == max_group)[0]
    ax2.scatter(X_concat_pca[max_indices,0],X_concat_pca[max_indices,1],color = 'blue',alpha = 0.3)
    ax2.set_title('largest cluster')
    return max_group
