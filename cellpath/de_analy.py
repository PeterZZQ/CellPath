import statsmodels.api as sm 
import statsmodels
from scipy import stats
import numpy as np
import anndata
import matplotlib.pyplot as plt
import scanpy as sc

def GAM_pt(pse_t, expr, smooth = 'BSplines', df = 5, degree = 3, family = sm.families.NegativeBinomial()):
    """\
    Fit a Generalized Additive Model with the exog to be the pseudo-time. The likelihood ratio test is performed 
    to test the significance of pseudo-time in affecting gene expression value

    Parameters
    ----------
    pse_t
        pseudo-time
    expr
        expression value
    smooth
        choose between BSplines and CyclicCubicSplines
    df
        degree of freedom of the model
    degree
        degree of the spline function
    family
        distribution family to choose, default is negative binomial.

    Returns
    -------
    y_full
        predict regressed value with full model
    y_reduced
        predict regressed value from null hypothesis
    lr_pvalue
        p-value
    """ 
    from statsmodels.gam.api import GLMGam, BSplines, CyclicCubicSplines

    if smooth == 'BSplines':
        spline = BSplines(pse_t, df = [df], degree = [degree])
    elif smooth == 'CyclicCubicSplines':
        spline = CyclicCubicSplines(pse_t, df = [df])

    exog, endog = sm.add_constant(pse_t),expr
    # calculate full model
    model_full = sm.GLMGam(endog = endog, exog = exog, smoother = spline, family = family)
    try:
        res_full = model_full.fit()
    except:
        # print("The gene expression is mostly zero")
        return None, None, None
    else:
        # default is exog
        y_full = res_full.predict()
        # reduced model
        y_reduced = res_full.null

        # number of samples - number of paras (res_full.df_resid)
        df_full_residual = expr.shape[0] - df
        df_reduced_residual = expr.shape[0] - 1

        # likelihood of full model
        llf_full = res_full.llf
        # likelihood of reduced(null) model
        llf_reduced = res_full.llnull

        lrdf = (df_reduced_residual - df_full_residual)
        lrstat = -2*(llf_reduced - llf_full)
        lr_pvalue = stats.chi2.sf(lrstat, df=lrdf)
        return y_full, y_reduced, lr_pvalue
    

def de_analy(cellpath_obj, p_val_t = 0.05, verbose = False, distri = "neg-binomial", fdr_correct = True):
    """\
    Conduct differentially expressed gene analysis.

    Parameters
    ----------
    cellpath_obj
        cellpath object
    p_val_t
        the threshold of p-value
    verbose
        output the differentially expressed gene
    distri
        distribution of gene expression: either "neg-binomial" or "log-normal"
    fdr_correct
        conduct fdr correction for multiple tests or not

    Returns
    -------
    de_genes
        dictionary that store the differentially expressed genes
    """ 

    pseudo_order = cellpath_obj.pseudo_order

    de_genes = {}
    for reconst_i in pseudo_order.columns:
        de_genes[reconst_i] = []
        sorted_pt = pseudo_order[reconst_i].dropna(axis = 0).sort_values()
        # ordering = [int(x.split("_")[1]) for x in sorted_pt.index]
        ordering = sorted_pt.index.values.squeeze()

        adata = cellpath_obj.adata[ordering,:]
        # filter out genes that are expressed in a small proportion of cells 
        sc.pp.filter_genes(adata, min_cells = int(0.05 * ordering.shape[0]))
        # spliced stores the count before log transform, but after library size normalization. 
        X_traj = adata.layers["spliced"].toarray()


        for idx, gene in enumerate(adata.var.index):
            gene_dynamic = np.squeeze(X_traj[:,idx])
            pse_t = np.arange(gene_dynamic.shape[0])[:,None]
            if distri == "neg-binomial":
                gene_pred, gene_null, p_val = GAM_pt(pse_t, gene_dynamic, smooth='BSplines', df = 4, degree = 3, family=sm.families.NegativeBinomial())
            
            elif distri == "log-normal":                
                gene_pred, gene_null, p_val = GAM_pt(pse_t, gene_dynamic, smooth='BSplines', df = 4, degree = 3, family=sm.families.Gaussian(link = sm.families.links.log()))
            
            else:
                raise ValueError("distribution can only be `neg-binomial` or `log-normal`")

            if p_val != None:
                if verbose:
                    print("gene: ", gene, ", pvalue = ", p_val)
                # if p_val <= p_val_t:
                de_genes[reconst_i].append({"gene": gene, "regression": gene_pred, "null": gene_null,"p_val": p_val})
        
        # sort according to the p_val
        de_genes[reconst_i] = sorted(de_genes[reconst_i], key=lambda x: x["p_val"],reverse=False)

        if fdr_correct:
            pvals = [x["p_val"] for x in de_genes[reconst_i]]
            is_de, pvals = statsmodels.stats.multitest.fdrcorrection(pvals, alpha=p_val_t, method='indep', is_sorted=True)
            
            # update p-value
            for gene_idx in range(len(de_genes[reconst_i])):
                de_genes[reconst_i][gene_idx]["p_val"] = pvals[gene_idx]
            
            # remove the non-de genes
            de_genes[reconst_i] = [x for i,x in enumerate(de_genes[reconst_i]) if is_de[i] == True]

    return de_genes



def de_plot(cellpath_obj, de_genes, figsize = (20,40), n_genes = 20, save_path = None):
    """\
    Plot differentially expressed gene.

    Parameters
    ----------
    cellpath_obj
        cellpath object
    de_genes
        dictionary that store the differentially expressed genes
    figsize
        figure size
    n_genes
        the number of genes to keep
    save_path
        the saving directory 
    """ 
    import os
    import errno
    # # turn off interactive mode for matplotlib
    # plt.ioff()

    ncols = 2
    nrows = np.ceil(n_genes/2).astype('int32')

    adata = cellpath_obj.adata
    pseudo_order = cellpath_obj.pseudo_order

    for reconst_i in de_genes.keys():
        # ordering of genes
        sorted_pt = pseudo_order[reconst_i].dropna(axis = 0).sort_values()
        # ordering = [int(x.split("_")[1]) for x in sorted_pt.index]
        ordering = sorted_pt.index.values.squeeze()
        adata_i = adata[ordering, :]

        # create directory
        if save_path != None:
            directory = save_path + reconst_i + "/"
            if not os.path.exists(os.path.dirname(directory)):
                try:
                    os.makedirs(os.path.dirname(directory))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
        # make plot
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
        colormap = plt.cm.get_cmap('tab20b', n_genes)
        for idx, gene in enumerate(de_genes[reconst_i][:n_genes]):
            # plot log transformed version
            gene_dynamic = np.squeeze(adata_i[:,gene["gene"]].layers["spliced"].toarray())
            pse_t = np.arange(gene_dynamic.shape[0])[:,np.newaxis]

            gene_null = gene['null']
            gene_pred = gene['regression']

            axs[idx%nrows, idx//nrows].scatter(np.arange(gene_dynamic.shape[0]), gene_dynamic, color = colormap(idx), alpha = 0.7)
            axs[idx%nrows, idx//nrows].plot(pse_t, gene_pred, color = "black", alpha = 1)
            axs[idx%nrows, idx//nrows].plot(pse_t, gene_null, color = "red", alpha = 1)
            axs[idx%nrows, idx//nrows].set_title(gene['gene'])
        
        if save_path != None:
            fig.savefig(directory + "de.pdf", bbox_inches = 'tight')
    # store the gene_id result
    if save_path != None:
        import json
        gene_names_ordered = {}
        for reconst_i in de_genes.keys():
            gene_names_ordered[reconst_i] = [{"gene_id": gene["gene"], "p_val": gene["p_val"]} for gene in de_genes[reconst_i]]
        with open(save_path + "de_genes_ordered.json","w") as fp:
            json.dump(gene_names_ordered, fp)
                


def de_heatmap(cellpath_obj, de_genes, figsize = (20,10), n_genes = 20, save_path = None):
    """\
    Heatmap of differentially expressed gene analysis.

    Parameters
    ----------
    cellpath_obj
        cellpath object
    de_genes
        dictionary that store the differentially expressed genes
    figsize
        figure size
    n_genes
        the number of genes to keep
    save_path
        the saving directory 
    """ 
    import seaborn as sns
    import pandas as pd
    save_as = "/gene_pt.pdf"
    adata = cellpath_obj.adata

    for reconst_i in de_genes.keys():
        sorted_pt = cellpath_obj.pseudo_order[reconst_i].dropna(axis = 0).sort_values()
        ordering = [int(x.split("_")[1]) for x in sorted_pt.index]
        adata_i = adata[ordering, :]
        idices = [gene['gene'] for gene in de_genes[reconst_i][:n_genes]]
        adata_i = adata_i[:,idices]
        heatmap_data = pd.DataFrame(data = np.array([np.squeeze(adata_i[:,hv_gene].X.toarray()) for hv_gene in adata_i.var.index]), index = adata_i.var.index)
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot()
        sns.heatmap(heatmap_data, ax = ax)
        if save_path != None:
            fig.savefig(save_path + reconst_i + save_as)