import statsmodels.api as sm
from scipy import stats
import numpy as np
import anndata
import matplotlib.pyplot as plt

def GAM_pt(pse_t, expr, smooth = 'BSplines', n_splines = 5, degree = 3, family = sm.families.NegativeBinomial()):
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
    n_splines
        number of splines function, correspond to the degree of freedom of the model
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
        spline = BSplines(pse_t, df = [n_splines], degree = [degree])
    elif smooth == 'CyclicCubicSplines':
        spline = CyclicCubicSplines(pse_t, df = [n_splines])

    exog, endog = sm.add_constant(pse_t),expr
    # calculate full model
    model_full = sm.GLMGam(endog = endog, exog = exog, smoother= spline, family=family)
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
        df_full_residual = expr.shape[0] - n_splines
        df_reduced_residual = expr.shape[0] - 1

        # likelihood of full model
        llf_full = res_full.llf
        # likelihood of reduced(null) model
        llf_reduced = res_full.llnull

        lrdf = (df_reduced_residual - df_full_residual)
        lrstat = -2*(llf_reduced - llf_full)
        lr_pvalue = stats.chi2.sf(lrstat, df=lrdf)
        return y_full, y_reduced, lr_pvalue
    

def de_analy(adata, pseudo_order, p_val_t = 0.05, verbose = False):
    """\
    Conduct differentially expressed gene analysis.

    Parameters
    ----------
    adata
        data structure that store the gene expression value
    pseudo_order
        the pseudo-time ordering obtained from the trajectory inference algorithm
    p_val_t
        the threshold of p-value
    verbose
        output the differentially expressed gene

    Returns
    -------
    de_genes
        dictionary that store the differentially expressed genes
    """ 
    de_genes = {}
    for reconst_i in pseudo_order.keys():
        de_genes[reconst_i] = []
        ordering = pseudo_order[reconst_i]
        adata_i =anndata.AnnData(X = adata[ordering,:].X, obs= adata[ordering,:].obs[[]], var= adata[ordering,:].var[[]])
        for idx, gene in enumerate(adata_i.var.index):
            gene_dynamic = np.squeeze(adata_i[:,gene].X.toarray())
            pse_t = np.arange(gene_dynamic.shape[0])[:,np.newaxis]
            # 10x dataset uses UMI count, de and dg are all created using 10x, and SMART-Seq use relative expression count, FKPM/TPM
            # With UMI count the distribution follows negative binomial, and the FKPM count follows the log-normal distribution, most use 10x
            gene_pred, gene_null, p_val = GAM_pt(pse_t, gene_dynamic, smooth='CyclicCubicSplines', n_splines = 4, degree = 3, family=sm.families.NegativeBinomial())
            # sm.families.Gaussian(link = sm.families.links.log())
            if p_val != None:
                if verbose:
                    print("gene: ", gene, ", pvalue = ", p_val)
                if p_val <= p_val_t:
                    de_genes[reconst_i].append({"gene": gene, "regression": gene_pred, "null": gene_null,"p_val": p_val})
        
        # sort according to the p_val
        de_genes[reconst_i] = sorted(de_genes[reconst_i], key=lambda x: x["p_val"],reverse=False)

    return de_genes

def de_plot(adata, pseudo_order, de_genes, figsize = (20,40), n_genes = 20, prefix = None):
    """\
    Plot differentially expressed gene.

    Parameters
    ----------
    adata
        data structure that store the gene expression value
    pseudo_order
        the pseudo-time ordering obtained from the trajectory inference algorithm
    de_genes
        dictionary that store the differentially expressed genes
    figsize
        figure size
    n_genes
        the number of genes to keep
    prefix
        the saving directory 
    """ 
    import os
    import errno
    # # turn off interactive mode for matplotlib
    # plt.ioff()

    ncols = 2
    nrows = np.ceil(n_genes/2).astype('int32')

    for reconst_i in de_genes.keys():
        # ordering of genes
        ordering = pseudo_order[reconst_i]
        adata_i =anndata.AnnData(X = adata[ordering,:].X, obs= adata[ordering,:].obs[[]], var= adata[ordering,:].var[[]])

        # create directory
        if prefix != None:
            directory = prefix + reconst_i + "/"
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
            gene_dynamic = np.squeeze(adata_i[:,gene['gene']].X.toarray())
            pse_t = np.arange(gene_dynamic.shape[0])[:,np.newaxis]

            gene_null = gene['null']
            gene_pred = gene['regression']

            axs[idx%nrows, idx//nrows].scatter(np.arange(gene_dynamic.shape[0]), gene_dynamic, c = colormap(idx), alpha = 0.7)
            axs[idx%nrows, idx//nrows].plot(pse_t, gene_pred, color = "black", alpha = 1)
            axs[idx%nrows, idx//nrows].plot(pse_t, gene_null, color = "red", alpha = 1)
            axs[idx%nrows, idx//nrows].set_title(gene['gene'])
        
        if prefix != None:
            fig.savefig(directory + "de.pdf", bbox_inches = 'tight')
    # store the gene_id result
    if prefix != None:
        import json
        gene_names_ordered = {}
        for reconst_i in de_genes.keys():
            gene_names_ordered[reconst_i] = [{"gene_id": gene["gene"], "p_val": gene["p_val"]} for gene in de_genes[reconst_i]]
        with open(prefix + "de_genes_ordered.json","w") as fp:
            json.dump(gene_names_ordered, fp)
                


def de_heatmap(adata, pseudo_order, de_genes, figsize = (20,10), n_genes = 20, prefix = "./real_results/pe/"):
    """\
    Heatmap of differentially expressed gene analysis.

    Parameters
    ----------
    adata
        data structure that store the gene expression value
    pseudo_order
        the pseudo-time ordering obtained from the trajectory inference algorithm
    de_genes
        dictionary that store the differentially expressed genes
    figsize
        figure size
    n_genes
        the number of genes to keep
    prefix
        the saving directory 

    Returns
    -------
    de_genes
        dictionary that store the differentially expressed genes
    """ 
    import seaborn as sns
    import pandas as pd
    save_as = "/gene_pt.pdf"
    for reconst_i in pseudo_order.keys():
        ordering = pseudo_order[reconst_i]
        adata_i = anndata.AnnData(X = adata[ordering,:].X, obs= adata[ordering,:].obs[[]], var= adata[ordering,:].var[[]])
        idices = [gene['gene'] for gene in de_genes[reconst_i][:n_genes]]
        adata_i = adata_i[:,idices]
        heatmap_data = pd.DataFrame(data = np.array([np.squeeze(adata_i[:,hv_gene].X.toarray()) for hv_gene in adata_i.var.index]), index = adata_i.var.index)
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot()
        sns.heatmap(heatmap_data, ax = ax)
        fig.savefig(prefix + reconst_i + save_as)