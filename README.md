# CellPath(Inference of multiple trajectories in single cell RNA-Seq data from RNA velocity)

CellPath v0.1.0

[Zhang's Lab](https://xiuweizhang.wordpress.com), Georgia Institute of Technology

Developed by Ziqi Zhang

## Description
CellPath is a single cell trajectory inference method that infers cell developmental trajectory using single-cell RNA Sequencing data and RNA-velocity data. The preprint is posted on bioarxiv: https://www.biorxiv.org/content/10.1101/2020.09.30.321125v2

## News
Include leiden algorithm for meta-cell clustering, which is more suitable for datasets with intricate trajectories. You can specify the clustering algorithm you use with either `flavor = "leiden"` or `flavor = "k-means"` in `cellpath.meta_cell_construction()` or `cellpath.all_in_one()`, please check the `run_cellpath.ipynb` for more details.

## Dependencies
```
Python >= 3.6.0

numpy >= 1.18.2

scipy >= 1.4.1

networkx>=2.5

pandas >= 1.1.5

scikit-learn >= 0.22.1

anndata >= 0.7.6

scvelo >= 0.2.3

seaborn >= 0.10.0

statsmodels >= 0.12.1 (optional, for differentially expressed gene analysis)

rpy2 >= 3.3.0 (optional, for principal curve only)
```

## Installation

### Install from pypi

```
pip install cellpath
```

### Install from github

Clone the repository with

```
git clone https://github.com/PeterZZQ/CellPaths.git
```

And run 

```
cd CellPaths/
pip install .
```

Uninstall using

```
pip uninstall cellpath
```

## Usage

`run_cellpath.ipynb` provide a short pipeline of running cellpaths using **cycle-tree** trajectory dataset in the paper.

* Initialize using adata with calculated velocity using scvelo
```
cellpath_obj = cp.CellPath(adata = adata, preprocess = True)
```

`preprocessing`: the velocity has been calculated and stored in adata or not, if False, the velocity will be calculated during initialization with scvelo

* Run cellpath all in one
```
cellpath_obj.all_in_one(num_metacells = num_metacells, n_neighs = 10, pruning = False, num_trajs = num_trajs, insertion = True, prop_insert = 0.50)
```

`num_metacells`: number of meta-cells in total

`n_neighs`: number of neighbors for each meta-cell

`pruning`: way to construct symmetric k-nn graph, prunning knn edges or including more edges

`num_trajs`: number of trajectories to output in the end

`insertion`: insert unassigned cells to trajectories or not

`prop_insert`: proportion of cells to be incorporated into the trajectories

`Pseudo-time and branching assignment result

```
cellpath_obj.pseudo_order
```
* Additional visualizations, please check `run_cellpath.ipynb` for details.

## Datasets
* You can access the real dataset that we used for the benchmarking through: https://www.dropbox.com/sh/6wcxj6x5szrp29v/AAB1FtWR18n41xoBn9tbGHKBa?dl=0. You can reproduce the result by putting the file into the root directory and run the notebook in `./Examples/`. 

    * `./Examples/CellPath_hema.ipynb`: mouse hematopoiesis dataset.
    * `./Examples/CellPath_dg.ipynb`: dentate-gyrus dataset.
    * `./Examples/CellPath_pe.ipynb`: pancreatic endocrinogenesis dataset.
    * `./Examples/CellPath_forebrain.ipynb`: forebrain dataset.


## Contents

* `CellPath/` contains the python code for the package
* `example_data/real/` contains four real datasets, used in the paper, dentate-gyrus dataset, pancreatic endocrinogenesis dataset and human forebrain dataset. Files in real_data folder can be downloaded from [dropbox](https://www.dropbox.com/sh/x0635h41ipcxqu0/AAAjfq5Nue7DxR5mCrQ4Gv6Ba?dl=0)
* `example_data/simulated/` contains simulated cycle-tree dataset


## Test in manuscript
Test script for the result in manuscript can be found with the [link](https://github.com/PeterZZQ/CellPath_test)