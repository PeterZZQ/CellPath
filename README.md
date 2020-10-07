# CellPath(Inference of multiple trajectories in single cell RNA-Seq data from RNA velocity)

CellPath v0.1.0

[Zhang's Lab](https://xiuweizhang.wordpress.com), Georgia Institute of Technology

Developed by Ziqi Zhang

## Description
CellPath is a single cell trajectory inference method that infers cell developmental trajectory using single-cell RNA Sequencing data and RNA-velocity data. The preprint is posted on bioarxiv: https://www.biorxiv.org/content/10.1101/2020.09.30.321125v1?rss=1

## Dependencies
```
Python >= 3.6.0

numpy >= 1.18.2

scipy >= 1.4.1

pandas >= 1.0.3

sklearn >= 0.22.1

anndata >= 0.7.1

scvelo >= 0.2.0

statsmodels >= 0.11.1

seaborn >= 0.10.0

rpy2 >= 3.3.0
```

## Installation
Clone the repository with

```
git clone https://github.com/PeterZZQ/CellPaths.git
```

And run 

```
pip install .
```

Uninstall using

```
pip uninstall cellpath
```

## Usage

`run_cellpath.ipynb` provide a short pipeline of running cellpaths using **cycle-tree** trajectory dataset in the paper.

## Contents

* `CellPath/` contains the python code for the package
* `real_data/` contains three real datasets, used in the paper, dentate-gyrus dataset, pancreatic endocrinogenesis dataset and human forebrain dataset. Files in real_data folder can be downloaded from [dropbox](https://www.dropbox.com/sh/nix4wnoiwda5id5/AACTxvGTQ82UzwMJs2IWSriKa?dl=0)
* `sim_data/` contains four simulated datasets, used in the paper, with **trifurcating**, **double-batches**, **cycle-tree** and **multiple-cycles** structure. 
    * **Trifurcating** and **double-batches** datasets are generated using [`dyngen`](https://github.com/dynverse/dyngen) 
    * **cycle-tree** and **multiple-cycles** datasets are generated using `symsim`.