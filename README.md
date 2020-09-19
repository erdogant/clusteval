# clusteval

[![Python](https://img.shields.io/pypi/pyversions/clusteval)](https://img.shields.io/pypi/pyversions/clusteval)
[![PyPI Version](https://img.shields.io/pypi/v/clusteval)](https://pypi.org/project/clusteval/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/clusteval/blob/master/LICENSE)
[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)
[![Github Forks](https://img.shields.io/github/forks/erdogant/clusteval.svg)](https://github.com/erdogant/clusteval/network)
[![GitHub Open Issues](https://img.shields.io/github/issues/erdogant/clusteval.svg)](https://github.com/erdogant/clusteval/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Downloads](https://pepy.tech/badge/clusteval/month)](https://pepy.tech/project/clusteval/month)
[![Downloads](https://pepy.tech/badge/clusteval)](https://pepy.tech/project/clusteval)

* clusteval is Python package for unsupervised cluster evaluation.
Five methods are implemented that can be used to evalute clusterings; silhouette, dbindex, derivative, dbscan and hdbscan.


## Contents
- [Installation](#-installation)
- [Requirements](#-Requirements)
- [Quick Start](#-quick-start)
- [Contribute](#-contribute)
- [Citation](#-citation)

## Installation
* Install clusteval from PyPI (recommended). clusteval is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* It is distributed under the MIT license.

* A new environment can be created as following:

```python
conda create -n env_clusteval python=3.6
conda activate env_clusteval
```

```bash
pip install clusteval
```


* Beta version can be installed from the GitHub source:
```bash
git clone https://github.com/erdogant/clusteval
cd clusteval
pip install -U .
```  

## Import clusteval package
```python
from clusteval import clusteval
```

## Create example data set
```python
# Generate random data
from sklearn.datasets import make_blobs
X, labx_true = make_blobs(n_samples=750, centers=4, n_features=2, cluster_std=0.5)
```

## Cluster validation using Silhouette score
```python
# Determine the optimal number of clusters

ce = clusteval(method='silhouette')
ce.fit(X)
ce.plot()
ce.dendrogram()
ce.scatter(X)
```
<p align="center">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig1a_sil.png" width="600" />
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/dendrogram.png" width="600" />
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig1b_sil.png" width="600" />
</p>

## Cluster validation using davies-boulin index
```python
# Determine the optimal number of clusters
ce = clusteval(method='dbindex')
ce.fit(X)
ce.plot()
ce.scatter(X)
ce.dendrogram()
```

<p align="center">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig2_dbindex.png" width="600" />
</p>

## Cluster validation using derivative method
```python
# Determine the optimal number of clusters
ce = clusteval(method='derivative')
ce.fit(X)
ce.plot()
ce.scatter(X)
ce.dendrogram()
```

<p align="center">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig3_der.png" width="600" />
</p>


## Cluster validation using dbscan
```python
# Determine the optimal number of clusters using dbscan and silhoutte
ce = clusteval(cluster='dbscan')
ce.fit(X)
ce.plot()
ce.scatter(X)
ce.dendrogram()
```

<p align="center">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig5_dbscan.png" width="600" />
</p>

## Cluster validation using hdbscan
To run hdbscan, it requires an installation. This library is not included in the ``clusteval`` setup file because it frequently gives installation issues.
```bash
pip install hdbscan
```

```python
# Determine the optimal number of clusters
ce = clusteval(cluster='hdbscan')
ce.plot()
ce.scatter(X)
```
<p align="center">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig4a_hdbscan.png" width="600" />
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig4b_hdbscan.png" width="600" />
</p>



## Citation
Please cite clusteval in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2019clusteval,
  title={clusteval},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/clusteval}},
}
```
## TODO
* Use ARI when the ground truth clustering has large equal sized clusters
* Usa AMI when the ground truth clustering is unbalanced and there exist small clusters
* https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
* https://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html#sphx-glr-auto-examples-cluster-plot-adjusted-for-chance-measures-py


### Maintainer
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated :)
	Star it if you like it!
