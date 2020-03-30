# clusteval

[![Python](https://img.shields.io/pypi/pyversions/clusteval)](https://img.shields.io/pypi/pyversions/clusteval)
[![PyPI Version](https://img.shields.io/pypi/v/clusteval)](https://pypi.org/project/clusteval/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/clusteval/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/clusteval/week)](https://pepy.tech/project/clusteval/week)
[![Donate](https://img.shields.io/badge/donate-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)

* clusteval is Python package for unsupervised cluster evaluation. Five methods are implemented that can be used to evalute clusterings; silhouette, dbindex, derivative, dbscan and hdbscan.

## Methods
```python
# X is your data
out = clusteval.fit(X)
clusteval.plot(out, X)
```

## Contents
- [Installation](#-installation)
- [Requirements](#-Requirements)
- [Quick Start](#-quick-start)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

## Installation
* Install clusteval from PyPI (recommended). clusteval is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* It is distributed under the MIT license.

## Requirements
* It is advisable to create a new environment. 
```python
conda create -n env_clusteval python=3.6
conda activate env_clusteval
pip install matplotlib numpy pandas tqdm seaborn hdbscan sklearn
```

## Quick Start
```
pip install clusteval
```

* Alternatively, install clusteval from the GitHub source:
```bash
git clone https://github.com/erdogant/clusteval.git
cd clusteval
python setup.py install
```  

## Import clusteval package
```python
import clusteval as clusteval
```

## Create example data set
```python
# Generate some random data
from sklearn.datasets import make_blobs
[X,_] = make_blobs(n_samples=750, centers=4, n_features=2, cluster_std=0.5)
```

## Cluster validation using Silhouette score
```python
# Determine the optimal number of clusters
out = clusteval.fit(X, method='silhouette')
fig = clusteval.plot(out, X)
```
<p align="center">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig1a_sil.png" width="600" />
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig1b_sil.png" width="600" />
</p>

## Cluster validation using davies-boulin index
```python
# Determine the optimal number of clusters
out = clusteval.fit(X, method='dbindex')
fig = clusteval.plot(out, X)
```
<p align="center">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig2_dbindex.png" width="600" />
</p>

## Cluster validation using derivative method
```python
# Determine the optimal number of clusters
out = clusteval.fit(X, method='derivative')
fig = clusteval.plot(out)
```
<p align="center">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig3_der.png" width="600" />
</p>

## Cluster validation using hdbscan
```python
# Determine the optimal number of clusters
out = clusteval.fit(X, method='hdbscan')
fig = clusteval.plot(out)
```
<p align="center">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig4a_hdbscan.png" width="600" />
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig4b_hdbscan.png" width="600" />
</p>

## Cluster validation using dbscan
```python
# Determine the optimal number of clusters
out = clusteval.fit(X, method='dbscan')
fig = clusteval.plot(out, X)
```
<p align="center">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig5_dbscan.png" width="600" />
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
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
https://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html#sphx-glr-auto-examples-cluster-plot-adjusted-for-chance-measures-py

## Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

## Contribute
* Contributions are welcome.

## Licence
See [LICENSE](LICENSE) for details.

### Donation
* This work is created and maintained in my free time. If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated.
