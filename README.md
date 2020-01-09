# clusteval

[![Python](https://img.shields.io/pypi/pyversions/clusteval)](https://img.shields.io/pypi/pyversions/clusteval)
[![PyPI Version](https://img.shields.io/pypi/v/clusteval)](https://pypi.org/project/clusteval/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/clusteval/blob/master/LICENSE)

* clusteval is Python package for unsupervised cluster evaluation. Five methods are implemented that can be used to evalute clusterings; silhouette, dbindex, derivative, dbscan and hdbscan.

## Methods
out = clusteval.fit(X, <optional>)
      clusteval.plot(out, X)

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
# Plot
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
# Plot
fig = clusteval.plot(out, X)
```
<p align="center">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig2_dbindex.png" width="600" />
</p>

## Cluster validation using derivative method
```python
# Determine the optimal number of clusters
out = clusteval.fit(X, method='derivative')
# Plot
fig = clusteval.plot(out)
```
<p align="center">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig3_der.png" width="600" />
</p>

## Cluster validation using hdbscan
```python
# Determine the optimal number of clusters
out = clusteval.fit(X, method='hdbscan')
# Plot
fig = clusteval.plot(out)
```
<p align="center">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig4a_hdbscan.png" width="600" />
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig4b_hdbscan.png" width="600" />
</p>

## Cluster validation using dbscan
```python
# Determine the optimal number of clusters
out = clusteval.fit(X, method='hdbscan')
# Plot
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
   
## Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

## Contribute
* Contributions are welcome.

## © Copyright
See [LICENSE](LICENSE) for details.
