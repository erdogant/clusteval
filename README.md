# clusteval

[![Python](https://img.shields.io/pypi/pyversions/clusteval)](https://img.shields.io/pypi/pyversions/clusteval)
[![PyPI Version](https://img.shields.io/pypi/v/clusteval)](https://pypi.org/project/clusteval/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/clusteval/blob/master/LICENSE)

* clusteval is Python package for unsupervised cluster evaluation.

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

## Example: Cluster validation
```python
from sklearn.datasets import make_blobs
[X,_] = make_blobs(n_samples=750, centers=4, n_features=2, cluster_std=0.5)
out = clusteval.fit(X, method='silhouette')
fig = clusteval.plot(out, X)
```
<p align="center">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig1.png" width="600" />
  
</p>

* Choosing various methodtypes and scoringtypes:
```python
model_hc_bic  = clusteval.structure_learning(df, methodtype='hc', scoretype='bic')
```

#### df looks like this:
```
     Cloudy  Sprinkler  Rain  Wet_Grass
0         0          1     0          1
1         1          1     1          1
2         1          0     1          1
3         0          0     1          1
4         1          0     1          1
..      ...        ...   ...        ...
995       0          0     0          0
996       1          0     0          0
997       0          0     1          0
998       1          1     0          1
999       1          0     1          1
```


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
