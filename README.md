# clusteval
<p align="center">
  <a href="https://erdogant.github.io/clusteval">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/logo_large_2.png" width="300" />
  </a>
</p>

[![Python](https://img.shields.io/pypi/pyversions/clusteval)](https://img.shields.io/pypi/pyversions/clusteval)
[![PyPI Version](https://img.shields.io/pypi/v/clusteval)](https://pypi.org/project/clusteval/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/clusteval/blob/master/LICENSE)
[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)
[![Github Forks](https://img.shields.io/github/forks/erdogant/clusteval.svg)](https://github.com/erdogant/clusteval/network)
[![GitHub Open Issues](https://img.shields.io/github/issues/erdogant/clusteval.svg)](https://github.com/erdogant/clusteval/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Downloads](https://pepy.tech/badge/clusteval/month)](https://pepy.tech/project/clusteval)
[![Downloads](https://pepy.tech/badge/clusteval)](https://pepy.tech/project/clusteval)
[![DOI](https://zenodo.org/badge/232915924.svg)](https://zenodo.org/badge/latestdoi/232915924)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/clusteval/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://erdogant.github.io/clusteval/pages/html/Documentation.html#colab-notebook)
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

``clusteval`` is a python package that is developed to evaluate detected clusters and return the cluster labels that have most optimal **clustering tendency**, **Number of clusters** and **clustering quality**. Multiple evaluation strategies are implemented for the evaluation; **silhouette**, **dbindex**, and **derivative**, and four clustering methods can be used: **agglomerative**, **kmeans**, **dbscan** and **hdbscan**.


# 
**⭐️ Star this repo if you like it ⭐️**
# 

### Blogs

Read the blog to get a structured overview how you can use ``clusteval``.
* [Medium Blog: A step-by-step guide for clustering images](https://towardsdatascience.com/a-step-by-step-guide-for-clustering-images-4b45f9906128)

In case you want to detect identical images, you can also use hash functionalities.
* [Medium Blog: Detection of Duplicate Images Using Image Hash Functions](https://towardsdatascience.com/detection-of-duplicate-images-using-image-hash-functions-4d9c53f04a75)

# 


### [Documentation pages](https://erdogant.github.io/clusteval/)

On the [documentation pages](https://erdogant.github.io/clusteval/) you can find detailed information about the working of the ``clusteval`` with many examples. 

# 

### Installation

##### It is advisable to create a new environment (e.g. with Conda). 
```bash
conda create -n env_clusteval python=3.8
conda activate clusteval
```

##### Install from PyPI
```bash
pip install clusteval
```

##### Import library
```python
from clusteval import clusteval
```

<hr>

### Examples
A structured overview of all examples are now available on the [documentation pages](https://erdogant.github.io/clusteval/).

<hr>


* [Example: Cluster validation using Silhouette score](https://erdogant.github.io/clusteval/pages/html/Examples.html#cluster-evaluation)

<p align="left">
  <a href="https://erdogant.github.io/clusteval/pages/html/Examples.html#cluster-evaluation">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig1b_sil.png" width="600" />
  </a>
</p>


#

* [Example: Determine the optimal number of clusters](https://erdogant.github.io/clusteval/pages/html/Plots.html#plot)

<p align="left">
  <a href="https://erdogant.github.io/clusteval/pages/html/Plots.html#plot">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig1a_sil.png" width="600" />
  </a>
</p>

#

* [Example: Plot the dendrogram](https://erdogant.github.io/clusteval/pages/html/Plots.html#dendrogram)

<p align="left">
  <a href="https://erdogant.github.io/clusteval/pages/html/Plots.html#dendrogram">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/dendrogram.png" width="600" />
  </a>
</p>

#

* [Example: Cluster validation using davies-boulin index](https://erdogant.github.io/clusteval/pages/html/Examples.html#dbindex-method)

<p align="left">
  <a href="https://erdogant.github.io/clusteval/pages/html/Examples.html#dbindex-method">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/dendrogram.png" width="600" />
  </a>
</p>

#

* [Example: Cluster validation using davies-boulin index](https://erdogant.github.io/clusteval/pages/html/Examples.html#dbindex-method)

<p align="left">
  <a href="https://erdogant.github.io/clusteval/pages/html/Examples.html#dbindex-method">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig2_dbindex.png" width="600" />
  </a>
</p>

#

* [Example: Cluster validation using derivative evaluation method](https://erdogant.github.io/clusteval/pages/html/Examples.html#derivative-method)

<p align="left">
  <a href="https://erdogant.github.io/clusteval/pages/html/Examples.html#derivative-method">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig3_der.png" width="600" />
  </a>
</p>

#


* [Example: Cluster validation using dbscan](https://erdogant.github.io/clusteval/pages/html/Examples.html#dbscan)

<p align="left">
  <a href="https://erdogant.github.io/clusteval/pages/html/Examples.html#dbscan">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig5_dbscan.png" width="600" />
  </a>
</p>

#

* [Example: Cluster validation using hdbscan](https://erdogant.github.io/clusteval/pages/html/Examples.html#hdbscan)

<p align="left">
  <a href="https://erdogant.github.io/clusteval/pages/html/Examples.html#hdbscan">
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig4a_hdbscan.png" width="600" />
  <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig4b_hdbscan.png" width="600" />
  </a>
</p>






## Citation
Please cite clusteval in your publications if this is useful for your research (see right top for citation).

## Other interesting techniques/blogs
* Use ARI when the ground truth clustering has large equal sized clusters
* Usa AMI when the ground truth clustering is unbalanced and there exist small clusters
* https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
* https://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html#sphx-glr-auto-examples-cluster-plot-adjusted-for-chance-measures-py
* https://github.com/idealo/imagededup
* https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34
* https://github.com/facebookresearch/deepcluster
* https://towardsdatascience.com/pca-on-hyperspectral-data-99c9c5178385
* https://machinelearningmastery.com/face-recognition-using-principal-component-analysis/

### Maintainer
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated :)
	Star it if you like it!
