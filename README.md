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
## 📘 Blogs

1. [A step-by-step guide for clustering images](https://towardsdatascience.com/a-step-by-step-guide-for-clustering-images-4b45f9906128)  
2. [Detection of Duplicate Images Using Image Hash Functions](https://towardsdatascience.com/detection-of-duplicate-images-using-image-hash-functions-4d9c53f04a75)  
3. [From Data to Clusters: When is Your Clustering Good Enough?](https://towardsdatascience.com/from-data-to-clusters-when-is-your-clustering-good-enough-5895440a978a)  
4. [From Clusters To Insights; The Next Step](https://towardsdatascience.com/from-clusters-to-insights-the-next-step-1c166814e0c6)

---

## 📚 Documentation

Full documentation is available at [erdogant.github.io/clusteval](https://erdogant.github.io/clusteval/), including examples and API references.

---

## ⚙️ Installation

It is advisable to use a virtual environment:

```bash
conda create -n env_clusteval python=3.12
conda activate env_clusteval
```

Install via PyPI:

```bash
pip install clusteval
```

To upgrade to the latest version:

```bash
pip install --upgrade clusteval
```

Import the library:

```python
from clusteval import clusteval
```

---

## 💡 Examples

A structured overview is available in the [documentation](https://erdogant.github.io/clusteval/pages/html/Examples.html).

<table>
<tr>
<td align="center">
  <a href="https://erdogant.github.io/clusteval/pages/html/Examples.html#cluster-evaluation">
    <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig1b_sil.png" width="300"/>
    <br>Silhouette Score
  </a>
</td>
<td align="center">
  <a href="https://erdogant.github.io/clusteval/pages/html/Plots.html#plot">
    <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig1a_sil.png" width="300"/>
    <br>Optimal Clusters
  </a>
</td>
</tr>
<tr>
<td align="center">
  <a href="https://erdogant.github.io/clusteval/pages/html/Plots.html#dendrogram">
    <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/dendrogram.png" width="300"/>
    <br>Dendrogram
  </a>
</td>
<td align="center">
  <a href="https://erdogant.github.io/clusteval/pages/html/Examples.html#dbindex-method">
    <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig2_dbindex.png" width="300"/>
    <br>Davies-Bouldin Index
  </a>
</td>
</tr>
<tr>
<td align="center">
  <a href="https://erdogant.github.io/clusteval/pages/html/Examples.html#derivative-method">
    <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig3_der.png" width="300"/>
    <br>Derivative Method
  </a>
</td>
<td align="center">
  <a href="https://erdogant.github.io/clusteval/pages/html/Examples.html#dbscan">
    <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig5_dbscan.png" width="300"/>
    <br>DBSCAN
  </a>
</td>
</tr>
<tr>
<td align="center">
  <a href="https://erdogant.github.io/clusteval/pages/html/Examples.html#hdbscan">
    <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig4a_hdbscan.png" width="300"/>
    <br>HDBSCAN A
  </a>
</td>
<td align="center">
  <a href="https://erdogant.github.io/clusteval/pages/html/Examples.html#hdbscan">
    <img src="https://github.com/erdogant/clusteval/blob/master/docs/figs/fig4b_hdbscan.png" width="300"/>
    <br>HDBSCAN B
  </a>
</td>
</tr>
</table>

---

## 📖 Citation

Please cite `clusteval` in your publications if it has been helpful in your research. Citation information is available at the top right of the [GitHub page](https://github.com/erdogant/clusteval).

---

## 🔍 Related Tools & Blogs

- Use **ARI** when clustering contains large equal-sized clusters  
- Use **AMI** for unbalanced clusters with small components  
- [Adjusted Rand Score — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)  
- [Adjusted for Chance Measures — scikit-learn](https://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html)  
- [imagededup GitHub repo](https://github.com/idealo/imagededup)  
- [Clustering images by visual similarity](https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34)  
- [Facebook DeepCluster](https://github.com/facebookresearch/deepcluster)  
- [PCA on Hyperspectral Data](https://towardsdatascience.com/pca-on-hyperspectral-data-99c9c5178385)  
- [Face Recognition with PCA](https://machinelearningmastery.com/face-recognition-using-principal-component-analysis/)

---

## ☕ Support

If you find this project useful, consider supporting me:

<a href="https://www.buymeacoffee.com/erdogant">
  <img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=erdogant&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff" />
</a>
