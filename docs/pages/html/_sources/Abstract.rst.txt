Background
#############

Clustering is an unsupervised machine learning algorithm where we aim to determine “natural” or “data-driven” groups in the data without using apriori knowledge about labels or categories. The challenge of using different unsupervised clustering methods is that it will result in different partitioning of the samples and thus different groupings since each method implicitly impose a structure on the data. Thus the question arises; What is a “good” clustering? Before we can use any clustering result, we need to evaluate strategies that can describe the **clustering tendency**, **Number of clusters** and the **clustering quality**.

Aim
#############

``clusteval`` is a python package that is developed to evaluate detected clusters and return the cluster labels that have most optimal **clustering tendency**, **Number of clusters** and **clustering quality**. Multiple evaluation strategies are implemented for the evaluation; **silhouette**, **dbindex**, and **derivative**, and four clustering methods can be used: **agglomerative**, **kmeans**, **dbscan** and **hdbscan**. The ``clusteval`` library then searches across the space of clusters and method-parameters to determine the optimal number of clusters given the input dataset.


Quickstart
################

A quick example how to learn a model on a given dataset.


.. code:: python

	# Load library
	from clusteval import clusteval

	# Initialize with default parameters
	ce = clusteval()

	# Fit data X
	results = ce.fit(X)
	
	# Plot
	ce.plot()
	ce.scatter(X)
	ce.dendrogram(X)



.. _schematic_overview:

.. figure:: ../figs/cluster.png


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>
