Generate data
####################

Install requried libraries

.. code:: bash

	pip install scatterd
	pip install sklearn


.. code:: python

	# Imports
	from sklearn.datasets import make_blobs
	
	# Generate random data
	X, _ = make_blobs(n_samples=750, centers=6, n_features=2, cluster_std=1)
	
	# Scatter samples
	scatterd(X[:,0], X[:,1], figsize=(15, 10));


.. |figP1| image:: ../figs/medium_clusters.png

.. table:: Six circular clusters
   :align: center

   +----------+
   | |figP1|  |
   +----------+


Plot
######################################


.. code:: python

	# Import
	from clusteval import clusteval

	# Silhouette cluster evaluation.
	ce = clusteval(evaluate='silhouette')

	# In case of using dbindex, it is best to clip the maximum number of clusters to avoid finding local minima.
	ce = clusteval(evaluate='dbindex', max_clust=10)

	# Derivative method.
	ce = clusteval(evaluate='derivative')

	# DBscan method.
	ce = clusteval(cluster='dbscan')

	# Fit
	ce.fit(X)

	# Plot
	ce.plot()


.. |figP2| image:: ../figs/medium_clusters_sil.png
.. |figP3| image:: ../figs/medium_clusters_dbindex.png
.. |figP4| image:: ../figs/medium_clusters_der.png
.. |figP5| image:: ../figs/medium_clusters_dbscan.png


.. table:: Method vs. number of clusters
   :align: center

   +----------+----------+
   | |figP2|  | |figP3|  |
   +----------+----------+
   | |figP4|  | |figP5|  |
   +----------+----------+

Scatterplot
################################################

The scatter plots for the methods **silhouette** and **dbindex** were identical. Small differences were seen for the **derivative** method.
Only the **dbscan** method was not able to identify the six clusters but detected five instead.

.. code:: python

	# Plot
	ce.scatter(X)


.. |figP6| image:: ../figs/medium_clusters_sil_scatter.png
.. |figP7| image:: ../figs/medium_clusters_sil_scatter.png
.. |figP8| image:: ../figs/medium_clusters_der_scatter.png
.. |figP9| image:: ../figs/medium_clusters_dbscan_scatter.png


.. table:: Method vs. number of clusters
   :align: center

   +----------+----------+
   | |figP6|  | |figP7|  |
   +----------+----------+
   | |figP8|  | |figP9|  |
   +----------+----------+

Dendrogram
#################

Hierarchical tree plot
***************************

To furter investigate the clustering results, a dendrogram can be created.

.. code:: python

	ce.dendrogram()

.. |figP10| image:: ../figs/medium_clusters_sil_dendrogram.png


.. table:: 
   :align: center

   +----------+
   | |figP10| |
   +----------+

Change the cut threshold
***************************

The dendrogram function can now also be used to create differents cuts in the hierarchical clustering and retrieve the associated cluster labels. Let's cut the tree at level 60

.. code:: python
	
	# Plot the dendrogram and make the cut at distance height 60
	y = ce.dendrogram(max_d=60)

	# Cluster labels for this particular cut
	print(y['labx'])


.. |figP11| image:: ../figs/medium_clusters_sil_dendrogram_60.png


.. table:: 
   :align: center

   +----------+
   | |figP11| |
   +----------+


Orientation
***********************

Change various parameters, such as orientation, leaf rotation, and the font size.

.. code:: python
	
	# Plot the dendrogram
	ce.dendrogram(orientation='left', leaf_rotation=180, leaf_font_size=8, figsize=(25,30))


.. |figP12| image:: ../figs/medium_clusters_sil_dendrogram_orientation.png


.. table:: 
   :align: center

   +----------+
   | |figP12| |
   +----------+




.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>
