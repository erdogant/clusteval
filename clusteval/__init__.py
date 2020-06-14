from clusteval.clusteval import clusteval

# import clusteval.dbindex as dbindex
# import clusteval.silhouette as silhouette
# import clusteval.derivative as derivative
# import clusteval.dbscan as dbscan
# import clusteval.dendrogram
# import clusteval.hdbscan

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.0.0'


__doc__ = """
clusteval
=====================================================================

Description
-----------
clusteval is a python package for unsupervised cluster validation using five popular methods, i.e. silhouette, dbindex, derivative, dbscan and hdbscan.
The metric and linkage types can choosen independently per validation method.

Examples
--------
clusteval provides methods for unsupervised cluster validation and returns the cluster labels with the optimal cutt-off.

>>> # Import library
>>> from clusteval import clusteval
>>> # Initialize clusteval with default parameters
>>> ce = clusteval()
>>> # Generate random data
>>> from sklearn.datasets import make_blobs
>>> X, labels_true = make_blobs(n_samples=750, centers=4, n_features=2, cluster_std=0.5)
>>> # Fit best clusters
>>> results = ce.fit(X)
>>> # Make plot
>>> ce.plot()
>>> # Make scatter plot with samples
>>> ce.scatter(X)

References
----------
* https://github.com/erdogant/clusteval

"""
