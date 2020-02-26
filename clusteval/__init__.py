from clusteval.clusteval import (
    fit,
    plot,
)
import clusteval.hdbscan as hdbscan
import clusteval.dbscan as dbscan
import clusteval.silhouette as silhouette
import clusteval.derivative as derivative
import clusteval.dbindex as dbindex
import clusteval.dendrogram as dendrogram

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.1'


# module level doc-string
__doc__ = """
clusteval is a python package that provides various methods for unsupervised cluster validation.
=====================================================================

Description
-----------
    Probability density function fitting and hypothesis testing.
    Computes best fit to the input emperical distribution for 89 theoretical
    distributions using the Sum of Squared errors (SSE) estimates.

Example
-------
   clusteval provides methods for unsupervised cluster validation and 
   returns the cluster labels with the optimal cutt-off.
   
   >>> import clusteval

   >>> # Generate random data
   >>> from sklearn.datasets import make_blobs
   >>> [X, labels_true] = make_blobs(n_samples=750, centers=4, n_features=2, cluster_std=0.5)

   >>> model = clusteval.fit(X)

   >>> fig = clusteval.plot(out, X)


References
----------
    https://github.com/erdogant/clusteval

"""
