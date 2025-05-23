import logging

from clusteval.clusteval import clusteval

from clusteval.clusteval import (
    dbindex,
    silhouette,
    derivative,
    dbscan,
    # hdbscan_custom,
    )

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '2.2.5'

# Setup root logger
_logger = logging.getLogger('clusteval')
_log_handler = logging.StreamHandler()
_fmt = '[{asctime}] [{name}] [{levelname}] {msg}'
_formatter = logging.Formatter(fmt=_fmt, style='{', datefmt='%d-%m-%Y %H:%M:%S')
_log_handler.setFormatter(_formatter)
_log_handler.setLevel(logging.DEBUG)
_logger.addHandler(_log_handler)
_logger.propagate = False

__doc__ = """
clusteval
=====================================================================

clusteval is a python package for unsupervised cluster validation.

Examples
--------
>>> # Import library
>>> from clusteval import clusteval
>>> # Initialize clusteval with default parameters
>>> ce = clusteval()
>>>
>>> # Generate random data
>>> from sklearn.datasets import make_blobs
>>> X, labels_true = make_blobs(n_samples=750, centers=4, n_features=2, cluster_std=0.5)
>>>
>>> # Fit best clusters
>>> results = ce.fit(X)
>>>
>>> # Make plot
>>> ce.plot()
>>>
>>> # Scatter plot
>>> ce.scatter(X)
>>>
>>> # Silhouette plot
>>> ce.plot_silhouette(X)
>>>
>>> # Dendrogram
>>> ce.dendrogram()


References
----------
* https://github.com/erdogant/clusteval

"""
