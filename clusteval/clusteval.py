"""clusteval provides methods for unsupervised cluster validation to return the cluster labels with the optimal cutt-off based on the choosen clustering method.

   TODO:
   ----
   * https://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html#sphx-glr-auto-examples-cluster-plot-adjusted-for-chance-measures-py
   * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
"""

# -----------------------------------
# Name        : clusteval.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# -----------------------------------

# %% Libraries
import clusteval.dbindex as dbindex
import clusteval.silhouette as silhouette
import clusteval.derivative as derivative
import clusteval.dbscan as dbscan
from scipy.cluster.hierarchy import linkage as scipy_linkage
# from cuml import DBSCAN

# %% Main function
def fit(X, method='silhouette', metric='euclidean', linkage='ward', minclusters=2, maxclusters=25, savemem=False, verbose=1):
    """Cluster validation.

    Parameters
    ----------
    X : Numpy array
        rows = features
        colums = samples

    method : str, optional
        Method type for cluster validation
        'silhouette' (default)
        'dbindex'
        'derivative'
        'hdbscan'
        'dbscan' (the default settings it the use of silhoutte)

    metric : str, optional (default: 'euclidean')
        Distance measure for the clustering
        'euclidean' (default, hierarchical)
        'hamming'
        'kmeans' (prototypes)

    linkage : str, optional (default: 'ward')
        Linkage type for the clustering
        'ward' (default)
        'single
        'complete'
        'average'
        'weighted'
        'centroid'
        'median'

    minclusters : int, optional (default: 2)
        Minimum or more number of clusters >=

    maxclusters : int, optional (default: 25)
        Maximum or more number of clusters =<

    savemem : bool, optional
        This works only for KMeans.
        [False]: No (default)
        [True]: Yes

    verbose : int [1-5], optional (default: 1)
        Print messages.

    Returns
    -------
    None.

    """
    assert 'array' in str(type(X)), 'Input data must be of type numpy array'
    out ={}
    Param = {}
    Param['method'] = method
    Param['verbose'] = verbose
    Param['metric'] = metric
    Param['linkage'] = linkage
    Param['minclusters'] = minclusters
    Param['maxclusters'] = maxclusters
    Param['savemem'] = savemem

    # Cluster hierarchical using on metric/linkage
    Z = []
    if Param['metric']!='kmeans':
        Z=scipy_linkage(X, method=Param['linkage'], metric=Param['metric'])

    # Choosing method
    if Param['method']=='silhouette':
        out=silhouette.fit(X, Z=Z, metric=Param['metric'], minclusters=Param['minclusters'], maxclusters=Param['maxclusters'], savemem=Param['savemem'], verbose=Param['verbose'])

    if Param['method']=='dbindex':
        out=dbindex.fit(X, Z=Z, metric=Param['metric'], minclusters=Param['minclusters'], maxclusters=Param['maxclusters'], savemem=Param['savemem'], verbose=Param['verbose'])

    if Param['method']=='derivative':
        out=derivative.fit(X, Z=Z, metric=Param['metric'], minclusters=Param['minclusters'], maxclusters=Param['maxclusters'], verbose=Param['verbose'])

    if Param['method']=='dbscan':
        out=dbscan.fit(X, eps=None, epsres=50, min_samples=0.01, metric=Param['metric'], norm=True, n_jobs=-1, minclusters=Param['minclusters'], maxclusters=Param['maxclusters'], verbose=Param['verbose'])

    if Param['method']=='hdbscan':
        import clusteval.hdbscan as hdbscan
        out=hdbscan.fit(X, min_samples=0.01, metric=Param['metric'], norm=True, n_jobs=-1, minclusters=Param['minclusters'], verbose=Param['verbose'])

    return(out)


# %% Plot
def plot(out, X=None, figsize=(15, 8)):
    """Make a plot.

    Parameters
    ----------
    out : dict
        The output of the fit() functoin.
    X : Data set, optional
        Some plots will be more extensive if the input data is also provided. The default is None.
    width : TYPE, optional
        DESCRIPTION. The default is 15.
    height : TYPE, optional
        DESCRIPTION. The default is 8.

    Returns
    -------
    None.

    """
    if out['methodtype']=='silhoutte':
        silhouette.plot(out, X=X, width=figsize[0], height=figsize[1])
    if out['methodtype']=='dbindex':
        dbindex.plot(out, width=figsize[0], height=figsize[1])
    if out['methodtype']=='derivative':
        derivative.plot(out, width=figsize[0], height=figsize[1])
    if out['methodtype']=='dbscan':
        dbscan.plot(out, X=X, width=figsize[0], height=figsize[1])
    if out['methodtype']=='hdbscan':
        import clusteval.hdbscan as hdbscan
        hdbscan.plot(out, width=figsize[0], height=figsize[1])
