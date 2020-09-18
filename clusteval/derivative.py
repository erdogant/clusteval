# -----------------------------------------------
# Name        : derivative.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : MIT
# Respect the autor and leave this here
# -----------------------------------------------

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as linkage_scipy
import matplotlib.pyplot as plt


# %% Main
def fit(X, cluster='agglomerative', metric='euclidean', linkage='ward', min_clust=2, max_clust=25, Z=None, verbose=3):
    """ Determine optimal number of clusters using dbindex.

    Description
    -----------
    This function return the cluster labels for the optimal cutt-off based on the choosen hierarchical clustering method.

    Parameters
    ----------
    X : Numpy-array.
        The rows are the features and the colums are the samples.
    cluster : str, (default: 'agglomerative')
        Clustering method type for clustering.
            * 'agglomerative'
            * 'kmeans'
    metric : str, (default: 'euclidean').
        Distance measure for the clustering, such as 'euclidean','hamming', etc.
    linkage : str, (default: 'ward')
        Linkage type for the clustering.
        'ward','single',',complete','average','weighted','centroid','median'.
    min_clust : int, (default: 2)
        Number of clusters that is evaluated greater or equals to min_clust.
    max_clust : int, (default: 25)
        Number of clusters that is evaluated smaller or equals to max_clust.
    Z : Object, (default: None).
        This will speed-up computation if you readily have Z. e.g., Z=linkage(X, method='ward', metric='euclidean').
    verbose : int, optional (default: 3)
        Print message to screen [1-5]. The larger the number, the more information.

    Returns
    -------
    dict. with various keys. Note that the underneath keys can change based on the used methodtype.
    method: str
        Method name that is used for cluster evaluation.
    score: None
        Nothing in here but incuded for consistency
    labx: list
        Cluster labels.
    fig: list
        Relevant information to make the plot.

    Examples
    --------
    >>> # Import library
    >>> import clusteval.derivative as derivative
    >>> from sklearn.datasets import make_blobs
    >>> Generate demo data
    >>> X, labels_true = make_blobs(n_samples=750, centers=6, n_features=10)
    >>> # Fit with default parameters
    >>> results = derivative.fit(X)
    >>> # plot
    >>> derivative.plot(results)

    """
    Param = {}
    Param['verbose'] = verbose
    Param['cluster'] = cluster
    Param['metric'] = metric
    Param['linkage'] = linkage
    Param['min_clust'] = min_clust
    Param['max_clust'] = max_clust

    if verbose>=3: print('[clusteval] >Evaluate using derivatives.')

    if Param['cluster']=='kmeans':
        if verbose>=3: print('[clusteval] >Does not work with Kmeans! <return>')
        results = {}
        results['method']='derivative'
        results['labx'] = None
        results['score'] = None
        results['fig'] = {}
        results['fig']['last_rev'] = None
        results['fig']['acceleration_rev'] = None
        return results

    # Cluster hierarchical using on metric/linkage
    if Z is None:
        Z = linkage_scipy(X, method=Param['linkage'], metric=Param['metric'])

    # Make all possible cluster-cut-offs
    if Param['verbose']>=3: print('[clusteval] >Determining optimal clustering by derivatives..')

    # Run over all cluster cutoffs
    last = Z[-10:, 2]
    last_rev = last[::-1]

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]

    # Only focus on the min-max clusters
    acceleration_rev[:Param['min_clust']]=0
    acceleration_rev[Param['max_clust']:]=0
    last_rev[:Param['min_clust']]=0
    last_rev[Param['max_clust']:]=0

    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    if Param['verbose']>=3: print('[clusteval] >Clusters: %d' %k)

    # Now use the optimal cluster cut-off for the selection of clusters
    clustlabx = fcluster(Z, k, criterion='maxclust')

    # Convert to array
    clustlabx = np.array(clustlabx)

    # Store results
    results = {}
    results['method']='derivative'
    results['labx'] = clustlabx
    results['score'] = None
    results['fig'] = {}
    results['fig']['last_rev'] = last_rev
    results['fig']['acceleration_rev'] = acceleration_rev
    # Return
    return(results)


# %% Plot
def plot(results, figsize=(15,8), verbose=3):
    """Make plot for the gridsearch over the number of clusters.

    Parameters
    ----------
    results : dict.
        Dictionary that is the output of the .fit() function.
    figsize : tuple, (default: (15,8))
        Figure size, (heigh,width).
    verbose : int, optional (default: 3)
        Print message to screen [1-5]. The larger the number, the more information.

    Returns
    -------
    tuple, (fig, ax)
        Figure and axis of the figure.

    """
    idxs = np.arange(1, len(results['fig']['last_rev']) + 1)
    k = results['fig']['acceleration_rev'].argmax() + 2  # if idx 0 is the max of this we want 2 clusters

    # Make figure
    [fig, ax1] = plt.subplots(figsize=figsize)
    # Plot
    plt.plot(idxs, results['fig']['last_rev'])
    plt.plot(idxs[:-2] + 1, results['fig']['acceleration_rev'])

    # Plot optimal cut
    ax1.axvline(x=k, ymin=0, linewidth=2, color='r', linestyle="--")
    # Set fontsizes
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('xtick', labelsize=10)     # fontsize of the axes title
    plt.rc('ytick', labelsize=10)     # fontsize of the axes title
    plt.rc('font', size=10)
    # Set labels
    ax1.set_xticks(np.arange(0, len(idxs)))
    ax1.set_xlabel('#Clusters')
    ax1.set_ylabel('Score')
    ax1.set_title("Derivatives versus number of clusters")
    ax1.grid(color='grey', linestyle='--', linewidth=0.2)
    plt.show()
    # Return
    return(fig, ax1)
