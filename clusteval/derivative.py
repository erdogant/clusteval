"""Derivative.

# Name        : derivative.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : MIT
# Respect the autor and leave this here

"""
from clusteval.utils import init_figure
import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as linkage_scipy
import matplotlib.pyplot as plt
from clusteval.utils import init_logger, set_logger, set_font_properties
# logger = init_logger()
import logging
logger = logging.getLogger(__name__)


# %% Main
def fit(X, cluster='agglomerative', metric='euclidean', linkage='ward', min_clust=2, max_clust=25, Z=None, verbose='info'):
    """ Determine optimal number of clusters using dbindex.

    Description
    -----------
    This function returns the cluster labels for the optimal cutt-off based on the choosen hierarchical clustering method.
    The derivative or inconsistence method is one of the defaults for the fcluster() function in scipy.
    It compares each cluster merge's height h to the average avg and normalizing it by the standard deviation std formed over the depth previous levels.

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
    evaluate: str
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

    set_logger(verbose=verbose)
    # Make all possible cluster-cut-offs
    logger.info('Evaluate clustering using [derivatives] method')

    if Param['cluster']=='kmeans':
        logger.info('Does not work with Kmeans! <return>')
        results = {}
        results['evaluate']='derivative'
        results['labx'] = None
        results['score'] = None
        results['fig'] = {}
        results['fig']['last_rev'] = None
        results['fig']['acceleration_rev'] = None
        return results

    # Cluster hierarchical using on metric/linkage
    if Z is None:
        Z = linkage_scipy(X, method=Param['linkage'], metric=Param['metric'])

    # Run over all cluster cutoffs
    last = Z[-max_clust:, 2]
    last_rev = last[::-1]

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]

    # Only focus on the min-max clusters
    acceleration_rev[:Param['min_clust']]=0
    acceleration_rev[Param['max_clust']:]=0
    last_rev[:Param['min_clust']]=0
    last_rev[Param['max_clust']:]=0

    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    logger.info('Clusters: %d' %k)

    # Now use the optimal cluster cut-off for the selection of clusters
    clustlabx = fcluster(Z, k, criterion='maxclust')

    # Convert to array
    clustlabx = np.array(clustlabx)

    # Store results
    results = {}
    results['evaluate']='derivative'
    results['labx'] = clustlabx
    results['score'] = None
    results['fig'] = {}
    results['fig']['last_rev'] = last_rev
    results['fig']['acceleration_rev'] = acceleration_rev
    # Return
    return results


# %% Plot
def plot(results,
         title='Derivative (Elbow method)',
         xlabel='Nr. Clusters',
         ylabel='Score',
         font_properties={},
         params_line={'color': 'k'},
         params_line2={'color': 'b'},
         params_vline={'color': 'r', 'linewidth': 2, 'linestyle': "--"},
         figsize=(15, 8),
         ax=None,
         visible=True,
         showfig=True,
         verbose=3):
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
    # Set font properties
    font_properties = set_font_properties(font_properties)
    params_line = {**{'color': 'k'}, **params_line}
    params_line2 = {**{'color': 'b'}, **params_line2}
    params_vline = {**{'color': 'r', 'linewidth': 2, 'linestyle': "--"}, **params_vline}
    if title is None: title='Derivative (Elbow method)'
    idxs = np.arange(1, len(results['fig']['last_rev']) + 1)
    k = results['fig']['acceleration_rev'].argmax() + 2  # if idx 0 is the max of this we want 2 clusters

    # Make figure
    fig, ax = init_figure(fig=None, ax=ax, dpi=100, figsize=figsize, visible=visible)
    # Plot
    ax.plot(idxs, results['fig']['last_rev'], **params_line)
    ax.plot(idxs[:-2] + 1, results['fig']['acceleration_rev'], **params_line2)

    # Plot optimal cut
    ax.axvline(x=k, ymin=0, **params_vline)
    # Set fontsizes
    # plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('xtick', labelsize=font_properties['size_x_axis'])  # fontsize of the axes title
    plt.rc('ytick', labelsize=font_properties['size_y_axis'])  # fontsize of the axes title
    # plt.rc('font', size=10)
    # Set labels
    ax.set_xticks(np.arange(0, len(idxs)))
    ax.set_xlabel(xlabel, fontsize=font_properties['size_x_axis'])
    ax.set_ylabel(ylabel, fontsize=font_properties['size_y_axis'])
    ax.set_title(title, fontsize=font_properties['size_title'])
    ax.grid(color='grey', linestyle='--', linewidth=0.2)
    if showfig: plt.show()
    # Return
    return fig, ax
