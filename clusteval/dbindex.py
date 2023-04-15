# -----------------------------------------------
# Name        : dbindex.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : MIT
# Respect the autor and leave this here
# -----------------------------------------------

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.cluster import KMeans, MiniBatchKMeans
from clusteval.utils import init_logger, set_logger, disable_tqdm, set_font_properties
logger = init_logger()


# %% main
def fit(X, cluster='agglomerative', metric='euclidean', linkage='ward', min_clust=2, max_clust=25, Z=None, savemem=False, verbose='info'):
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
        Minimum number of clusters (>=).
    max_clust : int, (default: 25)
        Maximum number of clusters (<=).
    Z : Object, (default: None).
        This will speed-up computation if you readily have Z. e.g., Z=linkage(X, method='ward', metric='euclidean').
    savemem : bool, (default: False)
        Save memmory when working with large datasets. Note that htis option only in case of KMeans.
    verbose : int, optional (default: 3)
        Print message to screen [1-5]. The larger the number, the more information.

    Returns
    -------
    dict. with various keys. Note that the underneath keys can change based on the used methodtype.
    method: str
        Method name that is used for cluster evaluation.
    score: pd.DataFrame()
        The scoring values per clusters.
    labx: list
        Cluster labels.
    fig: list
        Relevant information to make the plot.

    Examples
    --------
    >>> # Import library
    >>> import clusteval.dbindex as dbindex
    >>> from sklearn.datasets import make_blobs
    >>> Generate demo data
    >>> X, labels_true = make_blobs(n_samples=750, centers=6, n_features=10)
    >>> # Fit with default parameters
    >>> results = dbindex.fit(X)
    >>> # plot
    >>> dbindex.plot(results)
    """
    # Make dictionary to store Parameters
    Param = {}
    Param['verbose'] = verbose
    Param['cluster'] = cluster
    Param['metric'] = metric
    Param['linkage'] = linkage
    Param['min_clust'] = min_clust
    Param['max_clust'] = max_clust
    Param['savemem'] = savemem
    set_logger(verbose=verbose)
    logger.info('Evaluate using dbindex.')

    # Savemem for kmeans
    if Param['cluster']=='kmeans':
        if Param['savemem']:
            kmeansmodel=MiniBatchKMeans
            logger.info('Save memory enabled for kmeans.')
        else:
            kmeansmodel=KMeans

    # Cluster hierarchical using on metric/linkage
    if (Z is None) and (Param['cluster']!='kmeans'):
        Z = scipy_linkage(X, method=Param['linkage'], metric=Param['metric'])

    # Setup storing parameters
    clustcutt = np.arange(Param['min_clust'], Param['max_clust'])
    scores = np.zeros((len(clustcutt))) * np.nan
    dbclust = np.zeros((len(clustcutt))) * np.nan
    clustlabx = []

    # Run over all cluster cutoffs
    for i in tqdm(range(len(clustcutt)), disable=disable_tqdm(), desc='[clusteval] >INFO'):
        # Cut the dendrogram for i clusters
        if Param['cluster']=='kmeans':
            labx=kmeansmodel(n_clusters=clustcutt[i], verbose=0).fit(X).labels_
        else:
            labx = fcluster(Z, clustcutt[i], criterion='maxclust')

        # Store labx for cluster-cut
        clustlabx.append(labx)
        # Store number of unique clusters
        dbclust[i]=len(np.unique(labx))
        # Compute silhouette (can only be done if more then 1 cluster)
        if dbclust[i]>1:
            scores[i]=_dbindex_score(X, labx)

    # Convert to array
    clustlabx = np.array(clustlabx)

    # Store only if agrees to restriction of input clusters number
    I1 = np.isnan(scores)==False
    I2 = dbclust>=Param['min_clust']
    I3 = dbclust<=Param['max_clust']
    Iloc = I1 & I2 & I3

    # Get only clusters of interest
    if sum(Iloc)>0:
        scores = scores[Iloc]
        dbclust = dbclust[Iloc]
        clustlabx = clustlabx[Iloc, :]
        clustcutt = clustcutt[Iloc]
        idx = np.argmin(scores)
        clustlabx = clustlabx[idx, :] - 1
    else:
        logger.info('No clusters detected.')
        if len(clustlabx.shape)>1:
            clustlabx = np.zeros(clustlabx.shape[1]).astype(int)
        else:
            clustlabx = [0]

    # Store results
    results = {}
    results['evaluate'] = 'dbindex'
    results['score'] = pd.DataFrame(np.array([dbclust, scores]).T, columns=['clusters', 'score'])
    results['score'].clusters = results['score'].clusters.astype(int)
    results['labx'] = clustlabx
    results['fig'] = {}
    results['fig']['dbclust'] = dbclust
    results['fig']['scores'] = scores
    results['fig']['clustcutt'] = clustcutt

    # Return
    return(results)


# %% Compute DB-score
def _dbindex_score(X, labels):
    n_cluster = np.unique(labels)
    cluster_k=[]
    for k in range(0, len(n_cluster)):
        cluster_k.append(X[labels==n_cluster[k]])

    centroids = [np.mean(k, axis=0) for k in cluster_k]
    variances = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]

    db = []
    for i in range(0, len(n_cluster)):
        for j in range(0, len(n_cluster)):
            if n_cluster[j] != n_cluster[i]:
                db.append((variances[i] + variances[j]) / euclidean(centroids[i], centroids[j]))

    outscore = np.max(db) / len(n_cluster)
    return(outscore)


# %% plot
def plot(results, title='Davies Bouldin index', xlabel='Nr. Clusters', ylabel='Score', font_properties={}, figsize=(15, 8), ax=None, showfig=True):
    """Make plot for the gridsearch over the number of clusters.

    Parameters
    ----------
    results : dict.
        Dictionary that is the output of the .fit() function.
    figsize : tuple, (default: (15,8))
        Figure size, (heigh,width).

    Returns
    -------
    tuple, (fig, ax)
        Figure and axis of the figure.

    """
    # Set font properties
    font_properties = set_font_properties(font_properties)
    idx = np.argmin(results['fig']['scores'])
    fig=None
    # Make figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
    # Plot
    ax.plot(results['fig']['dbclust'], results['fig']['scores'], color='k')
    # Plot optimal cut
    ax.axvline(x=results['fig']['clustcutt'][idx], ymin=0, ymax=results['fig']['dbclust'][idx], linewidth=2, color='r', linestyle="--")
    # Set fontsizes
    ax.tick_params(axis='x', labelsize=font_properties['size_x_axis'])
    ax.tick_params(axis='y', labelsize=font_properties['size_y_axis'])
    # Set labels
    ax.set_xticks(results['fig']['clustcutt'])
    ax.set_xlabel(xlabel, fontsize=font_properties['size_x_axis'])
    ax.set_ylabel(ylabel, fontsize=font_properties['size_y_axis'])
    ax.set_title(title, fontsize=font_properties['size_title'])
    ax.grid(color='grey', linestyle='--', linewidth=0.2)
    if showfig: plt.show()
    # Return
    return(fig, ax)
