#--------------------------------------------------------------------------
# Name        : dbindex.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
#--------------------------------------------------------------------------

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.cluster import KMeans, MiniBatchKMeans

# %% main
def fit(X, metric='euclidean', linkage='ward', minclusters=2, maxclusters=25, Z=None, savemem=False, verbose=3):
    """ Determine optimal number of clusters using dbindex.

    Description
    -----------
    This function return the cluster labels for the optimal cutt-off based on the choosen hierarchical clustering method.

    Parameters
    ----------
    X : Numpy-array.
        The rows are the features and the colums are the samples.
    metric : str, (default: 'euclidean').
        Distance measure for the clustering, such as 'euclidean','hamming', etc.
    linkage : str, (default: 'ward')
        Linkage type for the clustering.
        'ward','single',',complete','average','weighted','centroid','median'.
    minclusters : int, (default: 2)
        Minimum number of clusters (>=).
    maxclusters : int, (default: 25)
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
    Param['metric'] = metric
    Param['linkage'] = linkage
    Param['minclusters'] = minclusters
    Param['maxclusters'] = maxclusters
    Param['savemem'] = savemem

    # Savemem for kmeans
    if Param['metric']=='kmeans':
        if Param['savemem']==True:
            kmeansmodel=MiniBatchKMeans
            print('>Save memory enabled for kmeans.')
        else:
            kmeansmodel=KMeans

    # Cluster hierarchical using on metric/linkage
    if (Z is None) and (Param['metric']!='kmeans'):
        Z = scipy_linkage(X, method=Param['linkage'], metric=Param['metric'])
    
    # Make all possible cluster-cut-offs
    if Param['verbose']>=3: print('[dbindex] >Determining optimal clustering by Davies-Bouldin Index score..')

    # Setup storing parameters
    clustcutt = np.arange(Param['minclusters'],Param['maxclusters'])
    scores = np.zeros((len(clustcutt)))*np.nan
    dbclust = np.zeros((len(clustcutt)))*np.nan
    clustlabx = []

    # Run over all cluster cutoffs
    for i in tqdm(range(len(clustcutt))):
        # Cut the dendrogram for i clusters
        if Param['metric']=='kmeans':
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
    I2 = dbclust>=Param['minclusters']
    I3 = dbclust<=Param['maxclusters']
    I = I1 & I2 & I3

    # Get only clusters of interest
    scores = scores[I]
    dbclust = dbclust[I]
    clustlabx = clustlabx[I,:]
    clustcutt = clustcutt[I]
    idx = np.argmin(scores)

    # Store results
    results = {}
    results['method'] = 'dbindex'
    results['score'] = pd.DataFrame(np.array([dbclust,scores]).T, columns=['clusters','score'])
    results['score'].clusters = results['score'].clusters.astype(int)
    results['labx'] = clustlabx[idx,:]-1
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

    centroids = [np.mean(k, axis = 0) for k in cluster_k]
    variances = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]

    db = []
    for i in range(0,len(n_cluster)):
        for j in range(0,len(n_cluster)):
            if n_cluster[j] != n_cluster[i]:
                db.append( (variances[i] + variances[j]) / euclidean(centroids[i], centroids[j]) )

    outscore = np.max(db) / len(n_cluster)
    return(outscore)

# %% plot
def plot(results, figsize=(15,8)):
    idx = np.argmin(results['fig']['scores'])
    # Make figure
    [fig, ax1] = plt.subplots(figsize=figsize)
    # Plot
    ax1.plot(results['fig']['dbclust'], results['fig']['scores'], color='k')
    # Plot optimal cut
    ax1.axvline(x=results['fig']['clustcutt'][idx], ymin=0, ymax=results['fig']['dbclust'][idx], linewidth=2, color='r',linestyle="--")
    # Set fontsizes
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('xtick', labelsize=10)     # fontsize of the axes title
    plt.rc('ytick', labelsize=10)     # fontsize of the axes title
    plt.rc('font', size=10)
    # Set labels
    ax1.set_xticks(results['fig']['clustcutt'])
    ax1.set_xlabel('#Clusters')
    ax1.set_ylabel('Score')
    ax1.set_title("Davies Bouldin index versus number of clusters")
    ax1.grid(color='grey', linestyle='--', linewidth=0.2)
