# -----------------------------------------------------
# Name        : silhouette.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# ------------------------------------------------------

# %% Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from clusteval.silhouette_plot import silhouette_plot
import matplotlib.pyplot as plt

# %% Main
def fit(X, metric='euclidean', linkage='ward', minclusters=2, maxclusters=25, Z=[], savemem=False, verbose=3):
    """This function return the cluster labels for the optimal cutt-off based on the choosen hierarchical clustering method.

    Parameters
    ----------
    X : Numpy-array,
        Where rows is features and colums is samples.
    metric : str, (default: 'euclidean').
        Distance measure for the clustering. Options are: 'euclidean','kmeans'.
    linkage : str, (default: 'ward')
        Linkage type for the clustering.
        'ward','single',',complete','average','weighted','centroid','median'.
    minclusters : int, (default: 2)
        Minimum number of clusters >=.
    maxclusters : int, (default: 25)
        Maximum number of clusters <=.
    savemem : bool, (default: False)
        Save memmory when working with large datasets. Note that htis option only in case of KMeans.
    verbose : int, optional (default: 3)
        Print message to screen [1-5]. The larger the number, the more information is returned.
    Z : array-like, (default: None).
        output of the linkage.

    Returns
    -------
    dict. with various keys. Note that the underneath keys can change based on the used methodtype.
    method: str
        Method name that is used for cluster evaluation.
    score: pd.DataFrame()
        The scoring values per clusters. The methods [silhouette, dbindex] provide this information.
    labx: list
        Cluster labels.
    fig: list
        Relevant information to make the plot.

    Examples
    --------
    >>> import clusteval.silhouette as silhouette
    >>> from sklearn.datasets.samples_generator import make_blobs
    >>> X, labels_true = make_blobs(n_samples=750, centers=5, n_features=10)
    >>> results = silhouette.fit(X)
    >>> results = silhouette.fit(X, metric='kmeans', savemem=True)
    >>> silhouette.plot(results, X)



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
        if Param['savemem']:
            kmeansmodel = MiniBatchKMeans
            if Param['verbose']>=3: print('[clusteval] >Save memory enabled for kmeans with method silhouette.')
        else:
            kmeansmodel = KMeans

    # Cluster hierarchical using on metric/linkage
    if len(Z)==0 and Param['metric']!='kmeans':
        from scipy.cluster.hierarchy import linkage
        Z = linkage(X, method=Param['linkage'], metric=Param['metric'])

    # Make all possible cluster-cut-offs
    if Param['verbose']>=3: print('[clusteval] >Determining optimal [%s] clustering by silhouette score..' %(Param['metric']))

    # Setup storing parameters
    clustcutt = np.arange(Param['minclusters'],Param['maxclusters'])
    silscores = np.zeros((len(clustcutt)))*np.nan
    sillclust = np.zeros((len(clustcutt)))*np.nan
    clustlabx = []

    # Run over all cluster cutoffs
    for i in tqdm(range(len(clustcutt))):
        # Cut the dendrogram for i clusters
        if Param['metric']=='kmeans':
            labx = kmeansmodel(n_clusters=clustcutt[i], verbose=0).fit(X).labels_
        else:
            labx = fcluster(Z, clustcutt[i], criterion='maxclust')

        # Store labx for cluster-cut
        clustlabx.append(labx)
        # Store number of unique clusters
        sillclust[i] = len(np.unique(labx))
        # Compute silhouette (can only be done if more then 1 cluster)
        if sillclust[i]>1:
            silscores[i] = silhouette_score(X, labx)

    # Convert to array
    clustlabx = np.array(clustlabx)
    
    # Store only if agrees to restriction of input clusters number
    I1 = np.isnan(silscores)==False
    I2 = sillclust>=Param['minclusters']
    I3 = sillclust<=Param['maxclusters']
    I  = I1 & I2 & I3

    # Get only clusters of interest
    silscores = silscores[I]
    sillclust = sillclust[I]
    clustlabx = clustlabx[I,:]
    clustcutt = clustcutt[I]
    idx = np.argmax(silscores)
    
    # Store results
    out = {}
    out['method']='silhouette'
    out['score'] = pd.DataFrame(np.array([sillclust,silscores]).T, columns=['clusters','score'])
    out['score']['clusters'] = out['score']['clusters'].astype(int)
    out['labx']  = clustlabx[idx,:]-1
    out['fig'] = {}
    out['fig']['silscores'] = silscores
    out['fig']['sillclust'] = sillclust
    out['fig']['clustcutt'] = clustcutt
    
    # Return
    return(out)

# %% plot
def plot(out, X=None, figsize=(15,8)):
    idx = np.argmax(out['fig']['silscores'])
    # Make figure
    [fig, ax1] = plt.subplots(figsize=figsize)
    # Plot
    ax1.plot(out['fig']['sillclust'], out['fig']['silscores'], color='k')
    # Plot optimal cut
    ax1.axvline(x=out['fig']['clustcutt'][idx], ymin=0, ymax=out['fig']['sillclust'][idx], linewidth=2, color='r',linestyle="--")
    # Set fontsizes
    plt.rc('axes', titlesize=14)  # fontsize of the axes title
    plt.rc('xtick', labelsize=10)  # fontsize of the axes title
    plt.rc('ytick', labelsize=10)  # fontsize of the axes title
    plt.rc('font', size=10)
    # Set labels
    ax1.set_xticks(out['fig']['clustcutt'])
    ax1.set_xlabel('#Clusters')
    ax1.set_ylabel('Score')
    ax1.set_title("silhouette score versus number of clusters")
    ax1.grid(color='grey', linestyle='--', linewidth=0.2)

    # Plot silhouette samples plot
    if not isinstance(X, type(None)):
        silhouette_plot(X,out['labx'])