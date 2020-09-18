# -----------------------------------
# Name        : dbscan.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : MIT
# Respect the autor and leave this here
# -----------------------------------------------

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import clusteval.silhouette as silhouette


# %% Main function
def fit(X, eps=None, min_samples=0.01, metric='euclidean', norm=True, n_jobs=-1, min_clust=2, max_clust=25, epsres=100, verbose=3):
    """Density Based clustering.

    Parameters
    ----------
    X : Numpy-array,
        Where rows is features and colums are samples.
    eps : float, (default: None)
        The maximum distance between two samples for them to be considered as in the same neighborhood.
        None: Determine automatically by the Siloutte score. Otherwise, 0.3 etc
    min_samples : float [0..1], (default: 0.01)
        Percentage of expected outliers among number of samples.
    metric : str, (default: 'euclidean').
        Distance measure for the clustering. Types can be found at [metrics.pairwise.calculate_distance] or a distance matrix if thats the case.
        'euclidean' (default) squared euclidean distance or 'precomputed' if input is a distance matrix!
    norm : bool, (default: True)
        Normalize the input data. You may want to set this when using a distance matrix as input.
    n_jobs : int, (default: -1)
        The number of parallel jobs to run. -1: ALL cpus, 1: Use a single core.
    min_clust : int, (default: 2)
        Number of clusters that is evaluated greater or equals to min_clust.
    max_clust : int, (default: 25)
        Number of clusters that is evaluated smaller or equals to max_clust.
    epsres : int, (default: 100)
        Resoultion to test the different epsilons. The higher the longer it will take.
    verbose : int, optional (default: 3)
        Print message to screen [1-5]. The larger the number, the more information is returned.

    Returns
    -------
    dict. with various keys. Note that the underneath keys can change based on the used methodtype.
    method: str
        Method name that is used for cluster evaluation.
    labx: list
        Cluster labels.

    Examples
    --------
    >>> Generate demo data
    >>> import clusteval.dbscan as dbscan
    >>> from sklearn.datasets import make_blobs
    >>> [X, labels_true] = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4,random_state=0)
    >>> [X, labels_true] = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]], cluster_std=0.4,random_state=0)
    >>> # Fit with default parameters
    >>> results = dbscan.fit(X)
    >>> # plot
    >>> dbscan.plot(results)

    References
    ----------
    * http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
    * http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

    """
    Param = {}
    Param['verbose'] = verbose
    Param['eps'] = eps
    Param['metric'] = metric
    Param['n_jobs'] = n_jobs
    Param['norm'] = norm
    Param['min_clust'] = min_clust
    Param['max_clust'] = max_clust
    Param['epsres'] = epsres  # Resolution of the epsilon to estimate % The higher the more detailed, the more time it costs to compute. Only for DBSCAN
    Param['min_samples'] = np.floor(min_samples * X.shape[0])  # Set max. outliers
    # if verbose>=3: print('[clusteval] >Fit using dbscan.')

    # Transform data
    if Param['norm']:
        if Param['verbose']>=3: print('[clusteval] >Normalize data (unit variance, zero-mean).')
        X = StandardScaler().fit_transform(X)

    # Iterate over epsilon
    results = {}
    if Param['eps'] is None:
        if Param['verbose']>=3: print('[clusteval] >Gridsearch across epsilon..')
        # Optimize
        [eps, sillclust, silscores, silllabx] = _optimize_eps(X, eps, Param, verbose=verbose)
        # Store results
        idx = np.argmax(silscores)
        results['method']='dbscan'
        results['labx'] = silllabx[idx, :]
        results['fig'] = {}
        results['fig']['eps'] = eps
        results['fig']['silscores'] = silscores
        results['fig']['sillclust'] = sillclust
        results['fig']['idx'] = idx
    else:
        db = cluster.DBSCAN(eps=Param['eps'], metric=Param['metric'], min_samples=Param['min_samples'], n_jobs=Param['n_jobs'])
        db.fit(X)
        results['labx'] = db.labels_

    # Nr of clusters
    results['n_clusters'] = len(set(results['labx'])) - (1 if -1 in results['labx'] else 0)
    # Return
    return(results)


# %% optimize_eps
def _optimize_eps(X, eps, Param, verbose=3):
    if verbose>=3: print('[clusteval] >Evaluate using silhouette..')

    # Setup resolution
    eps = np.arange(0.1, 5, 1 / Param['epsres'])
    silscores = np.zeros(len(eps)) * np.nan
    sillclust = np.zeros(len(eps)) * np.nan
    silllabx = []

    # Run over all Epsilons
    for i in tqdm(range(len(eps))):
        # DBSCAN
        db = cluster.DBSCAN(eps=eps[i], metric=Param['metric'], min_samples=Param['min_samples'], n_jobs=Param['n_jobs']).fit(X)
        # Get labx
        labx=db.labels_

        # Fill array
        sillclust[i]=len(np.unique(labx))
        # Store all labx
        silllabx.append(labx)
        # Compute Silhouette only if more then 1 cluster
        if sillclust[i]>1:
            silscores[i] = silhouette_score(X, db.labels_)

    # Convert to array
    silllabx = np.array(silllabx)
    # Store only if agrees to restriction of input clusters number
    I1 = np.isnan(silscores)==False
    I2 = sillclust >= Param['min_clust']
    I3 = sillclust <= Param['max_clust']
    Iloc = I1 & I2 & I3
    # Get only those of interest
    silscores = silscores[Iloc]
    sillclust = sillclust[Iloc]
    eps = eps[Iloc]
    silllabx = silllabx[Iloc, :]
    # Return
    return(eps, sillclust, silscores, silllabx)


# %% Plot
def plot(results, figsize=(15, 8), verbose=3):
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
    # Setup figure properties
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # Make figure 1
    idx = results['fig']['idx']
    ax1.plot(results['fig']['eps'], results['fig']['silscores'], color='k')
    ax1.set_xlabel('eps')
    ax1.set_ylabel('Silhouette score')
    ax1.grid(color='grey', linestyle='--', linewidth=0.2)

    # Make figure 2
    ax2.plot(results['fig']['eps'], results['fig']['sillclust'], color='b')
    ax2.set_ylabel('#Clusters')
    ax2.grid(color='grey', linestyle='--', linewidth=0.2)
    # Plot vertical line To stress the cut-off point
    ax2.axvline(x=results['fig']['eps'][idx], ymin=0, ymax=results['fig']['sillclust'][idx], linewidth=2, color='r')
    plt.show()
    # Return
    return (fig, (ax1, ax2))
