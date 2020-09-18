#-----------------------------------
# Name        : hdbscan.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : MIT
# Respect the autor and leave this here
#-----------------------------------------------

from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan as hdb
import seaborn as sns
import matplotlib.pyplot as plt


# %% Main
def fit(X, metric='euclidean', min_clust=2, min_samples=0.01, norm=True, n_jobs=-1, verbose=3):
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
    min_clust : int, (default: 2)
        Number of clusters that is evaluated greater or equals to min_clust.
    min_samples : float [0..1], (default: 0.01)
        Percentage of expected outliers among number of samples.
    norm : bool, (default: True)
        You may want to set this =0 using distance matrix as input.
    n_jobs : int, (default: -1)
        The number of parallel jobs to run. -1: ALL cpus, 1: Use a single core.
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
    >>> import clusteval.hdbscan as hdbscan
    >>> from sklearn.datasets import make_blobs
    >>> Generate demo data
    >>> X, labels_true = make_blobs(n_samples=750, centers=6, n_features=10)
    >>> # Fit with default parameters
    >>> results = hdbscan.fit(X)
    >>> # plot
    >>> hdbscan.plot(results)

    References
    ----------
    * http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
    * http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    * https://github.com/scikit-learn-contrib/hdbscan

    """
    Param = {}
    Param['min_samples'] = min_samples
    Param['min_clust'] = min_clust
    Param['metric'] = metric
    Param['n_jobs'] = n_jobs
    Param['norm'] = norm
    Param['gen_min_span_tree'] = False
    Param['min_samples'] = np.int(np.floor(min_samples * X.shape[0]))  # Set max. outliers
    # if verbose>=3: print('[clusteval] >Fit using hdbscan.')

    # Transform X
    if Param['norm']:
        X = StandardScaler().fit_transform(X)

    # Set parameters for hdbscan
    model = hdb.HDBSCAN(algorithm='best', metric=Param['metric'], min_samples=np.int(Param['min_samples']), core_dist_n_jobs=Param['n_jobs'], min_cluster_size=np.int(Param['min_clust']), p=None, gen_min_span_tree=Param['gen_min_span_tree'])
    model.fit(X)

    results = {}
    results['method'] ='hdbscan'
    results['labx'] = model.labels_  # Labels
    results['p'] = model.probabilities_  # The strength with which each sample is a member of its assigned cluster. Noise points have probability zero; points in clusters have values assigned proportional to the degree that they persist as part of the cluster.
    results['cluster_persistence'] = model.cluster_persistence_  # A score of how persistent each cluster is. A score of 1.0 represents a perfectly stable cluster that persists over all distance scales, while a score of 0.0 represents a perfectly ephemeral cluster. These scores can be guage the relative coherence of the clusters resultsput by the algorithm.
    results['outlier'] = model.outlier_scores_  # Outlier scores for clustered points; the larger the score the more outlier-like the point. Useful as an outlier detection technique. Based on the GLOSH algorithm by Campello, Moulavi, Zimek and Sander.
    # out2['predict'] = model.prediction_data_  # Cached data used for predicting the cluster labels of new or unseen points. Necessary only if you are using functions from hdbscan.prediction (see approximate_predict(), membership_vector(), and all_points_membership_vectors()).
    results['min_clust'] = Param['min_clust']
    results['model'] = model

    # Some info
    if verbose>=3:
        n_clusters = len(set(results['labx'])) - (1 if -1 in results['labx'] else 0)
        print('[clusteval] >Estimated number of clusters: %d' % n_clusters)

        if n_clusters!=X.shape[0] and n_clusters>1:
            print("[clusteval] >Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, results['labx']))

    return(results)


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
    model = results['model']
    if results['min_clust']==True:
        plt.subplots(figsize=figsize)
        model.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80, edge_linewidth=2)

    plt.subplots(figsize=figsize)
    model.single_linkage_tree_.plot(cmap='viridis', colorbar=True)

    plt.subplots(figsize=figsize)
    model.condensed_tree_.plot()

    plt.subplots(figsize=figsize)
    model.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
