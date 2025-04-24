"""Silhouette.

# Name        : silhouette.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : MIT
# Respect the autor and leave this here

"""
from clusteval.utils import init_figure
import colourmap
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from clusteval.utils import init_logger, set_logger, disable_tqdm, set_font_properties  # compute_embedding
# logger = init_logger()
import logging
logger = logging.getLogger(__name__)


# %% Main
def fit(X,
        cluster='agglomerative',
        metric='euclidean',
        linkage='ward',
        min_clust=2,
        max_clust=25,
        Z=None,
        savemem=False,
        verbose='info'):
    """This function return the cluster labels for the optimal cutt-off based on the choosen hierarchical clustering evaluate.

    Parameters
    ----------
    X : Numpy-array,
        Where rows is features and colums are samples.
    cluster : str, (default: 'agglomerative')
        Clustering evaluation type for clustering.
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
    savemem : bool, (default: False)
        Save memmory when working with large datasets. Note that this option only works in case of KMeans.
    Z : Object, (default: None).
        This will speed-up computation if you readily have Z. e.g., Z=linkage(X, method='ward', metric='euclidean').

    Returns
    -------
    dict. with various keys. Note that the underneath keys can change based on the used evaluatetype.
    evaluate: str
        evaluate name that is used for cluster evaluation.
    score: pd.DataFrame()
        The scoring values per clusters.
    labx: list
        Cluster labels.
    fig: list
        Relevant information to make the plot.

    Examples
    --------
    >>> # Import library
    >>> import clusteval.silhouette as silhouette
    >>> from sklearn.datasets import make_blobs
    >>>
    >>> # Example 1:
    >>> Generate demo data
    >>> X, labels_true = make_blobs(n_samples=750, centers=5, n_features=10)
    >>> # Fit with default parameters
    >>> results = silhouette.fit(X)
    >>> # plot
    >>> silhouette.scatter(results, X)
    >>> silhouette.plot(results)
    >>>
    >>> # Example 2:
    >>> # Try also alternative dataset
    >>> X, labels_true = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]], cluster_std=0.4,random_state=0)
    >>> # Fit with some specified parameters
    >>> results = silhouette.fit(X, metric='kmeans', savemem=True)
    >>> # plot
    >>> silhouette.scatter(results, X)
    >>> silhouette.plot(results)

    References
    ----------
    http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    """
    # Make dictionary to store Parameters
    Param = {}
    Param['cluster'] = cluster
    Param['metric'] = metric
    Param['linkage'] = linkage
    Param['min_clust'] = min_clust
    Param['max_clust'] = max_clust
    Param['savemem'] = savemem
    set_logger(verbose=verbose)
    logger.info('Evaluate using silhouette.')

    # Savemem for kmeans
    if Param['cluster']=='kmeans':
        if Param['savemem']:
            kmeansmodel = MiniBatchKMeans
            logger.info('Save memory enabled for kmeans with evaluation silhouette.')
        else:
            kmeansmodel = KMeans

    # Cluster hierarchical using on metric/linkage
    if (Z is None) and (Param['cluster']!='kmeans'):
        Z = scipy_linkage(X, method=Param['linkage'], metric=Param['metric'])

    # Setup storing parameters
    clustcutt = np.arange(Param['min_clust'], Param['max_clust'])
    silscores = np.zeros((len(clustcutt))) * np.nan
    sillclust = np.zeros((len(clustcutt))) * np.nan
    clustlabx = []

    # Run over all cluster cutoffs
    for i in tqdm(range(len(clustcutt)), disable=disable_tqdm(), desc='[clusteval] >INFO'):
        # Cut the dendrogram for i clusters
        if Param['cluster']=='kmeans':
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
    I2 = sillclust>=Param['min_clust']
    I3 = sillclust<=Param['max_clust']
    Iloc = I1 & I2 & I3

    logger.debug(clustlabx)
    logger.debug('Iloc: %s' %(str(Iloc)))
    logger.debug('silscores: %s' %(str(silscores)))
    logger.debug('sillclust: %s' %(str(sillclust)))
    logger.debug('clustlabx: %s' %(str(clustlabx)))

    if sum(Iloc)>0:
        # Get only clusters of interest
        silscores = silscores[Iloc]
        sillclust = sillclust[Iloc]
        clustlabx = clustlabx[Iloc, :]
        clustcutt = clustcutt[Iloc]
        idx = np.argmax(silscores)
        clustlabx = clustlabx[idx, :] - 1
    else:
        logger.info('No clusters detected.')
        if len(clustlabx.shape)>1:
            clustlabx = np.zeros(clustlabx.shape[1]).astype(int)
        else:
            clustlabx = [0]

    # Store results
    sillclust=sillclust.astype(int)
    clustcutt=clustcutt.astype(int)
    results = {}
    results['evaluate']='silhouette'
    results['score'] = pd.DataFrame(np.array([clustcutt, sillclust, silscores]).T, columns=['cluster_threshold', 'clusters', 'score'])
    results['score']['clusters'] = results['score']['clusters'].astype(int)
    results['score']['cluster_threshold'] = results['score']['cluster_threshold'].astype(int)
    results['labx'] = clustlabx
    results['fig'] = {}
    results['fig']['silscores'] = silscores
    results['fig']['sillclust'] = sillclust
    results['fig']['clustcutt'] = clustcutt

    # Return
    return results


# %% plot
def plot(results,
         title='Silhouette score',
         xlabel='Nr. Clusters',
         ylabel='Score',
         font_properties={},
         params_line={'color': 'k'},
         params_vline={'color': 'r', 'linewidth': 2, 'linestyle': "--"},
         figsize=(15, 8),
         ax=None,
         visible=True,
         showfig=True):
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
    params_line = {**{'color': 'k'}, **params_line}
    params_vline = {**{'color': 'r', 'linewidth': 2, 'linestyle': "--"}, **params_vline}
    idx = np.argmax(results['fig']['silscores'])

    # Make figure
    fig, ax = init_figure(fig=None, ax=ax, dpi=100, figsize=figsize, visible=visible)

    # Plot
    # ax.plot(results['fig']['sillclust'], results['fig']['silscores'], color='k')
    ax.plot(results['fig']['clustcutt'], results['fig']['silscores'], **params_line)
    # Plot optimal cut
    ax.axvline(x=results['fig']['clustcutt'][idx], ymin=0, ymax=results['fig']['sillclust'][idx], **params_vline)
    # Set fontsizes
    ax.tick_params(axis='x', labelsize=font_properties['size_x_axis'])
    ax.tick_params(axis='y', labelsize=font_properties['size_y_axis'])
    # Set labels
    ax.set_xticks(results['fig']['clustcutt'])
    ax.set_xticklabels(results['fig']['sillclust'])
    ax.set_xlabel(xlabel, fontsize=font_properties['size_x_axis'])
    ax.set_ylabel(ylabel, fontsize=font_properties['size_y_axis'])
    ax.set_title(title, fontsize=font_properties['size_title'])
    ax.grid(color='grey', linestyle='--', linewidth=0.2)

    if showfig: plt.show()
    # Return
    return fig, ax


# %% Scatter data
def scatter(y, X=None, dot_size=50, jitter=None, embedding=None, cmap='tab20c', figsize=(15, 8), font_properties={'size_title': 14, 'size_x_axis': 14, 'size_y_axis': 14}, savefig={'fname': None, format: 'png', 'dpi ': None, 'orientation': 'portrait', 'facecolor': 'auto'}, showfig=True):
    """Make scatter for the cluster labels with the samples.

    Parameters
    ----------
    y: list
        Cluster labels for the samples in X (some order).
    X : Numpy-array,
        Where rows is features and colums are samples. The first two columns of the matrix are used for plotting. Note that it is also possible provide tSNE coordinates for a better representation of the data.
    dot_size : int, (default: 50)
        Size of the dot in the scatterplot
    jitter : float, default: None
        Add jitter to data points as random normal data. Values of 0.01 is usually good for one-hot data seperation.
    embedding : str (default: None)
        In case high dimensional data, a embedding with t-SNE can be performed.
        * None
        * 'tsne'
    savefig : dict.
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
        {'dpi':'figure',
        'format':None,
        'metadata':None,
        'bbox_inches': None,
        'pad_inches':0.1,
        'facecolor':'auto',
        'edgecolor':'auto',
        'backend':None}
    figsize : tuple, (default: (15,8))
        Figure size, (heigh,width).

    Returns
    -------
    tuple, (fig, ax1, ax2)
        Figure and axis of the figure.

    """
    font_properties = set_font_properties(font_properties)
    fig, ax1, ax2 = None, None, None
    if X is None:
        logger.warning('Input data X is required for the scatterplot.')
        return None

    # # Compute embedding
    # X = compute_embedding(y, X, embedding, logger)

    if X.shape[1]>2:
        logger.info('Scatterplot is performed on the first two dimensions of input data X.')
        X = X[:, :2]

    # Extract label from dict
    if isinstance(y, dict):
        y = y.get('labx', None)

    # Check y
    if (y is None) or (len(np.unique(y))==1):
        logger.error('No valid labels provided.')
        return None

    # Add jitter
    if jitter is not None:
        X = X + np.random.normal(0, jitter, size=X.shape)

    # Plot silhouette samples plot
    n_clusters = len(set(y)) - (1 if -1 in y else 0)
    silhouette_avg = silhouette_score(X, y)
    logger.info('Estimated number of n_clusters: %d, average silhouette_score=%.3f' %(n_clusters, silhouette_avg))

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, y)

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=100)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])

    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    y_lower = 10
    uiclust = np.unique(y)
    colors = colourmap.fromlist(uiclust, cmap=cmap, scheme='hex')[1]

    # Make 1st plot
    for label in uiclust:
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        Iloc = y == label
        ith_cluster_silhouette_values = sample_silhouette_values[Iloc]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=colors[label], edgecolor=colors[label], alpha=0.7)
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
        # Scatter
        ax2.scatter(X[Iloc, 0], X[Iloc, 1], marker='.', s=dot_size, lw=0, alpha=0.8, c=colors[label], edgecolor='k')
        ax2.text(X[Iloc, 0].mean(), X[Iloc, 1].mean(), label, c='#000000')

    # Set ax properties
    ax1.set_title("Sample-wise silhouette scores across the clusters", fontsize=font_properties['size_title'])
    ax1.set_xlabel("The silhouette coefficient values", fontsize=font_properties['size_x_axis'])
    ax1.set_ylabel("Cluster label", fontsize=font_properties['size_y_axis'])
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.grid(color='grey', linestyle='--', linewidth=0.2)
    # Set fontsizes
    ax1.tick_params(axis='x', labelsize=font_properties['size_x_axis'])
    ax1.tick_params(axis='y', labelsize=font_properties['size_y_axis'])
    ax1.set_yticks([])  # Clear the yaxis labels / ticks

    # 2nd Plot showing the actual clusters formed
    ax2.grid(color='grey', linestyle='--', linewidth=0.2)
    ax2.set_title("Cluster labels", fontsize=font_properties['size_title'])
    ax2.set_xlabel("1st feature", fontsize=font_properties['size_x_axis'])
    ax2.set_ylabel("2nd feature", fontsize=font_properties['size_y_axis'])
    # Set fontsizes
    ax2.tick_params(axis='x', labelsize=font_properties['size_x_axis'])
    ax2.tick_params(axis='y', labelsize=font_properties['size_y_axis'])

    # General title
    plt.suptitle(("Silhouette analysis. Detected clusters: %d" %(n_clusters)), fontsize=font_properties['size_title'])
    if showfig:
        plt.show()

    # Save figure
    if (savefig['fname'] is not None) and (fig is not None):
        logger.info('Saving silhouetteplot to [%s]' %(savefig['fname']))
        fig.savefig(**savefig)

    # Return
    return fig, ax1, ax2
