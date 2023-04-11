# -----------------------------------
# Name        : clusteval.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See LICENSE
# Respect the autor and leave this here
# -----------------------------------------------

import clusteval.dbindex as dbindex
import clusteval.silhouette as silhouette
import clusteval.derivative as derivative
import clusteval.dbscan as dbscan
from clusteval.utils import init_logger, set_logger
from clusteval.plot_dendrogram import plot_dendrogram

import pypickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.cluster.hierarchy import fcluster
# from cuml import DBSCAN

from urllib.parse import urlparse
import logging
import requests
import os

logger = init_logger()

# %% Class
class clusteval:
    """Cluster evaluation.

    Description
    -----------
    clusteval is a python package that provides various evaluation approaches to measure the goodness of the unsupervised clustering.

    Parameters
    ----------
    cluster : str, (default: 'agglomerative')
        Type of clustering.
            * 'agglomerative'
            * 'kmeans'
            * 'dbscan'
            * 'hdbscan'
            * 'optics' # TODO
    evaluate : str, (default: 'silhouette')
        Evaluation method for cluster validation.
            * 'silhouette'
            * 'dbindex'
            * 'derivative'
    metric : str, (default: 'euclidean').
        Distance measures. All metrics from sklearn can be used such as:
            * 'euclidean'
            * 'hamming'
            * 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
    linkage : str, (default: 'ward')
        Linkage type for the clustering.
            * 'ward'
            * 'single'
            * 'complete'
            * 'average'
            * 'weighted'
            * 'centroid'
            * 'median'
    min_clust : int, (default: 2)
        Number of clusters that is evaluated greater or equals to min_clust.
    max_clust : int, (default: 25)
        Number of clusters that is evaluated smaller or equals to max_clust.
    savemem : bool, (default: False)
        Save memmory when working with large datasets. Note that htis option only in case of KMeans.
    verbose : int, optional (default: 3)
        Print message to screen [1-5]. The larger the number, the more information.

    Returns
    -------
    dict : dictionary with keys:

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
    >>> # Dendrogram
    >>> ce.dendrogram()

    """

    def __init__(self, cluster='agglomerative', evaluate='silhouette', metric='euclidean', linkage='ward', min_clust=2, max_clust=25, savemem=False, verbose='info', params_dbscan={'eps': None, 'epsres': 50, 'min_samples': 0.01, 'norm': False, 'n_jobs': -1}):
        """Initialize clusteval with user-defined parameters."""
        if ((min_clust is None) or (min_clust<2)):
            min_clust=2
        if ((max_clust is None) or (max_clust<min_clust)):
            max_clust=min_clust + 1

        if not np.any(np.isin(evaluate, ['silhouette', 'dbindex', 'derivative'])): raise ValueError("evaluate has incorrect input argument [%s]." %(evaluate))
        if not np.any(np.isin(cluster, ['agglomerative', 'kmeans', 'dbscan', 'hdbscan'])): raise ValueError("cluster has incorrect input argument [%s]." %(cluster))

        # Set parameters for dbscan
        dbscan_defaults = {'metric': metric, 'min_clust': min_clust, 'max_clust': max_clust, 'eps': None, 'epsres': 50, 'min_samples': 0.01, 'norm': False, 'n_jobs': -1, 'verbose': verbose}
        params_dbscan = {**dbscan_defaults, **params_dbscan}
        self.params_dbscan = params_dbscan

        # Store in object
        self.evaluate = evaluate
        self.cluster = cluster
        self.metric = metric
        self.linkage = linkage
        self.min_clust = min_clust
        self.max_clust = max_clust
        self.savemem = savemem
        self.verbose = verbose
        # Set the logger
        set_logger(verbose=verbose)

    # Fit
    def fit(self, X):
        """Cluster validation.

        Parameters
        ----------
        X : Numpy-array.
            The rows are the features and the colums are the samples.

        Returns
        -------
        dict. with various keys. Note that the underneath keys can change based on the used evaluation method.
            evaluate: str
                evaluate name that is used for cluster evaluation.
            score: pd.DataFrame()
                The scoring values per clusters [silhouette, dbindex] provide this information.
            labx: list
                Cluster labels.
            fig: list
                Relevant information to make the plot.

        """
        if 'array' not in str(type(X)): raise ValueError('Input data must be of type numpy array')
        max_d, max_d_lower, max_d_upper = None, None, None
        self.Z = []

        # Cluster using on metric/linkage
        logger.info('Fit using %s with metric: %s, and linkage: %s' %(self.cluster, self.metric, self.linkage))
        # Compute linkages
        if self.cluster!='kmeans':
            self.Z = scipy_linkage(X, method=self.linkage, metric=self.metric)

        # Choosing method
        if (self.cluster=='agglomerative') or (self.cluster=='kmeans'):
            if self.evaluate=='silhouette':
                self.results = silhouette.fit(X, Z=self.Z, cluster=self.cluster, metric=self.metric, min_clust=self.min_clust, max_clust=self.max_clust, savemem=self.savemem, verbose=self.verbose)
            elif self.evaluate=='dbindex':
                self.results = dbindex.fit(X, Z=self.Z, metric=self.metric, min_clust=self.min_clust, max_clust=self.max_clust, savemem=self.savemem, verbose=self.verbose)
            elif self.evaluate=='derivative':
                self.results = derivative.fit(X, Z=self.Z, cluster=self.cluster, metric=self.metric, min_clust=self.min_clust, max_clust=self.max_clust, verbose=self.verbose)
        elif (self.cluster=='dbscan') and (self.evaluate=='silhouette'):
            self.results = dbscan.fit(X, eps=self.params_dbscan['eps'], epsres=self.params_dbscan['epsres'], min_samples=self.params_dbscan['min_samples'], metric=self.metric, norm=self.params_dbscan['norm'], n_jobs=self.params_dbscan['n_jobs'], min_clust=self.min_clust, max_clust=self.max_clust, verbose=self.verbose)
        elif self.cluster=='hdbscan':
            try:
                import clusteval.hdbscan as hdbscan
            except:
                raise ValueError('hdbscan must be installed manually. Try to: <pip install hdbscan> or <conda install -c conda-forge hdbscan>')
            self.results = hdbscan.fit(X, min_samples=None, metric=self.metric, norm=True, n_jobs=-1, min_clust=self.min_clust, verbose=self.verbose)
        else:
            raise ValueError('The combination cluster"%s", evaluate="%s" is not implemented.' %(self.cluster, self.evaluate))

        # Compute the dendrogram threshold
        max_d, max_d_lower, max_d_upper = None, None, None

        # Compute the dendrogram threshold
        if (self.cluster!='kmeans') and hasattr(self, 'results') and (self.results['labx'] is not None) and (len(np.unique(self.results['labx']))>1):
            # logger.info(self.results['labx'])
            max_d, max_d_lower, max_d_upper = _compute_dendrogram_threshold(self.Z, self.results['labx'], verbose=self.verbose)

        if self.results['labx'] is not None:
            logger.info('Optimal number clusters detected: [%.0d].' %(len(np.unique(self.results['labx']))))

        self.results['max_d'] = max_d
        self.results['max_d_lower'] = max_d_lower
        self.results['max_d_upper'] = max_d_upper
        logger.info('Fin.')

        # Return
        return self.results

    # Plot
    def plot(self, title=None, figsize=(15, 8), savefig={'fname': None, format: 'png', 'dpi ': None, 'orientation': 'portrait', 'facecolor': 'auto'}, ax=None, verbose=3):
        """Make a plot.

        Parameters
        ----------
        figsize : tuple, (default: (15, 8).
            Size of the figure (height,width).
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
        verbose : int, optional (default: 3)
            Print message to screen [1-5]. The larger the number, the more information.

        Returns
        -------
        tuple: (fig, ax)

        """
        if ax is None: fig = None
        if (self.results is None) or (self.results['labx'] is None):
            logger.info('No results to plot. Tip: try the .fit() function first.')
            return None

        if (self.cluster=='agglomerative') or (self.cluster=='kmeans'):
            if self.evaluate=='silhouette':
                fig, ax = silhouette.plot(self.results, figsize=figsize, title=title, ax=ax)
            elif self.evaluate=='dbindex':
                fig, ax = dbindex.plot(self.results, figsize=figsize, title=title, ax=ax)
            elif self.evaluate=='derivative':
                fig, ax = derivative.plot(self.results, title=title, figsize=figsize, ax=ax)
        elif self.cluster=='dbscan':
            fig, ax = dbscan.plot(self.results, figsize=figsize, title=title, ax=ax)
        elif self.cluster=='hdbscan':
            import clusteval.hdbscan as hdbscan
            fig, ax = hdbscan.plot(self.results, figsize=figsize, savefig=savefig)

        # Save figure
        if (savefig['fname'] is not None) and (fig is not None) and (self.cluster!='hdbscan'):
            logger.info('Saving plot: [%s]' %(savefig['fname']))
            fig.savefig(**savefig)

        # Return
        return fig, ax

    # Plot
    def scatter(self, X,
                dot_size=75,
                jitter=None,
                figsize=(15, 8),
                savefig={'fname': None, format: 'png', 'dpi ': None, 'orientation': 'portrait', 'facecolor': 'auto'},
                verbose=3):
        """Make a plot.

        Parameters
        ----------
        X : array-like, (default: None)
            Input dataset used in the .fit() funciton. Some plots will be more extensive if the input data is also provided.
        dot_size : int, (default: 50)
            Size of the dot in the scatterplot
        jitter : float, default: None
            Add jitter to data points as random normal data. Values of 0.01 is usually good for one-hot data seperation.
        figsize : tuple, (default: (15,8).
            Size of the figure (height,width).
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
        verbose : int, optional (default: 3)
            Print message to screen [1-5]. The larger the number, the more information.

        Returns
        -------
        None.

        """
        if (self.results is None) or (self.results['labx'] is None):
            logger.info('No results to plot. Tip: try the .fit() function first.')
            return None
        # Make scatterplot
        fig, ax1, ax2 = silhouette.scatter(self.results, X=X, dot_size=dot_size, figsize=figsize, jitter=jitter, savefig=savefig)
        # Return
        return fig, ax1, ax2

    # Plot dendrogram
    def dendrogram(self, X=None, labels=None, leaf_rotation=90, leaf_font_size=12, orientation='top', show_contracted=True, max_d=None, showfig=True, metric=None, linkage=None, truncate_mode=None, figsize=(15, 10), savefig={'fname': None, format: 'png', 'dpi ': None, 'orientation': 'portrait', 'facecolor': 'auto'}, verbose=3):
        """Plot Dendrogram.

        Parameters
        ----------
        X : numpy-array (default : None)
            Input data.
        labels : list, (default: None)
            Plot the labels. When None: the index of the original observation is used to label the leaf nodes.
        leaf_rotation : int, (default: 90)
            Rotation of the labels [0-360].
        leaf_font_size : int, (default: 12)
            Font size labels.
        orientation : string, (default: 'top')
            Direction of the dendrogram: 'top', 'bottom', 'left' or 'right'
        show_contracted : bool, (default: True)
            The heights of non-singleton nodes contracted into a leaf node are plotted as crosses along the link connecting that leaf node.
        max_d : Float, (default: None)
            Height of the dendrogram to make a horizontal cut-off line.
        showfig : bool, (default = True)
            Plot the dendrogram.
        metric : str, (default: 'euclidean').
            Distance measure for the clustering, such as 'euclidean','hamming', etc.
        linkage : str, (default: 'ward')
            Linkage type for the clustering.
            'ward','single',',complete','average','weighted','centroid','median'.
        truncate_mode : string, (default: None)
            Truncation is used to condense the dendrogram, which can be based on: 'level', 'lastp' or None
        figsize : tuple, (default: (15, 10).
            Size of the figure (height,width).
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
        verbose : int, optional (default: 3)
            Print message to screen [1-5]. The larger the number, the more information.

        Returns
        -------
        results : dict
            * labx : int : Cluster labels based on the input-ordering.
            * order_rows : string : Order of the cluster labels as presented in the dendrogram (left-to-right).
            * max_d : float : maximum distance to set the horizontal threshold line.
            * max_d_lower : float : maximum distance lowebound
            * max_d_upper : float : maximum distance upperbound

        """
        fig, ax = None, None
        if (self.results is None) or (self.results['labx'] is None):
            logger.info('No results to plot. Tip: try the .fit() function first.')
            return None

        # Set parameters
        no_plot = False if showfig else True
        max_d_lower, max_d_upper = None, None

        # Check whether
        if (metric is not None) and (linkage is not None) and (X is not None):
            logger.warning('Compute dendrogram using metric=%s, linkage=%s' %(metric, linkage))
            Z = scipy_linkage(X, method=linkage, metric=metric)
        elif (metric is not None) and (linkage is not None) and (X is None):
            logger.warning('To compute the dendrogram, also provide the data: X=data <return>')
            return None
        elif (not hasattr(self, 'Z')):
            # Return if Z is not computed.
            logger.info('No results to plot. Tip: try the .fit() function (no kmeans) <return>')
            return None
        else:
            logger.info('Plotting the dendrogram with optimized settings: metric=%s, linkage=%s, max_d=%.3f. Be patient now..' %(self.metric, self.linkage, self.results['max_d']))
            Z = self.Z
            metric = self.metric
            linkage = self.linkage

        if self.cluster=='kmeans':
            logger.info('No results to plot. Tip: try the .fit() function with metric that is different than kmeans <return>')
            return None

        if max_d is None:
            max_d = self.results['max_d']
            max_d_lower = self.results['max_d_lower']
            max_d_upper = self.results['max_d_upper']

        # Make the dendrogram
        if showfig:
            fig, ax = plt.subplots(figsize=figsize)
        annotate_above = max_d
        results = plot_dendrogram(Z, labels=labels, leaf_rotation=leaf_rotation, leaf_font_size=leaf_font_size, orientation=orientation, show_contracted=show_contracted, annotate_above=annotate_above, max_d=max_d, truncate_mode=truncate_mode, ax=ax, no_plot=no_plot)

        # Compute cluster labels
        logger.info('Compute cluster labels.')
        labx = fcluster(Z, max_d, criterion='distance')

        # Store results
        results['order_rows'] = np.array(results['ivl'])
        results['labx'] = labx
        results['max_d'] = max_d
        results['max_d_lower'] = max_d_lower
        results['max_d_upper'] = max_d_upper
        results['ax'] = ax

        # Save figure
        if (savefig['fname'] is not None) and (fig is not None):
            logger.info('Saving dendrogram: [%s]' %(savefig['fname']))
            fig.savefig(**savefig)

        return results

    def save(self, filepath='clusteval.pkl', overwrite=False):
        """Save model in pickle file.

        Parameters
        ----------
        filepath : str, (default: 'clusteval.pkl')
            Pathname to store pickle files.
        overwrite : bool, (default=False)
            Overwite file if exists.
        verbose : int, optional
            Show message. A higher number gives more informatie. The default is 3.

        Returns
        -------
        bool : [True, False]
            Status whether the file is saved.

        """
        if (filepath is None) or (filepath==''):
            filepath = 'clusteval.pkl'
        if filepath[-4:] != '.pkl':
            filepath = filepath + '.pkl'
        # Store data
        storedata = {}
        storedata['results'] = self.results
        # Save
        status = pypickle.save(filepath, storedata, overwrite=overwrite, verbose=3)
        # return
        return status

    def load(self, filepath='clusteval.pkl', verbose=3):
        """Restore previous results.

        Parameters
        ----------
        filepath : str
            Pathname to stored pickle files.
        verbose : int, optional
            Show message. A higher number gives more information. The default is 3.

        Returns
        -------
        Object.

        """
        if (filepath is None) or (filepath==''):
            filepath = 'clusteval.pkl'
        if filepath[-4:]!='.pkl':
            filepath = filepath + '.pkl'

        # Load
        storedata = pypickle.load(filepath, verbose=verbose)

        # Restore the data in self
        if storedata is not None:
            self.results = storedata['results']
            return self.results

    def import_example(self, data='titanic', url=None, sep=',', verbose=3):
        """Import example dataset from github source.

        Description
        -----------
        Import one of the few datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail', 'breast', 'iris'
        url : str
            url link to to dataset.
        sep : str
            Delimiter of the data set.
        verbose : int, optional
            Show message. A higher number gives more information. The default is 3.

        Returns
        -------
        pd.DataFrame()
            Dataset containing mixed features.
        
        References
        ----------
            * student: https://archive-beta.ics.uci.edu/dataset/320/student+performance

        """
        return import_example(data=data, url=url, sep=sep, verbose=verbose)


# %% Compute dendrogram threshold
def _compute_dendrogram_threshold(Z, labx, verbose=3):
    logger.info('Compute dendrogram threshold.')
    Iloc = np.isin(Z[:, 3], np.unique(labx, return_counts=True)[1])
    max_d_lower = np.max(Z[Iloc, 2])
    # Find the next level
    if np.any(Z[:, 2] > max_d_lower):
        max_d_upper = Z[np.where(Z[:, 2] > max_d_lower)[0][0], 2]
    else:
        max_d_upper = np.sort(Z[Iloc, 2])[-2]
    # Average the max_d between the start and stop level
    max_d = max_d_lower + ((max_d_upper - max_d_lower) / 2)
    # Return
    return max_d, max_d_lower, max_d_upper


# %% Import example dataset from github.
def import_example(data='titanic', url=None, sep=',', verbose=3):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail', 'breast', 'iris'
    url : str
        url link to to dataset.
    sep : str
        Delimiter of the data set.
    verbose : int, (default: 20)
        Print progress to screen. The default is 3.
        60: None, 40: Error, 30: Warn, 20: Info, 10: Debug

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    from sklearn import datasets

    if url is None:
        if data=='sprinkler':
            url='https://erdogant.github.io/datasets/sprinkler.zip'
        elif data=='titanic':
            url='https://erdogant.github.io/datasets/titanic_train.zip'
        elif data=='student':
            url='https://erdogant.github.io/datasets/student_train.zip'
        elif data=='cancer':
            url='https://erdogant.github.io/datasets/cancer_dataset.zip'
        elif data=='fifa':
            url='https://erdogant.github.io/datasets/FIFA_2018.zip'
        elif data=='waterpump':
            url='https://erdogant.github.io/datasets/waterpump/waterpump_test.zip'
        elif data=='retail':
            url='https://erdogant.github.io/datasets/marketing_data_online_retail_small.zip'
            sep = ';'
        elif data=='iris':
            X, y = datasets.load_iris(return_X_y=True)
            return X, y
        elif data=='breast':
            X, y = datasets.load_breast_cancer(return_X_y=True)
            return X, y
    else:
        data = wget.filename_from_url(url)

    if url is None:
        logger.info('Nothing to download.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    filename = os.path.basename(urlparse(url).path)
    PATH_TO_DATA = os.path.join(curpath, filename)
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        logger.info('Downloading [%s] dataset from github source..' %(data))
        wget.download(url, PATH_TO_DATA)

    # Import local dataset
    logger.info('Import dataset [%s]' %(data))
    df = pd.read_csv(PATH_TO_DATA, sep=sep)
    # Return
    return df


# %% Retrieve files files.
class wget:
    """Retrieve file from url."""

    def filename_from_url(url):
        """Return filename."""
        return os.path.basename(url)

    def download(url, writepath):
        """Download.

        Parameters
        ----------
        url : str.
            Internet source.
        writepath : str.
            Directory to write the file.

        Returns
        -------
        None.

        """
        r = requests.get(url, stream=True)
        with open(writepath, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024):
                fd.write(chunk)
