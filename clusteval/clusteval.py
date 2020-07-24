# -----------------------------------
# Name        : clusteval.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : MIT
# Respect the autor and leave this here
#-----------------------------------------------

import clusteval.dbindex as dbindex
import clusteval.silhouette as silhouette
import clusteval.derivative as derivative
import clusteval.dbscan as dbscan
from clusteval.plot_dendrogram import plot_dendrogram
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.cluster.hierarchy import fcluster
# from cuml import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

# %%
class clusteval():
    """clusteval - Cluster evaluation.

    Description
    -----------
    clusteval is a python package that provides various methods for unsupervised cluster validation.

    Parameters
    ----------
    method : str, (default: 'silhouette' )
        Method type for cluster validation. 
        'silhouette', 'dbindex','derivative','dbscan','hdbscan'.
    metric : str, (default: 'euclidean').
        Distance measure for the clustering, such as 'euclidean','hamming', etc.
    linkage : str, (default: 'ward')
        Linkage type for the clustering.
        'ward','single',',complete','average','weighted','centroid','median'.
    minclusters : int, (default: 2)
        Number of clusters that is evaluated greater or equals to minclusters.
    maxclusters : int, (default: 25)
        Number of clusters that is evaluated smaller or equals to maxclusters.
    savemem : bool, (default: False)
        Save memmory when working with large datasets. Note that htis option only in case of KMeans.
    verbose : int, optional (default: 3)
        Print message to screen [1-5]. The larger the number, the more information.

    Returns
    -------
    dict : The output is a dictionary containing the following keys:

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
    def __init__(self, method='silhouette', metric='euclidean', linkage='ward', minclusters=2, maxclusters=25, savemem=False, verbose=3):
        """Initialize clusteval with user-defined parameters."""

        if (minclusters<2): minclusters=2
        # Store in object
        self.method = method
        self.metric = metric
        self.linkage = linkage
        self.minclusters = minclusters
        self.maxclusters = maxclusters
        self.savemem = savemem
        self.verbose = verbose

    # Fit
    def fit(self, X):
        """Cluster validation.

        Parameters
        ----------
        X : Numpy-array.
            The rows are the features and the colums are the samples.

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

        """
        assert 'array' in str(type(X)), 'Input data must be of type numpy array'
        max_d, max_d_lower, max_d_upper = None, None, None
        # Cluster hierarchical using on metric/linkage
        self.Z = []
        if self.metric!='kmeans':
            self.Z = scipy_linkage(X, method=self.linkage, metric=self.metric)

        # Choosing method
        if self.method=='silhouette':
            self.results = silhouette.fit(X, Z=self.Z, metric=self.metric, minclusters=self.minclusters, maxclusters=self.maxclusters, savemem=self.savemem, verbose=self.verbose)
        elif self.method=='dbindex':
            self.results = dbindex.fit(X, Z=self.Z, metric=self.metric, minclusters=self.minclusters, maxclusters=self.maxclusters, savemem=self.savemem, verbose=self.verbose)
        elif self.method=='derivative':
            self.results = derivative.fit(X, Z=self.Z, metric=self.metric, minclusters=self.minclusters, maxclusters=self.maxclusters, verbose=self.verbose)
        elif self.method=='dbscan':
            self.results = dbscan.fit(X, eps=None, epsres=50, min_samples=0.01, metric=self.metric, norm=True, n_jobs=-1, minclusters=self.minclusters, maxclusters=self.maxclusters, verbose=self.verbose)
        elif self.method=='hdbscan':
            try:
                import clusteval.hdbscan as hdbscan
                self.results = hdbscan.fit(X, min_samples=0.01, metric=self.metric, norm=True, n_jobs=-1, minclusters=self.minclusters, verbose=self.verbose)
            except:
                raise ImportError('hdbscan must be installed manually. Try to: <pip install hdbscan> or <conda install -c conda-forge hdbscan>')
        else:
            results = None
            if self.verbose>=3: print('[clusteval] >Method [%s] is not implemented.' %(self.method))

        # Compute the dendrogram threshold
        if self.metric!='kmeans':
            max_d, max_d_lower, max_d_upper = _compute_dendrogram_threshold(self.Z, self.results['labx'], verbose=self.verbose)

        # Return
        if self.verbose>=3: print('[clusteval] >Fin.')
        self.results['max_d'] = max_d
        self.results['max_d_lower'] = max_d_lower
        self.results['max_d_upper'] = max_d_upper
        return self.results

    # Plot
    def plot(self, figsize=(15, 8)):
        """Make a plot.

        Parameters
        ----------
        figsize : tuple, (default: (15, 8).
            Size of the figure (height,width).

        Returns
        -------
        None.

        """
        if self.results is None:
            if self.verbose>=3: print('[clusteval] >No results to plot. Tip: try the .fit() function first.')
        # if self.verbose>=3: print('[clusteval] >Make plot.')

        if self.method=='silhouette':
            silhouette.plot(self.results, figsize=figsize)
        elif self.method=='dbindex':
            dbindex.plot(self.results, figsize=figsize)
        elif self.method=='derivative':
            derivative.plot(self.results, figsize=figsize)
        elif self.method=='dbscan':
            dbscan.plot(self.results, figsize=figsize)
        elif self.method=='hdbscan':
            import clusteval.hdbscan as hdbscan
            hdbscan.plot(self.results, figsize=figsize)

    # Plot
    def scatter(self, X, figsize=(15, 8)):
        """Make a plot.

        Parameters
        ----------
        X : array-like, (default: None)
            Input dataset used in the .fit() funciton. Some plots will be more extensive if the input data is also provided.
        figsize : tuple, (default: (15,8).
            Size of the figure (height,width).

        Returns
        -------
        None.

        """
        if self.results is None:
            if self.verbose>=3: print('[clusteval] >No results to plot. Tip: try the .fit() function first.')
        # Make scatter
        silhouette.scatter(self.results, X=X, figsize=figsize)

    # Plot dendrogram
    def dendrogram(self, X=None, labels=None, leaf_rotation=90, leaf_font_size=12, orientation='top', show_contracted=True, max_d=None, showfig=True, metric=None, linkage=None, truncate_mode=None, figsize=(15, 10)):
        """Plot Dendrogram

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

        Returns
        -------
        results : dict
            Dictionary containing various keys.
            labx : int : Cluster labels based on the input-ordering.
            order_rows : string : Order of the cluster labels as presented in the dendrogram (left-to-right).
            max_d : float : maximum distance to set the horizontal threshold line.
            max_d_lower : float : maximum distance lowebound
            max_d_upper : float : maximum distance upperbound

        """
        # Set parameters
        no_plot = False if showfig else True
        max_d_lower, max_d_upper = None, None

        # Check whether
        if (metric is not None) & (linkage is not None) & (X is not None):
            if self.verbose>=2: print('[clusteval] >Compute dendrogram using metric=%s, linkage=%s' %(metric, linkage))
            Z = scipy_linkage(X, method=linkage, metric=metric)
        elif (metric is not None) & (linkage is not None) & (X is None):
            if self.verbose>=2: print('[clusteval] >To compute the dendrogram, also provide the data: X=data <return>')
            return None
        elif (not hasattr(self, 'Z')):
            # Return if Z is not computed.
            if self.verbose>=3: print('[clusteval] >No results to plot. Tip: try the .fit() function (no kmeans) <return>')
            return None
        elif self.metric=='kmeans':
            if self.verbose>=3: print('[clusteval] >No results to plot. Tip: try the .fit() function with metric that is different than kmeans <return>')
            return None
        else:
            if self.verbose>=3: print('[clusteval] >Plotting the dendrogram with optimized settings: metric=%s, linkage=%s, max_d=%.3f. Be patient now..' %(self.metric, self.linkage, self.results['max_d']))
            Z = self.Z
            metric = self.metric
            linkage = self.linkage

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
        labx = fcluster(Z, max_d, criterion='distance')

        # Store results
        results['order_rows'] = np.array(results['ivl'])
        results['labx'] = labx
        results['max_d'] = max_d
        results['max_d_lower'] = max_d_lower
        results['max_d_upper'] = max_d_upper
        results['ax'] = ax
        return results


def _compute_dendrogram_threshold(Z, labx, verbose=3):
    if verbose>=3: print('[clusteval] >Compute dendrogram threshold.')
    Iloc = np.isin(Z[:, 3], np.unique(labx, return_counts=True)[1])
    max_d_lower = np.max(Z[Iloc, 2])
    # Find the next level
    max_d_upper = Z[np.where(Z[:, 2] > max_d_lower)[0][0], 2]
    # Average the max_d between the start and stop level
    max_d = max_d_lower + ((max_d_upper - max_d_lower) / 2)
    # Return
    return max_d, max_d_lower, max_d_upper
