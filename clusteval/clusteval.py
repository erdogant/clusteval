"""clusteval provides methods for unsupervised cluster validation to return the cluster labels with the optimal cutt-off based on the choosen clustering method.

   TODO:
   ----
   * https://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html#sphx-glr-auto-examples-cluster-plot-adjusted-for-chance-measures-py
   * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
"""

# -----------------------------------
# Name        : clusteval.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# -----------------------------------

# %% Libraries
import clusteval.dbindex as dbindex
import clusteval.silhouette as silhouette
import clusteval.derivative as derivative
import clusteval.dbscan as dbscan
from scipy.cluster.hierarchy import linkage as scipy_linkage
# from cuml import DBSCAN

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
        Distance measure for the clustering.
        'euclidean','hamming','kmeans'.
    linkage : str, (default: 'ward')
        Linkage type for the clustering.
        'ward','single',',complete','average','weighted','centroid','median'.
    minclusters : int, (default: 2)
        Minimum number of clusters (>=). 
    maxclusters : int, (default: 25)
        Maximum number of clusters (<=). 
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
    >>> # Generate random data
    >>> from sklearn.datasets import make_blobs
    >>> X, labels_true = make_blobs(n_samples=750, centers=4, n_features=2, cluster_std=0.5)
    >>> # Fit best clusters
    >>> results = ce.fit(X)
    >>> # Make plot
    >>> ce.plot(X)

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
        X : Numpy-array, where rows is features and colums is samples.

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

        # Cluster hierarchical using on metric/linkage
        Z = []
        if self.metric!='kmeans':
            Z=scipy_linkage(X, method=self.linkage, metric=self.metric)

        # Choosing method
        if self.method=='silhouette':
            self.results = silhouette.fit(X, Z=Z, metric=self.metric, minclusters=self.minclusters, maxclusters=self.maxclusters, savemem=self.savemem, verbose=self.verbose)
        elif self.method=='dbindex':
            self.results = dbindex.fit(X, Z=Z, metric=self.metric, minclusters=self.minclusters, maxclusters=self.maxclusters, savemem=self.savemem, verbose=self.verbose)
        elif self.method=='derivative':
            self.results = derivative.fit(X, Z=Z, metric=self.metric, minclusters=self.minclusters, maxclusters=self.maxclusters, verbose=self.verbose)
        elif self.method=='dbscan':
            self.results = dbscan.fit(X, eps=None, epsres=50, min_samples=0.01, metric=self.metric, norm=True, n_jobs=-1, minclusters=self.minclusters, maxclusters=self.maxclusters, verbose=self.verbose)
        elif self.method=='hdbscan':
            try:
                import clusteval.hdbscan as hdbscan
                self.results = hdbscan.fit(X, min_samples=0.01, metric=self.metric, norm=True, n_jobs=-1, minclusters=self.minclusters, verbose=self.verbose)
            except:
                raise ImportError('hdbscan must be installed manually. Try to: <pip install hdbscan>')
        else:
            results = None
            if self.verbose>=3: print('[clusteval] >Method [%s] is not implemented.' %(self.method))

        # Return
        if self.verbose>=3: print('[clusteval] >Fin.')
        return self.results

    # Plot
    def plot(self, X=None, figsize=(15,8)):
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
        if self.verbose>=3: print('[clusteval] >Make plot.')

        if self.method=='silhouette':
            silhouette.plot(self.results, X=X, figsize=figsize)
            silhouette.scatter(self.results, X=X, figsize=figsize)
        elif self.method=='dbindex':
            dbindex.plot(self.results, width=figsize[0], height=figsize[1])
        elif self.method=='derivative':
            derivative.plot(self.results, width=figsize[0], height=figsize[1])
        elif self.method=='dbscan':
            dbscan.plot(self.results, X=X, figsize=figsize)
        elif self.method=='hdbscan':
            import clusteval.hdbscan as hdbscan
            hdbscan.plot(self.results, width=figsize[0], height=figsize[1])
