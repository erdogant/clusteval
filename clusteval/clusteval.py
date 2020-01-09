""" clusteval provides methods for unsupervised cluster validation

	out = clusteval.fit(X, <optional>)
	      clusteval.plot(out, X)

   X:            datamatrix
                 rows    = features
                 colums  = samples
 OPTIONAL

 method=          String: Method type for cluster validation
                 'silhouette' (default)
                 'dbindex'
                 'derivative'
                 'hdbscan'
                 'dbscan' (the default settings it the use of silhoutte)

 metric=         String: Distance measure for the clustering 
                 'euclidean' (default, hierarchical)
                 'hamming'
                 'kmeans' (prototypes)

 linkage=        String: Linkage type for the clustering 
                 'ward' (default)
                 'single
                 'complete'
                 'average'
                 'weighted'
                 'centroid'
                 'median'

 minclusters=    Integer: Minimum or more number of clusters >=
                 [2] (default)

 maxclusters=    Integer: Maximum or less number of clusters <=
                 [25] (default)

 savemem=        Boolean: This works only for KMeans
                 [False]: No (default)
                 [True]: Yes

 height=         Integer:  Height of figure
                 [5]: (default)

 width=          Integer:  Width of figure
                 [5]: (default)

 verbose=        Boolean [0,1]: Progressbar
                 [0]: No (default)
                 [1]: Yes

 OUTPUT
	output

 DESCRIPTION
  This function return the cluster labels for the optimal cutt-off based on the choosen clustering method
  
 EXAMPLE
   import clusteval as clusteval
   
   from sklearn.datasets import make_blobs
   [X, labels_true] = make_blobs(n_samples=750, centers=4, n_features=2, cluster_std=0.5)

   out = clusteval.fit(X, method='silhouette')
   out = clusteval.fit(X, method='dbindex')
   out = clusteval.fit(X, method='derivative')
   out = clusteval.fit(X, method='hdbscan')
   out = clusteval.fit(X, method='dbscan')
   
   clusteval.plot(out, X)


 SEE ALSO
   silhouette, silhouette_plot, dbindex, derivative, dbscan, hdbscan
   
"""

#--------------------------------------------------------------------------
# Name        : clusteval.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Feb. 2018
#--------------------------------------------------------------------------

#%% Libraries
import clusteval.dbindex as dbindex
import clusteval.silhouette as silhouette
import clusteval.derivative as derivative
import clusteval.dbscan as dbscan
import clusteval.hdbscan as hdbscan
from scipy.cluster.hierarchy import linkage as scipy_linkage

#%%
def fit(X, method='silhouette', metric='euclidean', linkage='ward', minclusters=2, maxclusters=25, savemem=False, verbose=1):
	# DECLARATIONS
    assert 'array' in str(type(X)), 'Input data must be of type numpy array'
    out ={}
    Param = {}
    Param['method']      = method
    Param['verbose']     = verbose
    Param['metric']      = metric
    Param['linkage']     = linkage
    Param['minclusters'] = minclusters
    Param['maxclusters'] = maxclusters
    Param['savemem']     = savemem

    # Cluster hierarchical using on metric/linkage
    Z = []
    if Param['metric']!='kmeans':
        Z=scipy_linkage(X, method=Param['linkage'], metric=Param['metric'])
    
    # Choosing method
    if Param['method']=='silhouette':
        out=silhouette.fit(X, Z=Z, metric=Param['metric'], minclusters=Param['minclusters'] , maxclusters=Param['maxclusters'], savemem=Param['savemem'], verbose=Param['verbose'])

    if Param['method']=='dbindex':
        out=dbindex.fit(X, Z=Z, metric=Param['metric'], minclusters=Param['minclusters'] , maxclusters=Param['maxclusters'], savemem=Param['savemem'], verbose=Param['verbose'])

    if Param['method']=='derivative':
        out=derivative.fit(X, Z=Z, metric=Param['metric'], minclusters=Param['minclusters'] , maxclusters=Param['maxclusters'], verbose=Param['verbose'])

    if Param['method']=='dbscan':
        out=dbscan.fit(X, eps=None, epsres=100, min_samples=0.01, metric=Param['metric'], norm=True, n_jobs=-1, minclusters=Param['minclusters'], maxclusters=Param['maxclusters'], verbose=Param['verbose'])

    if Param['method']=='hdbscan':
        out=hdbscan.fit(X, min_samples=0.01, metric=Param['metric'], norm=True, n_jobs=-1, minclusters=Param['minclusters'], verbose=Param['verbose'])

    return(out)

#%% Plot
def plot(out, X=None, width=15, height=8):
    if out['methodtype']=='silhoutte':
        silhouette.plot(out, X=X, width=15, height=8)
    if out['methodtype']=='dbindex':
        dbindex.plot(out, width=15, height=8)
    if out['methodtype']=='derivative':
        derivative.plot(out, width=15, height=8)
    if out['methodtype']=='dbscan':
        dbscan.plot(out, X=X, width=15, height=8)
    if out['methodtype']=='hdbscan':
        hdbscan.plot(out, width=15, height=8)
    