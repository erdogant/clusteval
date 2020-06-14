""" This function return the cluster labels for the optimal cutt-off based on the choosen hierarchical clustering method

	out = derivative.fit(X)
	fig = derivative.plot(out)

 INPUT:
   X:            datamatrix
                 rows    = features
                 colums  = samples
 OPTIONAL

 metric=         String: Distance measure for the clustering 
                 'euclidean' (default)

 linkage=        String: Linkage type for the clustering 
                 'ward' (default)

 minclusters=    Integer: Minimum or more number of clusters >=
                 [2] (default)

 maxclusters=    Integer: Maximum or less number of clusters <=
                 [25] (default)

 showfig=        Boolean [0,1]: Progressbar
                 [0]: No (default)
                 [1]: Yes (silhouette plot)
                   
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
   import clusteval.derivative as derivative

   from sklearn.datasets import make_blobs
   [X, labels_true] = make_blobs(n_samples=750, centers=6, n_features=10)
   out= derivative.fit(X)
   fig= derivative.plot(out)


 SEE ALSO
   silhouette, dbindex, 
"""
#print(__doc__)

#--------------------------------------------------------------------------
# Name        : derivative.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Feb. 2018
#--------------------------------------------------------------------------

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as linkage_scipy
import matplotlib.pyplot as plt

#%%
def fit(X, metric='euclidean', linkage='ward', minclusters=2, maxclusters=25, width=15, Z=None, verbose=3):
    # Make dictionary to store Parameters
    Param = {}
    Param['verbose']     = verbose
    Param['metric']      = metric
    Param['linkage']     = linkage
    Param['minclusters'] = minclusters
    Param['maxclusters'] = maxclusters
    
    if Param['metric']=='kmeans':
        print('Does not work with Kmeans! <return>')
        return

    # Cluster hierarchical using on metric/linkage
    if len(Z)==0:
        Z=linkage_scipy(X, method=Param['linkage'], metric=Param['metric'])

    # Make all possible cluster-cut-offs
    if Param['verbose']>=3: print('[DERIVATIVE] Determining optimal clustering by derivatives..')

    # Run over all cluster cutoffs
    last     = Z[-10:, 2]
    last_rev = last[::-1]
    idxs     = np.arange(1, len(last) + 1)
    
    acceleration     = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]

    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    if Param['verbose']>=3: print('[DERIVATIVE] Clusters: %d' %k)
    
    # Now use the optimal cluster cut-off for the selection of clusters
    clustlabx = fcluster(Z, k, criterion='maxclust')

    # Convert to array
    clustlabx = np.array(clustlabx)
    
    # Store results
    out['method']='derivative'
    out['labx'] = clustlabx
    out['fig'] = dict()
    out['fig']['last_rev']=last_rev
    out['fig']['acceleration_rev']=acceleration_rev
    
    return(out)
