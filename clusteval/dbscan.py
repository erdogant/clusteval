""" Density Based clustering.

   import clusteval.dbscan as dbscan

	out = dbscan.fit(X, <optional>)
	fig = dbscan.plot(out,X, <optional>)

 INPUT:
   X:              numpy datamatrix numerical
                   rows    = features
                   colums  = samples
 OPTIONAL

  eps=             Float: The maximum distance between two samples for them to be considered as in the same neighborhood.
                   [None] (default) Determine automatically by the Siloutte score
                   [0.3] 

  epsres=          Integer: Resoultion to test the different epsilons. The higher the longer it will take
                   [100] (default) 

  minclusters=     Integer: Minimum or more number of clusters >=
                   [2] (default)

  maxclusters=     Integer: Maximum or less number of clusters <=
                   [25] (default)

  min_samples=     Integer: [0.,,1] Percentage of expected outliers among number of samples.
                   [0.05] (default)

  metric=          string: Define your input data as type [metrics.pairwise.calculate_distance] or a distance matrix if thats the case!
                   'euclidean' (default) squared euclidean distance
                   'precomputed' if input is a distance matrix!

  norm=            Boolean You may want to set this =0 using distance matrix as input)
                   True: Yes (default) 
                   False: No

  n_jobs=          Integer: The number of parallel jobs to run
                   [-1] ALL cpus (default)
                   [1]  Use a single core
                   
  verbose=   Boolean [0,1]
                   [0]: No 
                   [1]: Some information about embedding
                   [2]: More information about embedding (default)

  showfig=         Boolean [0,1]
                   [0]: No 
                   [1]: Yes (default)

 OUTPUT
	output

 DESCRIPTION
   http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
   http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
   
   
 EXAMPLE
   import clusteval.dbscan as dbscan

   from sklearn.datasets.samples_generator import make_blobs
   [X, labels_true] = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4,random_state=0)
   [X, labels_true] = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]], cluster_std=0.4,random_state=0)

   out = dbscan.fit(X)
   dbscan.plot(out, X=X)


   from VIZ.scatter import scatter
   scatter(X[:,0],X[:,1], size=100, labx=out['labx'])

   
   EXAMPLE 2
   from sklearn.datasets import load_iris
   iris = load_iris()
   X=iris.data

   from tsneBH import tsneBH
   X   = tsneBH(iris.data)
   out = dbscan(X)
   scatter(X[:,0],X[:,1], size=100, labx=out['labx'])
   scatter(X[:,0],X[:,1], size=100, labx=iris.target, labxtype='unique',title='REAL')

   from UMAPet import UMAPet
   X    = UMAPet(iris.data)
   out = dbscan(X)
   scatter(X[:,0],X[:,1], size=100, labx=out['labx'])
   scatter(X[:,0],X[:,1], size=100, labx=iris.target, labx_type='unique',title='REAL')


 SEE ALSO
   HDBSCAN, import sklearn.cluster as cluster
"""

#--------------------------------------------------------------------------
# Name        : dbscan.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Nov. 2017
#--------------------------------------------------------------------------

#%% Libraries
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from clusteval.silhouette_plot import silhouette_plot

#%% Plot
def plot(out, X=None, width=15, height=8, verbose=3):
    idx=out['idx']
    [fig, ax1] = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(out['eps'], out['silscores'], color='k')
    ax1.set_xlabel('eps')
    ax1.set_ylabel('Silhoutte score')
    ax1.grid(color='grey', linestyle='--', linewidth=0.2)
    ax2.plot(out['eps'], out['sillclust'], color='b')
    ax2.set_ylabel('#Clusters')
    ax2.grid(color='grey', linestyle='--', linewidth=0.2)
    # Plot vertical line To stress the cut-off point
    ax2.axvline(x=out['eps'][idx], ymin=0, ymax=out['sillclust'][idx], linewidth=2, color='r')
    
    if not isinstance(X, type(None)):
        if out['n_clusters']!=X.shape[0] and out['n_clusters']>1:
            if verbose>=3: print('[DBSCAN] Estimated number of clusters: %d' %(out['n_clusters']))
            silhouette_plot(X,out['labx'])
    else:
        if verbose>=3: print('[DBSCAN] data required for silhouette plot')
        
#%% Main function
def fit(X, eps=None, min_samples=0.01, metric='euclidean', norm=True, n_jobs=-1, minclusters=2, maxclusters=25, epsres=100, verbose=3):
	# DECLARATIONS
    out={}
    Param                 = {}
    Param['verbose']      = verbose
    Param['eps']          = eps
    Param['metric']       = metric
    Param['n_jobs']       = n_jobs
    Param['norm']         = norm
    Param['minclusters']  = minclusters
    Param['maxclusters']  = maxclusters
    Param['epsres']       = epsres # Resolution of the epsilon to estimate % The higher the more detailed, the more time it costs to compute. Only for DBSCAN
    Param['min_samples']  = np.floor(min_samples*X.shape[0]) # Set max. outliers

    # Transform data
    if Param['norm']: 
        X = StandardScaler().fit_transform(X)

    # Iterate over epsilon
    if Param['eps']==None:
        if Param['verbose']>=3: print('[DBSCAN] Determining optimal clustering by Silhoutte score..')
        # Optimize
        [eps, sillclust, silscores, silllabx]=optimize_eps(X, eps, Param)
        # Store results
        idx = np.argmax(silscores)
        out['methodtype']='dbscan'
        out['labx']  = silllabx[idx,:]
        out['eps']  = eps
        out['silscores'] = silscores
        out['sillclust'] = sillclust
        out['idx'] = idx
    else:
        db = cluster.DBSCAN(eps=Param['eps'], metric=Param['metric'], min_samples=Param['min_samples'], n_jobs=Param['n_jobs'])
        db.fit(X)
        out['labx']=db.labels_

    # Nr of clusters
    out['n_clusters'] = len(set(out['labx'])) - (1 if -1 in out['labx'] else 0)
        
    return(out)

#%% optimize_eps
def optimize_eps(X, eps, Param):
    # Setup resolution
    eps       = np.arange(0.1,5,1/Param['epsres'])
    silscores = np.zeros(len(eps))*np.nan
    sillclust = np.zeros(len(eps))*np.nan
    silllabx  = []

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
        # Compute silhoutte only if more then 1 cluster
        if sillclust[i]>1:
            silscores[i]=silhouette_score(X, db.labels_)

    # Convert to array
    silllabx = np.array(silllabx)
    # Store only if agrees to restriction of input clusters number
    I1 = np.isnan(silscores)==False
    I2 = sillclust>=Param['minclusters']
    I3 = sillclust<=Param['maxclusters']
    I = I1 & I2 & I3
    # Get only those of interest
    silscores = silscores[I]
    sillclust = sillclust[I]
    eps       = eps[I]
    silllabx  = silllabx[I,:]

    return(eps, sillclust, silscores, silllabx)