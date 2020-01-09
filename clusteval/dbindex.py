""" This function return the cluster labels for the optimal cutt-off based on the choosen hierarchical clustering method

	out = dbindex.fit(X)
	fig = dbindex.plot(out, X)

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

 savemem=        Boolean [0,1]: This works only for KMeans
                 [False]: No (default)
                 [True]: Yes

 showfig=        Boolean [0,1]: Progressbar
                 [0]: No (default)
                 [1]: Yes (silhoutte plot)
                   
 height=         Integer:  Height of figure
                 [5]: (default)

 width=          Integer:  Width of figure
                 [5]: (default)


 Z=              Object from linkage function. This will speed-up computation if you readily have Z
                 [] (default)
                 Z=linkage(X, method='ward', metric='euclidean')
 
 verbose=        Boolean [0,1]: Progressbar
                 [0]: No (default)
                 [1]: Yes

 OUTPUT
	output

 DESCRIPTION
  This function return the cluster labels for the optimal cutt-off based on the choosen clustering method
  
 EXAMPLE
   import clusteval.dbindex as dbindex
   
   from sklearn.datasets.samples_generator import make_blobs
   [X, labels_true] = make_blobs(n_samples=750, centers=6, n_features=10)
   out = dbindex.fit(X)
   fig = dbindex.plot(out)


 SEE ALSO
   silhouette, silhouette_plot, elbowclust
"""
#print(__doc__)

#--------------------------------------------------------------------------
# Name        : dbindex.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Feb. 2018
#--------------------------------------------------------------------------

#%% Libraries
#from matplotlib.pyplot import plot
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans, MiniBatchKMeans

#%% plot
def plot(out, width=15, height=8):
    idx = np.argmin(out['fig']['scores'])
    # Make figure
    [fig, ax1] = plt.subplots(figsize=(width,height))
    # Plot
    ax1.plot(out['fig']['dbclust'], out['fig']['scores'], color='k')
    # Plot optimal cut
    ax1.axvline(x=out['fig']['clustcutt'][idx], ymin=0, ymax=out['fig']['dbclust'][idx], linewidth=2, color='r',linestyle="--")
    # Set fontsizes
    plt.rc('axes', titlesize=14)     # fontsize of the axes title
    plt.rc('xtick', labelsize=10)     # fontsize of the axes title
    plt.rc('ytick', labelsize=10)     # fontsize of the axes title
    plt.rc('font', size=10)
    # Set labels
    ax1.set_xticks(out['fig']['clustcutt'])
    ax1.set_xlabel('#Clusters')
    ax1.set_ylabel('Score')
    ax1.set_title("Davies Bouldin index versus number of clusters")
    ax1.grid(color='grey', linestyle='--', linewidth=0.2)
    
#%% main
def fit(X, metric='euclidean', linkage='ward', minclusters=2, maxclusters=25, Z=[], savemem=False, verbose=3):
	# DECLARATIONS
    out ={}
    
    # Make dictionary to store Parameters
    Param = {}
    Param['verbose']     = verbose
    Param['metric']      = metric
    Param['linkage']     = linkage
    Param['minclusters'] = minclusters
    Param['maxclusters'] = maxclusters
    Param['savemem']     = savemem

    # Savemem for kmeans
    if Param['metric']=='kmeans':
        if Param['savemem']==True:
            kmeansmodel=MiniBatchKMeans
            print('>Save memory enabled for kmeans.')
        else:
            kmeansmodel=KMeans

    # Cluster hierarchical using on metric/linkage
    if len(Z)==0 and Param['metric']!='kmeans':
        from scipy.cluster.hierarchy import linkage
        Z=linkage(X, method=Param['linkage'], metric=Param['metric'])
    
    # Make all possible cluster-cut-offs
    if Param['verbose']>=3: print('[DBINDEX] Determining optimal clustering by Davies-Bouldin Index score..')

    # Setup storing parameters
    clustcutt = np.arange(Param['minclusters'],Param['maxclusters'])
    scores = np.zeros((len(clustcutt)))*np.nan
    dbclust = np.zeros((len(clustcutt)))*np.nan
    clustlabx = []

    # Run over all cluster cutoffs
    for i in tqdm(range(len(clustcutt))):
        # Cut the dendrogram for i clusters
        if Param['metric']=='kmeans':
            labx=kmeansmodel(n_clusters=clustcutt[i], verbose=0).fit(X).labels_
        else:
            labx = fcluster(Z, clustcutt[i], criterion='maxclust')

        # Store labx for cluster-cut
        clustlabx.append(labx)
        # Store number of unique clusters
        dbclust[i]=len(np.unique(labx))
        # Compute silhoutte (can only be done if more then 1 cluster)
        if dbclust[i]>1:
            scores[i]=dbindex_score(X, labx)


    # Convert to array
    clustlabx = np.array(clustlabx)
    
    # Store only if agrees to restriction of input clusters number
    I1 = np.isnan(scores)==False
    I2 = dbclust>=Param['minclusters']
    I3 = dbclust<=Param['maxclusters']
    I  = I1 & I2 & I3

    # Get only clusters of interest
    scores = scores[I]
    dbclust = dbclust[I]
    clustlabx = clustlabx[I,:]
    clustcutt = clustcutt[I]
    idx = np.argmin(scores)
    
    # Store results
    out['methodtype']='dbindex'
    out['score'] = pd.DataFrame(np.array([dbclust,scores]).T, columns=['clusters','score'])
    out['score'].clusters = out['score'].clusters.astype(int)
    out['labx']  = clustlabx[idx,:]-1
    out['fig']=dict()
    out['fig']['dbclust']=dbclust
    out['fig']['scores']=scores
    out['fig']['clustcutt']=clustcutt
    
    return(out)

#%% Compute DB-score
def dbindex_score(X, labels):
    n_cluster = np.unique(labels)
    cluster_k=[]
    for k in range(0, len(n_cluster)):
        cluster_k.append(X[labels==n_cluster[k]])

    centroids = [np.mean(k, axis = 0) for k in cluster_k]
    variances = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]

    db = []
    for i in range(0,len(n_cluster)):
        for j in range(0,len(n_cluster)):
            if n_cluster[j] != n_cluster[i]:
                db.append( (variances[i] + variances[j]) / euclidean(centroids[i], centroids[j]) )

    outscore = np.max(db) / len(n_cluster)
    return(outscore)