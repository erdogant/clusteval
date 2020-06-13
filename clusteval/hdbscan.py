""" Density Based clustering.

   import clusteval.hdbscan as hdbscan

	out = hdbscan.fit(data, <optional>)
    fig = hdbscan.plot(out)

 INPUT:
   data:           numpy datamatrix numerical
                   rows    = features
                   colums  = samples
 OPTIONAL

  minclusters=     Integer: Minimum cluster size (only for hdbscan)
                   [2] (default)

  min_samples=     Integer: [0..1] Percentage of expected outliers among number of samples.
                   [0.05] (default)

  metric=          string: Define your input data as type [metrics.pairwise.calculate_distance] or a distance matrix if thats the case!
                   'euclidean' (default) squared euclidean distance
                   'precomputed' if input is a distance matrix!

  norm=            Boolean [0,1] (You may want to set this =0 using distance matrix as input)
                   [1]: Yes (default) 
                   [0]: No

  n_jobs=          Integer: The number of parallel jobs to run
                   [-1] ALL cpus (default)
                   [1]  Use a single core
                   
  showprogress=   Boolean [0,1]
                   [0]: No 
                   [1]: Some information about embedding
                   [2]: More information about embedding (default)

 OUTPUT
	output

 DESCRIPTION
   http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
   http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
   https://github.com/scikit-learn-contrib/hdbscan
   pip install hdbscan
   
   
 EXAMPLE
   import clusteval.hdbscan as hdbscan
   import numpy as np

   EXAMPLE 1
   from sklearn.datasets.samples_generator import make_blobs
   [X, labels_true] = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4,random_state=0)

   out = hdbscan.fit(X)
   out = hdbscan.plot(out)

 SEE ALSO
   import sklearn.cluster as cluster
"""
#print(__doc__)

#--------------------------------------------------------------------------
# Name        : hdbscan.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Nov. 2017
#--------------------------------------------------------------------------

#%% Libraries
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan as hdb
import seaborn as sns
import matplotlib.pyplot as plt

#%% Plot
def plot(out, width=15, height=8, verbose=3):
    model=out['model']
    if out['minclusters']==True:
        plt.subplots(figsize=(Param['width'],Param['height']))
        model.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80, edge_linewidth=2)

    plt.subplots(figsize=(width, height))
    model.single_linkage_tree_.plot(cmap='viridis', colorbar=True)

    plt.subplots(figsize=(width, height))
    model.condensed_tree_.plot()

    plt.subplots(figsize=(width, height))
    model.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
    
#%%
def fit(data, min_samples=0.01, metric='euclidean', norm=True, n_jobs=-1, minclusters=2, verbose=3):
	# DECLARATIONS
    out={}
    Param = {}
    Param['min_samples'] = min_samples
    Param['minclusters'] = minclusters
    Param['metric'] = metric
    Param['n_jobs'] = n_jobs
    Param['norm'] = norm
    Param['gen_min_span_tree'] = False
    Param['min_samples'] = np.int(np.floor(min_samples*data.shape[0])) # Set max. outliers

    # Transform data
    if Param['norm']:
        data = StandardScaler().fit_transform(data)
        
    # SET PARAMTERS FOR DBSCAN
    model = hdb.HDBSCAN(algorithm='best', metric=Param['metric'], min_samples=np.int(Param['min_samples']), core_dist_n_jobs=Param['n_jobs'], min_cluster_size=np.int(Param['minclusters']), p=None,gen_min_span_tree=Param['gen_min_span_tree'])
    model.fit(data) # Perform the clustering

    out['method']          ='hdbscan'
    out['labx']                = model.labels_        # Labels
    out['p']                   = model.probabilities_ # The strength with which each sample is a member of its assigned cluster. Noise points have probability zero; points in clusters have values assigned proportional to the degree that they persist as part of the cluster.
    out['cluster_persistence'] = model.cluster_persistence_ # A score of how persistent each cluster is. A score of 1.0 represents a perfectly stable cluster that persists over all distance scales, while a score of 0.0 represents a perfectly ephemeral cluster. These scores can be guage the relative coherence of the clusters output by the algorithm.
    out['outlier']             = model.outlier_scores_      # Outlier scores for clustered points; the larger the score the more outlier-like the point. Useful as an outlier detection technique. Based on the GLOSH algorithm by Campello, Moulavi, Zimek and Sander.
    # out2['predict'] = model.prediction_data_     # Cached data used for predicting the cluster labels of new or unseen points. Necessary only if you are using functions from hdbscan.prediction (see approximate_predict(), membership_vector(), and all_points_membership_vectors()).
    out['minclusters'] = Param['minclusters']
    out['model'] = model

    # Some info
    if verbose>=3:
        n_clusters = len(set(out['labx'])) - (1 if -1 in out['labx'] else 0)
        print('[HDBSCAN] Estimated number of clusters: %d' % n_clusters)

        if n_clusters!=data.shape[0] and n_clusters>1:
            print("[HDBSCAN] Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, out['labx']))
        
    return(out)
