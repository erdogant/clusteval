""" Plotting clusters with silhoutte analysis

	A= silhouette_plot(data, cluster_labels)

 INPUT:
   data:           datamatrix
                   rows    = features
                   colums  = samples

 
   cluster_labels: numpy array


 OUTPUT
	output

 DESCRIPTION
   Plotting clusters with silhoutte analysis
   http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
   
   
 EXAMPLE
   from clusteval.silhouette_plot import silhouette_plot

   from sklearn.datasets.samples_generator import make_blobs
   [X, labels_true] = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]], cluster_std=0.4,random_state=0)
   import clusteval.dbscan as dbscan
   out = dbscan.fit(X)
   silhouette_plot(X,out['labx'])

"""
#print(__doc__)

#--------------------------------------------------------------------------
# Name        : silhouette_plot.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Nov. 2017
#--------------------------------------------------------------------------
#%% Libraries
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np

#%%
def silhouette_plot(data, cluster_labels):
    #%% The silhouette_score gives the average value for all the samples. This gives a perspective into the density and separation of the formed clusters
#    n_clusters = len(np.unique(cluster_labels))
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    silhouette_avg=silhouette_score(data, cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    #%% Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    #%% Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    #%
    y_lower = 10
    uiclust = np.unique(cluster_labels)

    for i in range(0,len(uiclust)):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == uiclust[i]]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.Set2(float(i) / n_clusters)

        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
#        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=getcolors[i], edgecolor=getcolors[i], alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(uiclust[i]))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    #%%
    ax1.set_title("The silhouette plot for the various clusters")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.grid(color='grey', linestyle='--', linewidth=0.2)

    # 2nd Plot showing the actual clusters formed
    color = cm.Set2(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(data[:, 0], data[:, 1], marker='.', s=30, lw=0, alpha=0.8,c=color, edgecolor='k')
    ax2.grid(color='grey', linestyle='--', linewidth=0.2)
    ax2.set_title("Clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(("Silhouette analysis for clustering on sample data with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
    plt.show()
