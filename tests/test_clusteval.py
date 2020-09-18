from clusteval import clusteval
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np


# %%
def test_fit():
    X, y = make_blobs(n_samples=50, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4,random_state=0)

    # Set all parameters to be evaluated
    clusters = ['agglomerative', 'kmeans', 'dbscan']
    methods = ['silhouette', 'dbindex', 'derivative']
    metrics = ['euclidean', 'hamming']
    linkages = ['ward', 'single', 'complete']
    minclusters = [1, 2, 10]
    maxclusters = [1, 10, 2]

    # Evaluate across all paramters
    out = parameter_gridtest(X, y, clusters, methods, metrics, linkages, minclusters, maxclusters)

# %%
def parameter_gridtest(X, y, clusters, methods, metrics, linkages, minclusters, maxclusters):
    random_state = 42
    out = []
    count = 0

    for cluster in clusters:
        for method in methods:
            for metric in metrics:
                for linkage in linkages:
                    for mincluster in minclusters:
                        for maxcluster in maxclusters:
                            print(cluster)
                            print(method)
                            print(metric)
                            print(linkage)
                            print(mincluster)
                            print(maxcluster)

                            cluster='agglomerative'
                            method='derivative'
                            metric='hamming'
                            linkage='single'
                            mincluster=None
                            maxcluster=None

                            try:
                                status = 'OK'
                                ce = clusteval(method=method, cluster=cluster, metric=metric, linkage=linkage, minclusters=mincluster, maxclusters=maxcluster, verbose=3)
                                results = ce.fit(X)
                                assert ce.plot()
                                assert ce.scatter(X)
                                assert ce.dendrogram()
                            except ValueError as err:
                                assert not 'clusteval' in err.args
                                status = err.args
                                print(err.args)
                                # ce.results=None
    
                            # out.append(ce.results)
                            count=count+1

    print('Fin! Total number of models evaluated with different paramters: %.0d' %(count))
    return(pd.DataFrame(out))
