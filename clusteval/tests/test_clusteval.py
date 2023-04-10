from clusteval import clusteval
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import unittest


class TestCLUSTEVAL(unittest.TestCase):

    def test_import_example(self):
        cl = clusteval()
        sizes = [(1000, 4), (891, 12), (649, 33), (128, 27), (4674, 9), (14850, 40), (999, 8)]
        datasets = ['sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail']
        for data, size in zip(datasets, sizes):
            df = cl.import_example(data=data)
            assert df.shape==size

    def test_fit(self):
        X, y_true = make_blobs(n_samples=500, centers=6, n_features=10)
        # X, y_true = make_blobs(n_samples=50, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4,random_state=0)

        # Set all parameters to be evaluated
        clusters = ['agglomerative', 'kmeans', 'dbscan']
        evaluates = ['silhouette', 'dbindex', 'derivative']
        metrics = ['euclidean', 'hamming']
        linkages = ['ward', 'single', 'complete']
        min_clusts = [1, 2, 10]
        max_clusts = [1, 10, 2]

        # Evaluate across all paramters
        out = parameter_gridtest(X, y_true, clusters, evaluates, metrics, linkages, min_clusts, max_clusts)

# %%
def parameter_gridtest(X, y_true, clusters, evaluates, metrics, linkages, min_clusts, max_clusts):
    random_state = 42
    out = []
    count = 0

    for cluster in clusters:
        for evaluate in evaluates:
            for metric in metrics:
                for linkage in linkages:
                    for min_clust in min_clusts:
                        for max_clust in max_clusts:
                            print(cluster)
                            print(evaluate)
                            print(metric)
                            print(linkage)
                            print(min_clust)
                            print(max_clust)

                            # cluster='agglomerative'
                            # evaluate='derivative'
                            # metric='euclidean'
                            # linkage='complete'
                            # min_clust=1
                            # max_clust=10

                            try:
                                status = 'OK'
                                ce = clusteval(evaluate=evaluate, cluster=cluster, metric=metric, linkage=linkage, min_clust=min_clust, max_clust=max_clust, verbose=3)
                                results = ce.fit(X)
                                # print('Clusters: %s' %(str(np.unique(results['labx']))))
                                # assert ce.plot()
                                # assert ce.scatter(X)
                                # assert ce.dendrogram()

                                if (ce.results['labx'] is not None) and (linkage!='single') and (min_clust < len(np.unique(y_true))) and (max_clust > len(np.unique(y_true))) and (metric=='euclidean'):
                                    print(len(np.unique(results['labx'])))
                                    print(len(np.unique(y_true)))
                                    assert len(np.unique(results['labx']))==len(np.unique(y_true))

                            except ValueError as err:
                                assert not 'clusteval' in err.args
                                status = err.args
                                print(err.args)
                                # ce.results=None

                            # out.append(ce.results)
                            count=count + 1

    print('Fin! Total number of models evaluated with different paramters: %.0d' %(count))
    return(pd.DataFrame(out))
