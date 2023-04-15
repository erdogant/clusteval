from clusteval import clusteval
from sklearn.datasets import make_blobs
from sklearn import cluster, datasets, mixture
import pandas as pd
import numpy as np
import unittest
import matplotlib.pyplot as plt


class TestCLUSTEVAL(unittest.TestCase):

    def test_different_X():
        # Generate random data
        n_samples = 1000
        X1, y1 = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
        X2, y2 = datasets.make_moons(n_samples=n_samples, noise=0.05)
        X3, y3 = make_blobs(n_samples=200, n_features=2, centers=2, random_state=1)

        c = np.random.multivariate_normal([40, 40], [[20, 1], [1, 30]], size=[200,])
        d = np.random.multivariate_normal([80, 80], [[30, 1], [1, 30]], size=[200,])
        e = np.random.multivariate_normal([0, 100], [[200, 1], [1, 100]], size=[200,])
        X3 = np.concatenate((X3, c, d, e), )
        y3 = np.concatenate((y3, len(c) * [2], len(c) * [3], len(c) * [4]), )

        X4, y4 = make_blobs(n_samples=n_samples, centers=4, n_features=4, cluster_std=0.5)
        X5, y5 = np.random.rand(n_samples, 2), None

        # scatterd(X1[:,0], X1[:,1],labels=y1, figsize=(15, 10))
        # scatterd(X2[:,0], X2[:,1],labels=y2, figsize=(15, 10))
        # scatterd(X3[:,0], X3[:,1],labels=y3, figsize=(15, 10))
        # scatterd(X4[:,0], X4[:,1],labels=y4, figsize=(15, 10))
        # scatterd(X5[:,0], X5[:,1],labels=y5, figsize=(15, 10))

        datas = [X1, X2, X3, X4, X5]
        methods = [['kmeans'], ['dbscan'], ['agglomerative', 'single'], ['agglomerative', 'complete'], ['agglomerative', 'average'], ['agglomerative', 'ward']]
        evaluations = ['silhouette', 'dbindex', 'derivative']

        for k, method in enumerate(methods):
            plt.figure()
            fig, axs = plt.subplots(len(datas), len(evaluations), figsize=(25, 22))
            fig.suptitle(method)
            plt.figure()
            fig2, axs2 = plt.subplots(len(datas), len(evaluations), figsize=(25, 22))
            fig2.suptitle(method)
            for j, X in enumerate(datas):
                for i, evaluate in enumerate(evaluations):
                    linkage = 'ward' if len(method)==1 else method[1]
                    ce = clusteval(evaluate=evaluate, cluster=method[0], metric='euclidean', linkage=linkage, max_clust=10)
                    results = ce.fit(X)
                    if results is not None:
                        # ce.plot()
                        # ce.scatter(X)
                        # ce.dendrogram()
                        ce.plot(title='', ax=axs[j][i], showfig=False)
                        axs2[j][i].scatter(X[:, 0], X[:, 1], c=results['labx'])
                        axs2[j][i].grid(True)
                        if j==0:
                            axs[j][i].set_title(evaluate)
                            axs2[j][i].set_title(evaluate)
                        if i==0:
                            axs[j][i].set_ylabel(method)
                            axs2[j][i].set_ylabel(method)

    def test_import_example(self):
        cl = clusteval()
        sizes = [(1000, 4), (891, 12), (649, 33), (128, 27), (4674, 9), (14850, 40), (999, 8)]
        datasets = ['sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail']
        for data, size in zip(datasets, sizes):
            df = cl.import_example(data=data)
            assert df.shape==size

    def test_fit(self):
        X, y_true = make_blobs(n_samples=500, centers=6, n_features=10, random_state=1)
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
                            # evaluate='dbindex'
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
