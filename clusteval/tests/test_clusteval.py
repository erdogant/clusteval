from clusteval import clusteval
from sklearn.datasets import make_blobs
from sklearn import cluster, datasets, mixture
import pandas as pd
import numpy as np
import unittest
import matplotlib.pyplot as plt
import datazets as dz


class TestCLUSTEVAL(unittest.TestCase):

    def test_different_X(self):
        ce = clusteval()

        # Generate random data
        df1 = dz.get(data='circles')
        df2 = ce.import_example(data='moons')
        df3 = ce.import_example(data='anisotropic')
        df4 = ce.import_example(data='densities')
        df5 = ce.import_example(data='blobs', params={'random_state': 1})
        df6 = ce.import_example(data='globular')
        df7 = ce.import_example(data='uniform')

        dfs = [df1, df2, df3, df5, df4, df6, df7]
        titles = ['Noisy Circles', 'Noisy Moons', 'Anisotropic', 'Blobs', 'Different Densities', 'Globular', 'No Structure']
        fig, axs = plt.subplots(1, 7, figsize=(60, 8), dpi=100)
        for i, df in enumerate(dfs):
            axs[i].scatter(df.loc[:, 0], df.loc[:, 1], c=df.index.values)
            axs[i].set_title(titles[i], fontsize=33)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_xticklabels([])
            axs[i].set_yticklabels([])

        methods = [['kmeans'], ['dbscan'], ['agglomerative', 'single'], ['agglomerative', 'complete'], ['agglomerative', 'ward']]
        evaluations = ['silhouette', 'dbindex', 'derivative']
        font_properties = {'size_x_axis': 18, 'size_y_axis': 18, 'size_title': 26}
        
        for k, evaluate in enumerate(evaluations):
            fig, axs = plt.subplots(len(dfs), len(methods), figsize=(60, 60), dpi=75)
            fig.suptitle(evaluate.title(), fontsize=36)
            fig2, axs2 = plt.subplots(len(dfs), len(methods), figsize=(60, 60), dpi=75)
            fig2.suptitle(evaluate.title(), fontsize=36)
        
            # Run over data
            for j, X in enumerate(dfs):
                for i, method in enumerate(methods):
                    linkage = 'ward' if len(method)==1 else method[1]
                    ce = clusteval(evaluate=evaluate, cluster=method[0], metric='euclidean', linkage=linkage, max_clust=10)
                    results = ce.fit(X)
                    if (results is not None):
                        ce.plot(title='', ax=axs[j][i], showfig=False, xlabel='Nr. Clusters', ylabel='', font_properties=font_properties)
                        axs2[j][i].scatter(X.loc[:, 0], X.loc[:, 1], c=results['labx'])
                        axs2[j][i].grid(True)
                        if j==0:
                            axs[j][i].set_title(' '.join(method).title(), fontsize=42)
                            axs2[j][i].set_title(' '.join(method).title(), fontsize=42)


    def test_import_example(self):
        cl = clusteval()
        sizes = [(1000, 4), (891, 12), (649, 33), (128, 27), (4674, 9), (59400, 41), (999, 8)]
        datasets = ['sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'marketing_retail']
        for data, size in zip(datasets, sizes):
            df = cl.import_example(data=data)
            assert df.shape==size

    def test_fit(self):
        ce = clusteval()
        df = dz.get(data='blobs', params={'random_state': 1})

        # Set all parameters to be evaluated
        clusters = ['dbscan', 'agglomerative', 'kmeans']
        evaluates = ['dbindex', 'derivative', 'silhouette']
        metrics = ['euclidean', 'hamming']
        linkages = ['ward', 'single', 'complete']
        min_clusts = [1, 2, 10]
        max_clusts = [1, 10, 2]

        # Evaluate across all paramters
        out = parameter_gridtest(df, df.index.values, clusters, evaluates, metrics, linkages, min_clusts, max_clusts)

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
                            # print(cluster)
                            # print(evaluate)
                            # print(metric)
                            # print(linkage)
                            # print(min_clust)
                            # print(max_clust)

                            # cluster='agglomerative'
                            # evaluate='dbindex'
                            # metric='euclidean'
                            # linkage='complete'
                            # min_clust=1
                            # max_clust=10

                            try:
                                status = 'OK'
                                ce = clusteval(evaluate=evaluate, cluster=cluster, metric=metric, linkage=linkage, min_clust=min_clust, max_clust=max_clust, verbose=2)
                                results = ce.fit(X)
                                # print('Clusters: %s' %(str(np.unique(results['labx']))))
                                # assert ce.plot()
                                # assert ce.scatter(X)
                                # assert ce.dendrogram()

                                if (results is not None) and (ce.results['labx'] is not None) and (linkage!='single') and (min_clust < len(np.unique(y_true))) and (max_clust > len(np.unique(y_true))) and (metric=='euclidean'):
                                    # print(len(np.unique(results['labx'])))
                                    # print(len(np.unique(y_true)))
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
