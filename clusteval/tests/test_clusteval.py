from clusteval import clusteval
from sklearn.datasets import make_blobs
from sklearn import cluster, datasets, mixture
import pandas as pd
import numpy as np
import unittest
import matplotlib.pyplot as plt


class TestCLUSTEVAL(unittest.TestCase):

    def test_different_X(self):
        ce = clusteval()
        
        # Generate random data
        X1, y1 = ce.import_example(data='circles')
        X2, y2 = ce.import_example(data='moons')
        X3, y3 = ce.import_example(data='anisotropic')
        X4, y4 = ce.import_example(data='densities')
        X5, y5 = ce.import_example(data='blobs', params={'random_state': 1})
        X6, y6 = ce.import_example(data='globular')
        X7, y7 = ce.import_example(data='uniform')

        datas = [X1, X2, X3, X5, X4, X6, X7]
        ys = [y1, y2, y3, y5, y4, y6, y7]
        titles = ['Noisy Circles', 'Noisy Moons', 'Anisotropic', 'Blobs', 'Different Densities', 'Globular', 'No Structure']
        fig, axs = plt.subplots(1, 7, figsize=(60, 8), dpi=100)
        for i, data in enumerate(zip(datas, ys)):
            axs[i].scatter(data[0][:, 0], data[0][:, 1], c=data[1])
            axs[i].set_title(titles[i], fontsize=33)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_xticklabels([])
            axs[i].set_yticklabels([])
        
        methods = [['kmeans'], ['dbscan'], ['agglomerative', 'single'], ['agglomerative', 'complete'], ['agglomerative', 'ward']]
        evaluations = ['silhouette', 'dbindex', 'derivative']
        font_properties = {'size_x_axis': 18, 'size_y_axis': 18, 'size_title': 26}
        
        for k, evaluate in enumerate(evaluations):
            fig, axs = plt.subplots(len(datas), len(methods), figsize=(60, 60), dpi=75)
            fig.suptitle(evaluate.title(), fontsize=36)
            fig2, axs2 = plt.subplots(len(datas), len(methods), figsize=(60, 60), dpi=75)
            fig2.suptitle(evaluate.title(), fontsize=36)
        
            # Run over data
            for j, X in enumerate(datas):
                for i, method in enumerate(methods):
                    linkage = 'ward' if len(method)==1 else method[1]
                    ce = clusteval(evaluate=evaluate, cluster=method[0], metric='euclidean', linkage=linkage, max_clust=10)
                    results = ce.fit(X)
                    if (results is not None):
                        ce.plot(title='', ax=axs[j][i], showfig=False, xlabel='Nr. Clusters', ylabel='', font_properties=font_properties)
                        axs2[j][i].scatter(X[:, 0], X[:, 1], c=results['labx'])
                        axs2[j][i].grid(True)
                        if j==0:
                            axs[j][i].set_title(' '.join(method).title(), fontsize=42)
                            axs2[j][i].set_title(' '.join(method).title(), fontsize=42)


    def test_import_example(self):
        cl = clusteval()
        sizes = [(1000, 4), (891, 12), (649, 33), (128, 27), (4674, 9), (14850, 40), (999, 8)]
        datasets = ['sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail']
        for data, size in zip(datasets, sizes):
            df = cl.import_example(data=data)
            assert df.shape==size

    def test_fit(self):
        ce = clusteval()
        X, y_true = ce.import_example(data='blobs', params={'random_state': 1})

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

                                if (ce.results is not None) and (ce.results['labx'] is not None) and (linkage!='single') and (min_clust < len(np.unique(y_true))) and (max_clust > len(np.unique(y_true))) and (metric=='euclidean'):
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
