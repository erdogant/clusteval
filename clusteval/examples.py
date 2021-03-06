# EXAMPLE
import clusteval
from sklearn.datasets import make_blobs
print(clusteval.__version__)
print(dir(clusteval))

from clusteval import clusteval

# %% Example with titanic dataset and one-hot array
import clusteval
df = clusteval.import_example()
del df['PassengerId']
import df2onehot
dfhot = df2onehot.df2onehot(df, excl_background='0.0')['onehot']
X = dfhot.values

from clusteval import clusteval
ce = clusteval(cluster='agglomerative', metric='hamming', linkage='complete', min_clust=7, verbose=3)
# ce = clusteval(cluster='dbscan', metric='hamming', linkage='complete', min_clust=7, verbose=3)
results = ce.fit(X)
ce.plot()
ce.scatter(X)

# %% Example with textual data
# import clusteval
# df = clusteval.import_example(data='retail')
# corpus = df['Description'].values

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?',
          'This about cats',
          'This about red cars',
          'hello world',
          ]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())

svd = TruncatedSVD(n_components=2)
normalizer = Normalizer(copy=False, norm='l2')
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)

from clusteval import clusteval
ce = clusteval(cluster='dbscan')
ce.fit(X)
ce.plot()
ce.scatter(X)
ce.dendrogram(labels=corpus)

# %% Generate dataset
X, labels_true = make_blobs(n_samples=50, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4,random_state=0)
# [X, labels_true] = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]], cluster_std=0.4,random_state=0)
# X, labels_true = make_blobs(n_samples=750, centers=4, n_features=6, cluster_std=0.5)
# X, labels_true = make_blobs(n_samples=750, centers=6, n_features=10)

# %% Silhouette
# ce = clusteval(method='silhouette', metric='kmeans', savemem=True)
from clusteval import clusteval
ce = clusteval(method='silhouette', verbose=3)
results = ce.fit(X)
ce.plot()
ce.scatter(X)

results = ce.dendrogram()
results = ce.dendrogram(max_d=9)
results = ce.dendrogram(X=X, linkage='single', metric='euclidean')
results = ce.dendrogram(X=X, linkage='single', metric='euclidean', max_d=0.8)
results = ce.dendrogram(X=X, linkage='complete', metric='euclidean', max_d=2)
results = ce.dendrogram(figsize=(15,8), show_contracted=True)

results['labx']
results['order_rows']

# %% Silhouette
from clusteval import clusteval
ce = clusteval(method='silhouette')
results = ce.fit(X)
ce.plot()
ce.scatter(X)
results = ce.dendrogram()

# %% dbindex
from clusteval import clusteval
ce = clusteval(method='dbindex')
results = ce.fit(X)
ce.plot()
ce.scatter(X)
results = ce.dendrogram()

# %% derivative
from clusteval import clusteval
ce = clusteval(method='derivative')
results = ce.fit(X)
ce.plot()
ce.scatter(X)

# %% dbscan
from clusteval import clusteval
ce = clusteval(cluster='dbscan')
results = ce.fit(X)
ce.plot()
ce.scatter(X)
results = ce.dendrogram()

# %% hdbscan
from clusteval import clusteval
ce = clusteval(cluster='hdbscan')
results = ce.fit(X)
ce.plot()
ce.scatter(X)
results = ce.dendrogram(figsize=(15,8), orientation='top')

# %% Directly use the dbindex method
import clusteval
from sklearn.datasets import make_blobs
X, labels_true = make_blobs(n_samples=750, centers=6, n_features=10)

# dbindex
results = clusteval.dbindex.fit(X)
fig,ax = clusteval.dbindex.plot(results)

# silhouette
results = clusteval.silhouette.fit(X)
fig,ax = clusteval.silhouette.plot(results)

# derivative
results = clusteval.derivative.fit(X)
fig,ax = clusteval.derivative.plot(results)

# dbscan
results = clusteval.dbscan.fit(X)
fig,ax1,ax2 = clusteval.dbscan.plot(results)
