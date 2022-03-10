# EXAMPLE
from sklearn.datasets import make_blobs
# import clusteval
# print(clusteval.__version__)
# print(dir(clusteval))

from clusteval import clusteval


# %%
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from clusteval import clusteval

X, y = make_blobs(n_samples=600, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4, random_state=0)
# plt.figure(figsize=(15, 10));plt.scatter(X[:,0], X[:,1], c=y);plt.grid(True);plt.xlabel('Feature 1');plt.ylabel('Feature 2')
# plt.figure(figsize=(15, 10));plt.scatter(X[:,0], X[:,1], c='k');plt.grid(True);plt.xlabel('Feature 1');plt.ylabel('Feature 2')

# ce = clusteval(cluster='dbscan')
# ce = clusteval(cluster='kmeans')
ce = clusteval(cluster='agglomerative', evaluate='derivative')
results = ce.fit(X)
ce.plot()
ce.scatter(X, dot_size=100)



# %%
import numpy as np
from sklearn.datasets import make_blobs
from clusteval import clusteval

X, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=1)
c = np.random.multivariate_normal([40, 40], [[20, 1], [1, 30]], size=[200,])
d = np.random.multivariate_normal([80, 80], [[30, 1], [1, 30]], size=[200,])
e = np.random.multivariate_normal([0, 100], [[200, 1], [1, 100]], size=[200,])
X = np.concatenate((X, c, d, e),)
y = np.concatenate((y, len(c)*[2], len(c)*[3], len(c)*[4]),)

plt.figure(figsize=(15, 10));plt.scatter(X[:,0], X[:,1], c=y);plt.grid(True);plt.xlabel('Feature 1');plt.ylabel('Feature 2')

# Evaluate
# ce = clusteval(cluster='dbscan', params_dbscan={'epsres' :100, 'norm':True})
# ce = clusteval(cluster='dbscan')
# ce = clusteval(cluster='kmeans')

ce = clusteval(cluster='agglomerative', evaluate='dbindex')
results = ce.fit(X)
ce.plot()
ce.scatter(X)

# Set the evaluation method
# ce = clusteval(cluster='hdbscan')
# results = ce.fit(X)
# ce.plot()
# ce.scatter(X)

# %% Check
from sklearn.datasets import make_blobs

from clusteval import clusteval
ce = clusteval(cluster='dbscan')
X, labels_true = make_blobs(n_samples=50, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4,random_state=0)
results = ce.fit(X)
ce.plot()
ce.scatter(X)
cluster_labels = results['labx']

# %% Example with titanic dataset and one-hot array
import clusteval
df = clusteval.import_example()
del df['PassengerId']
from df2onehot import df2onehot
dfhot = df2onehot(df, excl_background='0.0')['onehot']
X = dfhot.values

from clusteval import clusteval
ce = clusteval(cluster='agglomerative', metric='hamming', linkage='complete', min_clust=7, verbose=3)
# ce = clusteval(cluster='dbscan', metric='hamming', linkage='complete', min_clust=7, verbose=3)
results = ce.fit(X)
ce.plot()
ce.scatter(X)

# %%
# from s_dbw import S_Dbw
# score = S_Dbw(X, df['Survived'].values, centers_id=None, method='Tong', alg_noise='bind',centr='mean', nearest_centr=True, metric='euclidean')

# from s_dbw import SD
# score = SD(X, df['Survived'].values, k=5, centers_id=None,  alg_noise='bind',centr='mean', nearest_centr=True, metric='euclidean')

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
# ce = clusteval(evaluate='silhouette', metric='kmeans', savemem=True)
from clusteval import clusteval
ce = clusteval(evaluate='silhouette', verbose=3)
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
ce = clusteval(evaluate='silhouette')
results = ce.fit(X)
ce.plot()
ce.scatter(X)
results_dendro = ce.dendrogram()

for i in zip(results['labx'], results_dendro['labx']):
    if not np.all(np.logical_and(np.where(results['labx']==i[0])[0]+1, np.where(results_dendro['labx']==i[1])[0]+1)):
        print('error')

# %% dbindex
from clusteval import clusteval
ce = clusteval(evaluate='dbindex')
results = ce.fit(X)
ce.plot()
ce.scatter(X)
results_dendro = ce.dendrogram()

results['labx']
results_dendro['labx']

for i in zip(results['labx'], results_dendro['labx']):
    assert np.all(np.logical_and(np.where(results['labx']==i[0])[0]+1, np.where(results_dendro['labx']==i[1])[0]+1))

# %% derivative
from clusteval import clusteval
ce = clusteval(evaluate='derivative')
results = ce.fit(X)
ce.plot()
ce.scatter(X)
results_dendro = ce.dendrogram()

np.unique(results_dendro['labx'])
np.unique(results['labx'])

for i in zip(results['labx'], results_dendro['labx']):
    assert np.all(np.logical_and(np.where(results['labx']==i[0])[0]+1, np.where(results_dendro['labx']==i[1])[0]+1))


# %% dbscan
from clusteval import clusteval
ce = clusteval(cluster='dbscan')
results = ce.fit(X)
ce.plot()
ce.scatter(X)
results_dendro = ce.dendrogram()

for i in zip(results['labx'], results_dendro['labx']):
    assert np.all(np.logical_and(np.where(results['labx']==i[0])[0]+1, np.where(results_dendro['labx']==i[1])[0]+1))

# %% hdbscan
from clusteval import clusteval
ce = clusteval(cluster='hdbscan')
results = ce.fit(X)
ce.plot()
ce.scatter(X)
results_dendro = ce.dendrogram(figsize=(15,8), orientation='top')

for i in zip(results['labx'], results_dendro['labx']):
    assert np.all(np.logical_and(np.where(results['labx']==i[0])[0]+1, np.where(results_dendro['labx']==i[1])[0]+1))

# %% Directly use the dbindex method
import clusteval
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
X, labels_true = make_blobs(n_samples=750, centers=4, n_features=10)


X, labx = make_blobs(n_samples=200, n_features=2, centers=2, random_state=1)
c = np.random.multivariate_normal([40, 40], [[20, 1], [1, 30]], size=[200,])
d = np.random.multivariate_normal([80, 80], [[30, 1], [1, 30]], size=[200,])
e = np.random.multivariate_normal([0, 100], [[200, 1], [1, 100]], size=[200,])
X = np.concatenate((X, c, d, e),)


plt.figure()
fig, axs = plt.subplots(2,4, figsize=(25,10))

# dbindex
results = clusteval.dbindex.fit(X)
_ = clusteval.dbindex.plot(results, title='dbindex', ax=axs[0][0], visible=False)
axs[1][0].scatter(X[:,0], X[:,1],c=results['labx']);axs[1][0].grid(True)

# silhouette
results = clusteval.silhouette.fit(X)
_ = clusteval.silhouette.plot(results, title='silhouette', ax=axs[0][1], visible=False)
axs[1][1].scatter(X[:,0], X[:,1],c=results['labx']);axs[1][1].grid(True)

# derivative
results = clusteval.derivative.fit(X)
_ = clusteval.derivative.plot(results, title='derivative', ax=axs[0][2], visible=False)
axs[1][2].scatter(X[:,0], X[:,1],c=results['labx']);axs[1][2].grid(True)

# dbscan
results = clusteval.dbscan.fit(X)
_ = clusteval.dbscan.plot(results, title='dbscan', ax=axs[0][3], visible=False)
axs[1][3].scatter(X[:,0], X[:,1],c=results['labx']);axs[1][3].grid(True)

plt.show()
