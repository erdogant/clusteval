# EXAMPLE
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from clusteval import clusteval

# import clusteval
# print(clusteval.__version__)
# print(dir(clusteval))

# %% Imort own data 
from df2onehot import df2onehot
from clusteval import clusteval
ce = clusteval()

url='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
df = ce.import_example(url=url)
# Add column names
df.columns=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','earnings']
# Set the following columns as floating type
cols_as_float = ['age','hours-per-week','capital-loss','capital-gain', 'fnlwgt']
df[cols_as_float]=df[cols_as_float].astype(float)
dfhot = df2onehot(df, excl_background=['0.0', 'None'], verbose=4)['onehot']

ce = clusteval(cluster='agglomerative', metric='hamming', linkage='complete', min_clust=7, verbose='info')
ce.fit(dfhot);
ce.plot()
ce.scatter(jitter=0.01)

from pca import pca
model = pca()
xycoord = model.fit_transform(dfhot)['PC'].values
ce.scatter(xycoord, jitter=0.05)

from sklearn.manifold import TSNE
xycoord = TSNE(n_components=2, init='random').fit_transform(dfhot)
ce.scatter(xycoord)
ce.enrichment()

# %% Enrichment analysis

from df2onehot import df2onehot
from clusteval import clusteval

ce = clusteval()
df = ce.import_example(data='titanic')
y = df['Survived'].values
df.drop(labels=['Survived', 'Name', 'Age', 'PassengerId', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)
dfhot = df2onehot(df, excl_background=['0.0', 'None'], verbose=4)['onehot']
X = dfhot.values

ce = clusteval(cluster='agglomerative', metric='hamming', linkage='complete', min_clust=7, verbose='info')
# ce = clusteval(cluster='agglomerative', linkage='complete', min_clust=7, max_clust=40, verbose='info')
# ce = clusteval(cluster='dbscan', linkage='complete', min_clust=7, max_clust=40, verbose='info')
# ce = clusteval(cluster='dbscan', metric='hamming', linkage='complete', min_clust=7, verbose='info')
ce.fit(dfhot);
ce.plot()
ce.scatter()
ce.scatter(embedding='tsne')
ce.enrichment(df)
ce.scatter(embedding='tsne')

from pca import pca
model = pca()
xycoord = model.fit_transform(X)['PC'].values
ce.scatter(X=xycoord)


from scatterd import scatterd
scatterd(xycoord[:,0], xycoord[:,1], labels=ce.results['labx']==16)
scatterd(xycoord[:,0], xycoord[:,1], labels=dfhot['Sex_female'])

# %%
# Import library
from clusteval import clusteval
# Initialize for DBSCAN and silhouette method
ce = clusteval(cluster='agglomerative', evaluate='silhouette', max_clust=10)
# Import example dataset
X, y = ce.import_example(data='blobs', params={'random_state':1})
# find optimal number of clusters
results = ce.fit(X)
# Make plot
ce.plot(figsize=(12, 7))
# Show scatterplot with silhouette scores
ce.scatter()
ce.plot_silhouette()

# Import library
from clusteval import clusteval
# Initialize
ce = clusteval(cluster='agglomerative', evaluate='dbindex', max_clust=10)
# Import example dataset
X, y = ce.import_example(data='blobs', params={'random_state':1})
# find optimal number of clusters
results = ce.fit(X)
# Make plot
ce.plot(figsize=(12, 8))
# Show scatterplot with silhouette scores
ce.scatter()
ce.plot_silhouette()

# Import library
from clusteval import clusteval
# Initialize
ce = clusteval(cluster='agglomerative', evaluate='derivative', max_clust=20)
# Import example dataset
X, y = ce.import_example(data='blobs', params={'random_state':1})
# find optimal number of clusters
results = ce.fit(X)
# Make plot
ce.plot(figsize=(12, 8))
# Show scatterplot with silhouette scores
ce.scatter()
ce.plot_silhouette()

# %%
X, y = ce.import_example(data='blobs', params={'random_state':1})
ce = clusteval(cluster='agglomerative', evaluate='silhouette')
results = ce.fit(X)
ce.plot()
ce.scatter()
ce.plot_silhouette()
cluster_labels = results['labx']


# %%
import clusteval
from scatterd import scatterd
import numpy as np

X, y = ce.import_example(data='blobs', params={'random_state': 1})

plt.figure(figsize=(15,10))
plt.scatter(X[:,0], X[:,1])
plt.grid(True); plt.xlabel('Feature 1'); plt.ylabel('Feature 2')

# X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

scatterd(X[:,0], X[:,1],labels=y, figsize=(15, 10))

plt.figure()
fig, axs = plt.subplots(2,4, figsize=(25,10))

# dbindex
results = clusteval.dbindex.fit(X)
_ = clusteval.dbindex.plot(results, title='dbindex', ax=axs[0][0], showfig=False)
axs[1][0].scatter(X[:,0], X[:,1],c=results['labx']);axs[1][0].grid(True)

# silhouette
results = clusteval.silhouette.fit(X)
_ = clusteval.silhouette.plot(results, title='silhouette', ax=axs[0][1], showfig=False)
axs[1][1].scatter(X[:,0], X[:,1],c=results['labx']);axs[1][1].grid(True)

# derivative
results = clusteval.derivative.fit(X)
_ = clusteval.derivative.plot(results, title='derivative', ax=axs[0][2], showfig=False)
axs[1][2].scatter(X[:,0], X[:,1],c=results['labx']);axs[1][2].grid(True)

# dbscan
results = clusteval.dbscan.fit(X)
_ = clusteval.dbscan.plot(results, title='dbscan', ax=axs[0][3], showfig=False)
axs[1][3].scatter(X[:,0], X[:,1],c=results['labx']);axs[1][3].grid(True)

# results = clusteval.hdbscan.fit(X)
# _ = clusteval.dbscan.plot(results, title='dbscan', ax=axs[0][3], showfig=False)
# axs[1][3].scatter(X[:,0], X[:,1],c=results['labx']);axs[1][3].grid(True)



# %%
from clusteval import clusteval
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


# %%
from df2onehot import df2onehot
from clusteval import clusteval

ce = clusteval()
df = ce.import_example(data='fifa')
df.drop(labels=['Fouls Committed', 'Date', 'Passes', 'Ball Possession %', 'Distance Covered (Kms)', 'Pass Accuracy %', 'Free Kicks', 'Attempts'], axis=1, inplace=True)
X = df2onehot(df, excl_background=['0.0', 'NaN', 'nan'], verbose=4)['onehot'].values

df = ce.import_example(data='retail')
df.drop(labels=['CustomerID', 'InvoiceNo', 'StockCode', 'Country', 'InvoiceDate', 'UnitPrice'], axis=1, inplace=True)
X = df2onehot(df, excl_background=['0.0', 'NaN', 'nan'], verbose=4)['onehot'].values


df = ce.import_example(data='titanic')
df.drop(labels=['Name', 'Age', 'PassengerId', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)
X = df2onehot(df, excl_background=['0.0', 'NaN', 'nan'], verbose=4)['onehot'].values

from clusteval import clusteval
ce = clusteval(cluster='agglomerative', metric='hamming', linkage='complete', min_clust=7, verbose=3)
ce = clusteval(cluster='agglomerative', metric='euclidean', linkage='complete', min_clust=7, verbose=3)
# ce = clusteval(cluster='dbscan', metric='hamming', linkage='complete', min_clust=7, verbose=3)
results = ce.fit(X)
ce.plot()
ce.scatter(embedding='tsne')
ce.plot_silhouette(jitter=0.01)

from pca import pca
model = pca()
xycoord = model.fit_transform(X)['PC'].values
ce.scatter(xycoord[:, :2], jitter=0.01)

# %%

from clusteval import clusteval
ce = clusteval()
df = ce.import_example(data='titanic')
df.drop(labels=['Name', 'Age', 'PassengerId', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)
# df = ce.import_example(data='student')
# df.drop(labels=['G1', 'G2', 'G3', 'address', 'age'], axis=1, inplace=True)
from df2onehot import df2onehot
dfhot = df2onehot(df, excl_background='0.0', verbose=4)['onehot']
X = dfhot.values

from clusteval import clusteval
ce = clusteval(cluster='agglomerative', metric='hamming', linkage='complete', min_clust=7, verbose=3)
ce = clusteval(cluster='agglomerative', linkage='complete', min_clust=7, max_clust=40, verbose=3)
ce = clusteval(cluster='dbscan', linkage='complete', min_clust=7, max_clust=40, verbose=3)
# ce = clusteval(cluster='dbscan', metric='hamming', linkage='complete', min_clust=7, verbose=3)
results = ce.fit(X)
ce.plot()

from pca import pca
model = pca()
xycoord = model.fit_transform(X)['PC'].values
ce.scatter(xycoord[:, :2], jitter=None)

from scatterd import scatterd
from sklearn.manifold import TSNE
xycoord = TSNE(n_components=2, init='random').fit_transform(X)
scatterd(xycoord[:,0], xycoord[:,1], labels=ce.results['labx'])


# %%
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from clusteval import clusteval

X, y = make_blobs(n_samples=600, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4, random_state=0)
# plt.figure(figsize=(15, 10));plt.scatter(X[:,0], X[:,1], c=y);plt.grid(True);plt.xlabel('Feature 1');plt.ylabel('Feature 2')
# plt.figure(figsize=(15, 10));plt.scatter(X[:,0], X[:,1], c='k');plt.grid(True);plt.xlabel('Feature 1');plt.ylabel('Feature 2')

# ce = clusteval(cluster='hdbscan')
# ce = clusteval(cluster='kmeans')
ce = clusteval(cluster='agglomerative', evaluate='derivative')
results = ce.fit(X)

ce.plot(savefig={'fname':'test_plot.png'})
ce.scatter(X, dot_size=100, savefig={'fname':'test_scatter.png'})
ce.dendrogram(savefig={'fname':'test_dendrogram.png'})



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
df = clusteval.import_example(data='titanic')
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

# for i in zip(results['labx'], results_dendro['labx']):
#     assert np.all(np.logical_and(np.where(results['labx']==i[0])[0]+1, np.where(results_dendro['labx']==i[1])[0]+1))

# %% hdbscan
from clusteval import clusteval
ce = clusteval(cluster='hdbscan')
results = ce.fit(X)
ce.plot(savefig={'fname':'hdbscan'})
ce.scatter(X)
results_dendro = ce.dendrogram(figsize=(15, 8), orientation='top')

# for i in zip(results['labx'], results_dendro['labx']):
#     assert np.all(np.logical_and(np.where(results['labx']==i[0])[0]+1, np.where(results_dendro['labx']==i[1])[0]+1))

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
_ = clusteval.dbindex.plot(results, title='dbindex', ax=axs[0][0], showfig=False)
axs[1][0].scatter(X[:,0], X[:,1],c=results['labx']);axs[1][0].grid(True)

# silhouette
results = clusteval.silhouette.fit(X)
_ = clusteval.silhouette.plot(results, title='silhouette', ax=axs[0][1], showfig=False)
axs[1][1].scatter(X[:,0], X[:,1],c=results['labx']);axs[1][1].grid(True)

# derivative
results = clusteval.derivative.fit(X)
_ = clusteval.derivative.plot(results, title='derivative', ax=axs[0][2], showfig=False)
axs[1][2].scatter(X[:,0], X[:,1],c=results['labx']);axs[1][2].grid(True)

# dbscan
results = clusteval.dbscan.fit(X)
_ = clusteval.dbscan.plot(results, title='dbscan', ax=axs[0][3], showfig=False)
axs[1][3].scatter(X[:,0], X[:,1],c=results['labx']);axs[1][3].grid(True)

plt.show()
