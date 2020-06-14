# EXAMPLE
import clusteval
from sklearn.datasets import make_blobs
print(clusteval.__version__)
print(dir(clusteval))

from clusteval import clusteval

# %% Generate dataset
X, labels_true = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4,random_state=0)
# [X, labels_true] = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]], cluster_std=0.4,random_state=0)
# X, labels_true = make_blobs(n_samples=750, centers=4, n_features=6, cluster_std=0.5)
# X, labels_true = make_blobs(n_samples=750, centers=6, n_features=10)

# %% Silhouette
# ce = clusteval(method='silhouette', metric='kmeans', savemem=True)
ce = clusteval(method='silhouette')
results = ce.fit(X)
ce.plot()
ce.scatter(X)

# %% dbscan
ce = clusteval(method='dbscan')
results = ce.fit(X)
ce.plot()
ce.scatter(X)

# %% dbindex
ce = clusteval(method='dbindex')
results = ce.fit(X)
ce.plot()
ce.scatter(X)

# %% derivative
ce = clusteval(method='derivative')
results = ce.fit(X)
ce.plot()
ce.scatter(X)

# %% hdbscan
ce = clusteval(method='hdbscan')
results = ce.fit(X)
ce.plot()
ce.scatter(X)

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
