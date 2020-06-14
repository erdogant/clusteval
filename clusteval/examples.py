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

# %% DBSCAN

ce = clusteval(method='dbscan')
results = ce.fit(X)
ce.plot()
ce.scatter(X)

# %%
ce = clusteval(method='dbindex')
results = ce.fit(X)
ce.plot()
ce.scatter(X)

# %%
ce = clusteval(method='derivative')
results = ce.fit(X)
ce.plot()
ce.scatter(X)

# %%

ce = clusteval(method='hdbscan')
results = ce.fit(X)
ce.plot()
ce.scatter(X)
