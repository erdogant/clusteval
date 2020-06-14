# EXAMPLE
import clusteval
from sklearn.datasets import make_blobs
print(clusteval.__version__)
print(dir(clusteval))

from clusteval import clusteval

# %% Silhouette
X, labels_true = make_blobs(n_samples=750, centers=4, n_features=6, cluster_std=0.5)

ce1 = clusteval(method='silhouette')
# ce1 = clusteval(method='silhouette', metric='kmeans', savemem=True)
out1 = ce1.fit(X)
ce1.plot()
ce1.scatter(X)

# %% DBSCAN
X, labels_true = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4,random_state=0)
# [X, labels_true] = make_blobs(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]], cluster_std=0.4,random_state=0)

ce4 = clusteval(method='dbscan')
out4 = ce4.fit(X)
ce4.plot()
ce4.scatter(X)

# %%
X, labels_true = make_blobs(n_samples=750, centers=6, n_features=10)

ce2 = clusteval(method='dbindex')
out2 = ce2.fit(X)
ce2.plot()
ce2.scatter(X)

# %%
ce3 = clusteval(method='derivative')
out3 = ce3.fit(X)
ce3.plot(X)

# %%

ce5 = clusteval(method='hdbscan')
out5 = ce5.fit(X)
ce5.plot(X)
