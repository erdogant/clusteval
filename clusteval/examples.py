# EXAMPLE
import clusteval
from sklearn.datasets import make_blobs


# %%
[X, labels_true] = make_blobs(n_samples=750, centers=4, n_features=2, cluster_std=0.5)

out = clusteval.fit(X, method='silhouette')
out = clusteval.fit(X, method='dbindex')
out = clusteval.fit(X, method='derivative')
out = clusteval.fit(X, method='hdbscan')
out = clusteval.fit(X, method='dbscan')

clusteval.plot(out, X)
