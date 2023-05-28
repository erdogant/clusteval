# -*- coding: utf-8 -*-
"""
Created on May 11, 2023

@author: Erdogan Taskesen
"""

#%% Import data set
from clusteval import clusteval
from df2onehot import df2onehot

# Load data set
ce = clusteval()
df = ce.import_example(data='ds_salaries')
X = df.copy()

# Pre processing
y = X['salary_in_usd'].values
X.drop(labels=['salary_in_usd','salary', 'salary_currency'], axis=1, inplace=True)
# Create one-hot data set
X = df2onehot(X, verbose=4)['onehot']

df.drop(labels=['salary', 'salary_currency'], axis=1, inplace=True)
#
ce = clusteval(evaluate='silhouette',
               cluster='agglomerative',
               metric='hamming',  # euclidean, hamming
               linkage='complete',
               min_clust=2,
               normalize=False,
               verbose='info')

ce.fit(X)
ce.plot()
# Plots first two features
ce.scatter(jitter=0.01)
ce.plot_silhouette()
ce.plot_silhouette(embedding='tsne')
ce.scatter(embedding='tsne', fontsize=26)
ce.dendrogram()

ce.enrichment(df)
ce.scatter(embedding='tsne', n_feat=3, fontcolor=None, fontsize=12)
ce.scatter(embedding='tsne', n_feat=3, fontcolor='k', fontsize=12)


# %%
from clusteval.utils import normalize_size
from sklearn.manifold import TSNE
xycoord = TSNE(n_components=2, init='random', perplexity=30).fit_transform(X.values)
# ce = clusteval(cluster='dbscan', min_clust=2, verbose='info')
ce = clusteval(evaluate='silhouette', cluster='agglomerative', linkage='complete', min_clust=5, max_clust=20, verbose='info')
ce.fit(xycoord)

ce.plot()
ce.plot_silhouette()
ce.enrichment(df)

yscaled = normalize_size(y.reshape(-1, 1), minscale=5, maxscale=300, scaler='minmax')
ce.scatter(n_feat=6, density=True, params_scatterd={'edgecolor': '#000000', 'gradient': None}, s=yscaled, fontsize=10)
ce.scatter(n_feat=6, density=True, params_scatterd={'edgecolor': None, 'gradient': '#FFFFFF'}, s=0, fontsize=10)
ce.scatter(n_feat=6, density=False, params_scatterd={'edgecolor': None, 'gradient': '#FFFFFF'}, s=yscaled, fontsize=10)


from scatterd import scatterd
# scatterd(xycoord[:,0], xycoord[:,1], labels=ce.results['labx'], cmap='tab20c')
scatterd(xycoord[:,0], xycoord[:,1], labels=df['Browser'], fontcolor='k', cmap='tab20c')
scatterd(xycoord[:,0], xycoord[:,1], labels=dfhot['TrafficType_11.0'], fontcolor='k', cmap='tab20c')
scatterd(xycoord[:,0], xycoord[:,1], labels=dfhot['Browser_2.0'], fontcolor='k', cmap='tab20c')
