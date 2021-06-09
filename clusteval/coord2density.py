""" This function computes the density of each point

	A= coord2density(data, <optional>)

 INPUT:
   X:              datamatrix
                   rows    = features
                   colums  = samples
 OPTIONAL

   kernel:         String: The kernel to use
                   'gaussian' (default)
                   'tophat'
                   'epanechnikov'
                   'exponential'
                   'linear'
                   'cosine'

   verbose:        Integer [0..5] if verbose >= DEBUG: print('debug message')
                   0: (default)
                   1: ERROR
                   2: WARN
                   3: INFO
                   4: DEBUG
                   
 OUTPUT
	output

 DESCRIPTION
   Short description what your function does and how it is processed

 EXAMPLE
   from sklearn.datasets.samples_generator import make_blobs
   from VIZ.scatter import scatter
   from TRANSFORMERS.coord2density import coord2density

   [X, labx] = make_blobs(n_samples=1000, centers=[[1, 1], [-1, -1]], cluster_std=0.4,random_state=0)
   dens = coord2density(X, showfig=True)
   
   out = scatter(X[:,0],X[:,1],density=0, colors=dens, size=25)


 SEE ALSO
"""

#--------------------------------------------------------------------------
# Name        : coord2density.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : April. 2019
#--------------------------------------------------------------------------

#%% Libraries
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

#%%
def coord2density(X, kernel='gaussian', metric='euclidean', showfig=False, verbose=3):
    config = dict()
    config['verbose'] = verbose

    kde = KernelDensity(kernel=kernel, metric=metric, bandwidth=0.2).fit(X)
    dens = kde.score_samples(X)

    if showfig:
        plt.figure(figsize=(8,8))
        plt.scatter(X[:, 0], X[:, 1], c=dens);        

    return(dens)
