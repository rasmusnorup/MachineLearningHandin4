import sklearn.datasets
import numpy as np
X, y = sklearn.datasets.load_iris(True)
X = X[:,0:2] # reduce to 2d so you can plot if you want
print(X.shape, y.shape)

def silhouette(data, clustering):
    n, d = data.shape
    k = np.unique(clustering)[-1]+1

    # YOUR CODE HERE
    silh = None

    for i in range(n):
        sameCluster = data[clustering == clustering[i]]
        nj = len(clustering == clustering[i])
        mu_in = sum([distance(data[i],sameCluster[j]) for j in range(nj)])/(nj-1) #we add the distance to itself, since it's zero anyways.
        mu_out = np.min([sum([distance(data[i],data[clustering == c]) for]) for c in range(k)])
    # END CODE

    return silh

def distance(x,y):
    dist = np.linalg.norm(x-y)
    return dist