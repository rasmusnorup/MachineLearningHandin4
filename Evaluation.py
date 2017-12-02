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
    s = []
    for i in range(n):
        sameCluster = data[clustering == clustering[i]]
        nj = len(sameCluster)
        mu_in = np.sum([distance(data[i],sameCluster[j]) for j in range(nj)])/(nj-1) #we add the distance to itself, since it's zero anyways.

        result = []
        list = range(k)
        for c in range(k):
            if c != clustering[i]:
                sameCluster = data[clustering == c]
                nj = len(sameCluster)
                result.append(np.sum([distance(data[i],sameCluster[j]) for j in range(nj)])/nj)

        mu_out = np.min(result)
        si = (mu_out - mu_in)/np.max([mu_in,mu_out])
        s.append(si)
    silh = (1/n) * np.sum(s)
    # END CODE

    return silh

def distance(x,y):
    dist = np.linalg.norm(x-y)
    return dist