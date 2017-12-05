import numpy as np

import matplotlib.pyplot as plt
#import Evaluation as eval

# Load the Iris data set
import sklearn.datasets
iris = sklearn.datasets.load_iris()
X = iris['data'][:,0:2] # reduce to 2d so you can plot if you want

def lloyds_algorithm(X, k, T):
    """ Clusters the data of X into k clusters using T iterations of Lloyd's algorithm.

        Parameters
        ----------
        X : Data matrix of shape (n, d)
        k : Number of clusters.
        T : Maximum number of iterations to run Lloyd's algorithm.

        Returns
        -------
        clustering: A vector of shape (n, ) where the i'th entry holds the cluster of X[i].
        centroids:  The centroids/average points of each cluster.
        cost:       The cost of the clustering
    """
    n, d = X.shape

    # Initialize clusters random.
    clustering = np.random.randint(0, k, (n, ))
    #print(clustering)
    centroids  = np.zeros((k, d))

    # Used to stop if cost isn't improving (decreasing)
    cost = 0
    oldcost = 0

    # Column names
    #print("Iterations\tCost")

    for i in range(T):
        # Update centroid
        # YOUR CODE HERE
        for j in range(k):
            clusterSum = (clustering == j).sum()
            if clusterSum != 0:
                centroids[j] = 1/(clusterSum) *np.sum( X[clustering == j], 0)
            else: print("Devide by zero prevented. You can thank me later!")
        # END CODE

        # Update clustering
        # YOUR CODE HERE
        for j in range(n):
            temp = []
            for q in range(k):
                temp.append(np.square(np.linalg.norm(X[j]-centroids[q])))
            clustering[j] = np.argmin(temp)
        # END CODEp

        # Compute and print cost
        cost = 0
        for j in range(n):
            cost += np.linalg.norm(X[j] - centroids[clustering[j]])**2
        #print(i+1, "\t\t", cost)

        # Stop if cost didn't improve more than epislon (decrease)
        if np.isclose(cost, oldcost): break #TODO
        oldcost = cost

    return clustering, centroids, cost

#print(X.shape)
#clustering, centroids, cost = lloyds_algorithm(X, 3, 100)
#print(clustering)
#print(centroids)
#print(cost)
#print("Silhouette: " + str(eval.silhouette(X,clustering)))