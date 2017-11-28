
import numpy as np

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
    centroids  = np.zeros((k, d))

    # Used to stop if cost isn't improving (decreasing)
    cost = 0
    oldcost = 0

    # Column names
    print("Iterations\tCost")

    for i in range(T):

        # Update centroid

        # YOUR CODE HERE
        # END CODE


        # Update clustering

        # YOUR CODE HERE
        # END CODE


        # Compute and print cost
        cost = 0
        for j in range(n):
            cost += np.linalg.norm(X[j] - centroids[clustering[j]] ) **2
        print( i +1, "\t\t", cost)


        # Stop if cost didn't improve more than epislon (decrease)
        if np.isclose(cost, oldcost): break  # TODO
        oldcost = cost

    return clustering, centroids, cost

clustering, centroids, cost = lloyds_algorithm(X, 3, 100)

