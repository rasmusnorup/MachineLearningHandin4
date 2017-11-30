import matplotlib.pyplot as plt
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


def compute_probs_cx(points, means, covs, probs_c):
    '''
    Input
      - points: (n times d) array containing the dataset
      - means:  (k times d) array containing the k means
      - covs:   (k times d times d) array such that cov[j,:,:] is the covariance matrix of the j-th Gaussian.
      - priors: (k) array containing priors
    Output
      - probs:  (k times n) array such that the entry (i,j) represents Pr(C_i|x_j)
    '''
    # Convert to numpy arrays.
    points, means, covs, probs_c = np.asarray(points), np.asarray(means), np.asarray(covs), np.asarray(probs_c)

    # Get sizes
    n, d = points.shape
    k = means.shape[0]

    # Compute probabilities
    # This will be a (k, n) matrix where the (i,j)'th entry is Pr(C_i)*Pr(x_j|C_i).
    probs_cx = np.zeros((k, n))
    for i in range(k):
        try:
            probs_cx[i] = probs_c[i] * multivariate_normal.pdf(mean=means[i], cov=covs[i], x=points)
        except Exception as e:
            print("Cov matrix got singular: ", e)

    # The sum of the j'th column of this matrix is P(x_j); why?
    probs_x = np.sum(probs_cx, axis=0, keepdims=True)
    assert probs_x.shape == (1, n)

    # Divide the j'th column by P(x_j). The the (i,j)'th then
    # becomes Pr(C_i)*Pr(x_j)|C_i)/Pr(x_j) = Pr(C_i|x_j)
    probs_cx = probs_cx / probs_x

    return probs_cx, probs_x

def em_algorithm_visualize(X, k, T, epsilon=0.001, means=None, detail=20):
    """ Clusters the data X into k clusters using the Expectation Maximization algorithm.

        Parameters
        ----------
        X : Data matrix of shape (n, d)
        k : Number of clusters.
        T : Maximum number of iterations
        epsilon :  Stopping criteria for the EM algorithm. Stops if the means of
                   two consequtive iterations are less than epsilon.
        means : (k times d) array containing the k initial means (optional)

        Returns
        -------
        means:     (k, d) array containing the k means
        covs:      (k, d, d) array such that cov[j,:,:] is the covariance matrix of
                   the Gaussian of the j-th cluster
        probs_c:   (k, ) containing the probability Pr[C_i] for i=0,...,k.
        llh:       The log-likelihood of the clustering (this is the objective we want to maximize)
    """
    n, d = X.shape

    # Visualization stuff.
    fig, (ax, _, _, _, _) = plt.subplots(5, 1, figsize=(10, 16))
    ax.axis('off')
    colors = ["r", "g", "b"]
    ax3d = fig.add_subplot(2, 1, 2, projection='3d')
    ax3d1 = fig.add_subplot(3, 1, 2, projection='3d')
    ax3d2 = fig.add_subplot(4, 1, 2, projection='3d')
    ax3d3 = fig.add_subplot(5, 1, 2, projection='3d')
    ax3d.set_axis_off()
    ax3d1.set_axis_off()
    ax3d2.set_axis_off()
    ax3d3.set_axis_off()

    # Initialize and validate mean
    if means is None: means = np.random.rand(k, d)

    # Initialize
    probs_x = np.zeros(n)
    probs_cx = np.zeros((k, n))
    probs_c = np.zeros(k)
    covs = np.zeros((k, d, d))
    for i in range(k): covs[i] = np.identity(d)
    probs_c = np.ones(k) / k

    # END CODE

    # Column names
    print("Iterations\tLLH")

    close = False
    old_means = np.zeros_like(means)
    iterations = 0
    while not (close) and iterations < T:
        old_means[:] = means  # This allows us to actually copy the array mean

        # Expectation step
        probs_cx, probs_x = compute_probs_cx(X, means, covs, probs_c)
        if probs_cx is None: return em_algorithm(X, k, T, epsilon=epsilon)
        assert probs_cx.shape == (k, n)

        # Maximization step
        # YOUR CODE HERE
        newMeans = means
        newCovs = covs
        for i in range(k):
            meanAbove = [0]*d
            meanBelow = 0
            covsAbove = np.zeros((d,d))
            for j in range(n):
                meanAbove = meanAbove + X[j]*probs_cx[i][j]
                meanBelow = meanBelow + probs_cx[i][j]
            newMeans[i] = np.divide(meanAbove,meanBelow)
            for j in range(n):
                covsAbove = covsAbove + probs_cx[i][j] * np.outer(X[j] - newMeans[i],X[j] - newMeans[i])
            newCovs[i] = np.divide(covsAbove,meanBelow)
            probs_c[i] = 1/n*np.sum(probs_cx[i])
        means = newMeans
        covs = newCovs
        # END CODE

        # Compute per-sample average log likelihood (llh) of this iteration
        llh = 1 / n * np.sum(np.log(probs_x))
        print(iterations + 1, "\t\t", llh)

        # Finish condition
        dist = np.sqrt(((means - old_means) ** 2).sum(axis=1))
        close = np.all(dist < epsilon)
        iterations += 1

        # !----------- VISUALIZATION CODE -----------!
        centroids = means
        # probs_cx's (i,j) is Pr[C_i, x_j]
        # assign each x_i to the cluster C_i that maximizes P(C_i | x_j)
        clustering = np.argmax(probs_cx, axis=0)
        assert clustering.shape == (n,), clustering.shape

        # Draw clusters
        ax.cla()
        for j in range(k):
            centroid = centroids[j]
            c = colors[j]
            ax.scatter(centroid[0], centroid[1], s=123, c=c, marker='^')
            data = X[clustering == j]
            x = data[:, 0]
            y = data[:, 1]
            ax.scatter(x, y, s=3, c=c)

        # draw 3d gaussians.
        # Create grid and multivariate normal
        xs = np.linspace(4, 7, 50)
        ys = np.linspace(2, 4.5, 50)
        Xs, Ys = np.meshgrid(xs, ys)
        pos = np.empty(Xs.shape + (2,))
        pos[:, :, 0] = Xs;
        pos[:, :, 1] = Ys
        prob_space = sum([multivariate_normal(means[j], covs[j]).pdf(pos) for j in range(k)])

        # Make a 3D plot
        ax3d.cla()
        ax3d1.cla()
        ax3d2.cla()
        ax3d3.cla()
        ax3d.plot_surface(Xs, Ys, prob_space, cmap='viridis', linewidth=0)
        ax3d1.plot_surface(Xs, Ys, multivariate_normal(means[0], covs[0]).pdf(pos), cmap='viridis', linewidth=0)
        ax3d2.plot_surface(Xs, Ys, multivariate_normal(means[1], covs[1]).pdf(pos), cmap='viridis', linewidth=0)
        ax3d3.plot_surface(Xs, Ys, multivariate_normal(means[2], covs[2]).pdf(pos), cmap='viridis', linewidth=0)

        fig.canvas.draw()

    # Validate output
    assert means.shape == (k, d)
    assert covs.shape == (k, d, d)
    assert probs_c.shape == (k,)
    return means, covs, probs_c


# Load the Iris data set
import sklearn.datasets

iris = sklearn.datasets.load_iris()
X = iris['data'][:, 0:2]  # reduce dimensions so we can plot what happens.
k = 3

# the higher the detail the slower plotting
detail = 50  # 50 looks very nice but your computer might not be able to handle it.
means, covs, priors = em_algorithm_visualize(X, 3, 40, 0.001, detail=detail)
plt.show()