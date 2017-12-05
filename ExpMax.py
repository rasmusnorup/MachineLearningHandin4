import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import Lloyd as lloyd

def getBestLloyd(X, k , T, times):
    bestCost = np.inf
    for i in range(times):
        clustering, centroids, cost = lloyd.lloyds_algorithm(X, k, T)
        if cost < bestCost:
            bestCentroids = centroids
            bestCost = cost
    return cost, bestCentroids

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


def em_algorithm(X, k, T, epsilon=0.001, means=None):
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

    #Use Lloyd to find good starting centroids:
    #cost, means = getBestLloyd(X,k,30,30)

    # Initialize and validate mean
    if means is None:
        means = np.random.rand(k, d)

    # Initialize cov, prior
    probs_x = np.zeros(n)
    probs_cx = np.zeros((k, n))
    probs_c = np.zeros(k) + np.random.rand(k)

    covs = np.zeros((k, d, d))
    for i in range(k): covs[i] = np.identity(d)
    probs_c = np.ones(k) / k

    # Column names
    #print("Iterations\tLLH")

    close = False
    old_means = np.zeros_like(means)
    iterations = 0
    while not (close) and iterations < T:
        old_means[:] = means

        # Expectation step
        probs_cx, probs_x = compute_probs_cx(X, means, covs, probs_c)
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

        """
        covs = np.zeros((k, d, d))
        meanAbove = np.matmul(probs_cx, X)
        meanBelow = np.sum(probs_cx,1)
        means = np.transpose(np.divide(np.transpose(meanAbove),meanBelow))
        probs_c = 1/n * meanBelow
        for i in range(k):
            Xminus = X-means[i]
            for j in range(n):
                covs[i] = covs[i] + probs_cx[i][j] * np.outer(Xminus[j],Xminus[j])
            covs[i] = np.divide(covs[i],meanBelow[i])
        """
        # END CODE

        # Compute per-sample average log likelihood (llh) of this iteration
        llh = 1 / n * np.sum(np.log(probs_x))
        #print(iterations + 1, "\t\t", llh)

        # Stop condition
        dist = np.sqrt(((means - old_means) ** 2).sum(axis=1))
        close = np.all(dist < epsilon)
        iterations += 1

    # Validate output
    assert means.shape == (k, d)
    assert covs.shape == (k, d, d)
    assert probs_c.shape == (k,)

    return means, covs, probs_c, llh

# Load the Iris data set
import sklearn.datasets
iris = sklearn.datasets.load_iris()
X = iris['data'][:,0:2] # reduce dimensions so we can plot what happens.
k = 3

means, covs, priors, llh = em_algorithm(X, 3, 100, 0.001)

fig, ax = plt.subplots(1, 1, figsize=(8,3))
llhs = []

for i in range(1):
    _, _, _, llh = em_algorithm(X, 3, 100)
    llhs.append(llh)
    ax.plot(llhs, 'bx')
    fig.canvas.draw()
#plt.show()


from sklearn.mixture import GaussianMixture as EM
expectation_maximization = EM(n_components=3, init_params='random', covariance_type='diag', verbose=2, verbose_interval =1).fit(X)

#print(expectation_maximization.score(X))

def compute_em_cluster(means, covs, probs_c, data):
    probs_x, probs_cx = compute_probs_cx(data, means, covs, probs_c)
    cluster = np.zeros(len(data)).astype(int)
    for i in range(len(data)):
        cluster[i] = np.argmax([probs_x[j][i] for j in range(means.shape[0])])
    return cluster

