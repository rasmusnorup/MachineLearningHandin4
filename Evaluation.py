import sklearn.datasets
import numpy as np
import ExpMax as EM
import Lloyd as lloyd
X, y = sklearn.datasets.load_iris(True)
X = X[:,0:2] # reduce to 2d so you can plot if you want
#print(X.shape, y.shape)

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



def f1(predicted, labels):
    n, = predicted.shape
    assert labels.shape == (n,)
    r = np.max(predicted) + 1
    k = np.max(labels) + 1

    # Implement the F1 score here
    # YOUR CODE HERE
    contingency = np.zeros((r,k))
    for i in range(n):
        contingency[predicted[i]][labels[i]] += 1
    F_individual = []
    for i in range(r):
        ni = (predicted == i).sum()
        niji = np.max(contingency[i])
        mji = (labels == np.argmax(contingency[i])).sum()
    F_individual.append(2*niji/(ni+mji))

    F_overall = 1/r*sum(F_individual)
    # END CODE

    assert contingency.shape == (r, k)
    return F_individual, F_overall, contingency


def distance(x,y):
    dist = np.linalg.norm(x-y)
    return dist


for k in range(2, 10):
    total_em = 0
    total_l = 0
    for i in range(10):
        means, covs, probs_c, llh = EM.em_algorithm(X, k, 50)
        clustering = EM.compute_em_cluster(means, covs, probs_c, X)
        em_sc = silhouette(X, clustering)
        total_em += em_sc
    print("Silhouette coefficient for EM with k=" +str(k)+" : " + str(total_em/10))
    """
    for i in range(10):
        clustering, centroids, cost = lloyd.lloyds_algorithm(X, k, 50)
        lloyd_sc = silhouette(X, clustering)
        total_l += lloyd_sc
    print("Silhouette coefficient for Lloyd's with k=" + str(k) + " : " + str(total_l/10))
    """

"""
total = 0
for i in range(200):

    means, covs, probs_c, llh = EM.em_algorithm(X, 3, 50, epsilon=0.0001, means=None)
    clustering = EM.compute_em_cluster(means, covs, probs_c, X)
    #clustering, centroids, cost = lloyd.lloyds_algorithm(X, 3, 50)
    F_individual, F_overall, contingency = f1(clustering , y)
    print("F1 Score: " + str(F_overall))
    total += F_overall
print("F1 average over 50 runs: " + str(total/200))
"""