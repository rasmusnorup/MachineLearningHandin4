import tensorflow as tf
from sklearn.mixture import GaussianMixture as EM
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/")

X = mnist.train.images
y = mnist.train.labels

# One cluster for each digit
k = 10

# Run EM algorithm on 1000 images from the MNIST dataset.
expectation_maximization = EM(n_components=k, max_iter=10, init_params='kmeans', covariance_type='diag', verbose=1,
                              verbose_interval=1).fit(X)

means = expectation_maximization.means_
covs = expectation_maximization.covariances_

fig, ax = plt.subplots(1, k, figsize=(8, 1))

for i in range(k):
    ax[i].imshow(means[i].reshape(28, 28), cmap='gray')

plt.show()

from scipy.stats import multivariate_normal
import numpy as np


def sample(means, covs, num):
    mean = means[num]
    cov = covs[num]

    fig, ax = plt.subplots(1, 10, figsize=(8, 1))

    for i in range(10):
        img = multivariate_normal.rvs(mean=mean, cov=cov)  # draw random sample
        ax[i].imshow(img.reshape(28, 28), cmap='gray')  # draw the random sample
    plt.show()


sample(means, covs, 0)