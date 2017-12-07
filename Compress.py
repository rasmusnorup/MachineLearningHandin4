import numpy as np
import os
import scipy.misc
import matplotlib.pyplot as plt
import ExpMax
import Lloyd
from PIL import Image





def download_image(url):
    filename = url[url.rindex('/')+1:]
    try:
        with open(filename, 'rb') as fp:
            return scipy.misc.imread(fp) / 255
    except FileNotFoundError:
        import urllib.request
        with open(filename, 'w+b') as fp, urllib.request.urlopen(url) as r:
            fp.write(r.read())
            return scipy.misc.imread(fp) / 255
"""
img_facade = download_image('https://uploads.toptal.io/blog/image/443/toptal-blog-image-1407508081138.png')

fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.imshow(img_facade)
plt.show()

size = os.stat('toptal-blog-image-1407508081138.png').st_size

print("The image consumes a total of %i bytes. \n"%size)
print("You should compress your image as much as possible! ")
"""
def compress_Mix(im, k, T, name):
    height, width, depth = im.shape
    data = im.reshape((height * width, depth))
    clustering, centroids, score = Lloyd.lloyds_algorithm(data, k, 10)
    centroids, covs, probs_c, llh = ExpMax.em_algorithm(data, k, T, epsilon = 0.001, means=centroids)
    clustering = ExpMax.compute_em_cluster(centroids, covs, probs_c, data)

    # make each entry of data to the value of it's cluster
    data_compressed = data

    for i in range(k): data_compressed[clustering == i] = centroids[i]

    im_compressed = data_compressed.reshape((height, width, depth))

    # The following code should not be changed.
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(im_compressed)
    plt.savefig("compressed.jpg")
    #plt.show()

    original_size   = os.stat(name).st_size
    compressed_size = os.stat('compressed.jpg').st_size
    print("Original Size: \t\t", original_size)
    print("Compressed Size: \t", compressed_size)
    print("Compression Ratio: \t", round(original_size/compressed_size, 5))
    return original_size/compressed_size


def compress_kmeans(im, k, T, name):
    height, width, depth = im.shape
    data = im.reshape((height * width, depth))
    clustering, centroids, score = Lloyd.lloyds_algorithm(data, k, T)
    # make each entry of data to the value of it's cluster
    data_compressed = data

    for i in range(k): data_compressed[clustering == i] = centroids[i]

    im_compressed = data_compressed.reshape((height, width, depth))

    # The following code should not be changed.
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(im_compressed)
    plt.savefig("compressed.jpg")
    #plt.show()

    original_size   = os.stat(name).st_size
    compressed_size = os.stat('compressed.jpg').st_size
    print("Original Size: \t\t", original_size)
    print("Compressed Size: \t", compressed_size)
    print("Compression Ratio: \t", round(original_size/compressed_size, 5))
    return original_size/compressed_size

def compress_EM(im, k, T, name):
    height, width, depth = im.shape
    data = im.reshape((height * width, depth))

    centroids, covs, probs_c, llh = ExpMax.em_algorithm(data, k, T, epsilon = 0.001)
    clustering = ExpMax.compute_em_cluster(centroids, covs, probs_c, data)
    # make each entry of data to the value of it's cluster
    data_compressed = data

    for i in range(k): data_compressed[clustering == i] = centroids[i]
    data_compressed = data_compressed*255
    im_compressed = data_compressed.reshape((height, width, depth))

    # The following code should not be changed.
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(im_compressed)
    plt.savefig("compressed.jpg")
    #plt.show()

    original_size   = os.stat(name).st_size
    compressed_size = os.stat('compressed.jpg').st_size
    print("Original Size: \t\t", original_size)
    print("Compressed Size: \t", compressed_size)
    print("Compression Ratio: \t", round(original_size/compressed_size, 5))
    return original_size/compressed_size

def compress_facade(k=3, T=1, best = np.asarray([0,0])):
    #img_facade = download_image('https://www.sixt.ca/fileadmin/files/global/user_upload/pictures-city-page/nice-ville.jpg')
    img_facade = scipy.misc.imread('Hotels.jpg') / 255
    ratio_mix = compress_Mix(img_facade, k, T, 'Hotels.jpg')
    print("ratios")
    if ratio_mix > best[1]:
        best = [k,ratio_mix]
    return best
"""
h = np.asarray([0,0])
for i in range(3,11):
    h = compress_facade(i,30,h)
    print(h)
print("Finalle h")
print(h)
"""
compress_facade(3,100,[0,0])
os.rename('compressed.jpg', 'compressed_k3.jpg')
compress_facade(12,100,[0,0])
os.rename('compressed.jpg', 'compressed_k12.jpg')
