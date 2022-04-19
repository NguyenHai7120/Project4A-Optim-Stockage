import os
import cv2
import numpy as np
import glob
from min_max_predictive import MMPredictive_Encoder, MMPredictive_Decoder, image_decoder, image_encoder
from delta import Delta_Decoder, Delta_Encoder
import matplotlib.pyplot as plt
from min_max_diff import MMD_Encoder


#used for calculating size in general
def size_calc_mmd(X_test, X_encoded, Imin, Imax):

    original_size = np.zeros(X_test.shape[0])
    X_encoded_size = np.zeros(X_test.shape[0])
    Imin_size = 0
    Imax_size = 0

    #clear existing images
    files = glob.glob('./Images_test/original_image/*')
    for f in files:
        os.remove(f)
    files = glob.glob('./Images_test/X_encoded/*')
    for f in files:
        os.remove(f)
    files = glob.glob('./Images_test/I_min/*')
    for f in files:
        os.remove(f)
    files = glob.glob('./Images_test/I_max/*')
    for f in files:
        os.remove(f)

    for i in range(X_test.shape[0]):

        cv2.imwrite("./Images_test/original_image/img" + str(i) + ".png", X_test[i,:,:,:])
        cv2.imwrite("./Images_test/X_encoded/img" + str(i) + ".png", X_encoded[i,:,:,:])

    cv2.imwrite("./Images_test/I_min/img.png", Imin)
    cv2.imwrite("./Images_test/I_max/img.png", Imax)

    for i in range(X_test.shape[0]):
        X_encoded_size[i] = os.path.getsize("./Images_test/X_encoded/img" + str(i) + ".png")
        original_size[i] = os.path.getsize("./Images_test/original_image/img" + str(i) + ".png")
    Imin_size = os.path.getsize("./Images_test/I_min/img.png")
    Imax_size = os.path.getsize("./Images_test/I_max/img.png")

    sum = 0
    og_sum = np.sum(original_size)
    for i in range(X_test.shape[0]):
        sum = sum + X_encoded_size[i]
    sum = sum + Imin_size + Imax_size

    return sum, og_sum


#used for calculating size after clustering
def cluster_calc_mmd(n_clusters, clusters, X):

    list_sum = np.zeros(n_clusters)
    list_ogsum = np.zeros(n_clusters)

    for i in range(0, n_clusters): 
        cluster_elements = np.empty([32, 32, 3])
        cluster_elements = cluster_elements[...,np.newaxis]
        cluster_idx = np.where(clusters==i)
        for k in cluster_idx[0]:
            cluster_elements = np.concatenate((cluster_elements, X[k][...,np.newaxis]),axis=3)
        cluster_elements = np.transpose(cluster_elements, (3, 0, 1, 2))
        X_encoded, Imin, Imax = MMD_Encoder(cluster_elements)
        sum, ogsum = size_calc_mmd(cluster_elements, X_encoded, Imin, Imax)
        list_sum[i] = sum
        list_ogsum[i] = ogsum
    return list_sum, list_ogsum
    #result_plot(list_sum, list_ogsum)

def result_plot(sum, ogsum):
    plt.plot(sum, label = 'Storage cost of compressed cluster set #')
    plt.plot(ogsum, label = 'Storage cost of original cluster set #')
    plt.xlabel('#nr of cluster')
    plt.ylabel('Storage cost [byte]')
    plt.legend()
    plt.show()
