import os
import cv2
import numpy as np
import glob

def size_calc(X_test, L_encoded, X_encoded, Imin, Imax):

    #os.path.getsize('level.png')
    #os.path.getsize('level_delta.png')
    non_delta_size = np.zeros(X_test.shape[0])
    delta_size = np.zeros(X_test.shape[0])
    original_size = np.zeros(X_test.shape[0])
    L_encoded_size = np.zeros(X_test.shape[0])
    X_encoded_size = np.zeros(X_test.shape[0])
    Imin_size = 0
    Imax_size = 0

    #clear existing images
    files = glob.glob('./Images_test/original_image/*')
    for f in files:
        os.remove(f)
    files = glob.glob('./Images_test/L_encoded/*')
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
        cv2.imwrite("./Images_test/L_encoded/img" + str(i) + ".png", L_encoded[i,:,:,:])
        cv2.imwrite("./Images_test/X_encoded/img" + str(i) + ".png", X_encoded[i,:,:,:])

    cv2.imwrite("./Images_test/I_min/img.png", Imin)
    cv2.imwrite("./Images_test/I_max/img.png", Imax)

    for i in range(X_test.shape[0]):
        #non_delta_size[i] = os.path.getsize("./Images_test/non-delta/level" + str(i) + ".png")
        #delta_size[i] = os.path.getsize("./Images_test/delta/level" + str(i) + ".png")
        L_encoded_size[i] = os.path.getsize("./Images_test/L_encoded/img" + str(i) + ".png")
        X_encoded_size[i] = os.path.getsize("./Images_test/X_encoded/img" + str(i) + ".png")
        original_size[i] = os.path.getsize("./Images_test/original_image/img" + str(i) + ".png")
    Imin_size = os.path.getsize("./Images_test/I_min/img.png")
    Imax_size = os.path.getsize("./Images_test/I_max/img.png")

    sum = 0
    og_sum = np.sum(original_size)
    for i in range(X_test.shape[0]):
        sum = sum + L_encoded_size[i] + X_encoded_size[i]
    sum = sum + Imin_size + Imax_size

    return sum, og_sum