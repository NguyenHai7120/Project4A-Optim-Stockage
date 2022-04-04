import os
import glob
import cv2
import pathlib


def single_compression(im_set, path, method):
    '''
    PNG compression
    im_set: set of image of shape (N, H, W, C)
    method: string, indicates which method is used for set extraction
    '''

    N = im_set.shape[0]

    # origin_path = '/home/mhnguyen/4GMM 2/Projet Tutore/Project4A-Optim-Stockage/Saved_datas'
    # os.chdir(origin_path)

    # path = origin_path + '/' + method

    path = path + '/' + method
    
    if os.path.exists(path):
        # os.rmdir(path)
        files = glob.glob(path + '/*')
        for f in files:
            os.remove(f)
    else:
        os.mkdir(path)
    
    os.chdir(path)
    for i in range(N):
        cv2.imwrite(str(i) + ".png", im_set[i])