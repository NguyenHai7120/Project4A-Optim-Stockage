import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as skm

def image_encoder(I, Imin, Imax, m = 20, block_size = (2,2,1)):
    '''
    Image encoder of min max predictive method
    Input:  I - image of shape (H, W, 3)

    Return: encoded_image : numpy array of the same shape as channel
            level_image : numpy array of shape image.shape / block_size (by each dim)
    '''
    encoded_image = np.zeros(I.shape)
    level_image = m*(I - Imin)/(Imax - Imin)
    level_image = level_image.astype(int)
    level_image = skm.block_reduce(level_image, block_size = block_size, func = np.min)
    I_hat = Imin + np.floor(np.kron(level_image, np.ones(block_size))*(Imax - Imin)/m)
    encoded_image = I - I_hat

    return encoded_image, level_image


def image_decoder(I_encoded, level, Imin, Imax, m = 20, block_size = (2,2,1)):
    I_hat = Imin + np.floor(np.kron(level, np.ones(block_size))*(Imax - Imin)/m)
    return I_encoded + I_hat

def MMPredictive_Encoder(X, m = 20, block_size = (2,2,1)):
    '''
    Min Max Predictive Encoder
    Input: X - numpy array of shape (N, H, W, C)
       block_size - block size used in predictive level
       m - number of levels 
    Return: Y - numpy array of shape (N, H, W, C), difference between predictive images and original images
    L - numpy array of shape (N, H/block_size, W/block_size, C) containing the levels  
    Imin
    Imax
    '''
    N = X.shape[0]
    Y = []
    L = []
    Imin = np.min(X, axis=0)
    Imax = np.max(X, axis=0)

    for i in range(N):
        Yi, Li = image_encoder(X[i], Imin, Imax, m = m, block_size=block_size)
        Y.append(Yi)
        L.append(Li)

    return np.array(Y), np.array(L), Imin, Imax

def MMPredictive_Decoder(Y, L, Imin, Imax, m = 20, block_size = (2,2,1)):
    '''
    Min Max Predictive Encoder
    Input: Y - numpy array of shape (N, H, W, C), difference between predictive images and original images
           L - numpy array of shape (N, H/block_size, W/block_size, C) containing the levels
           block_size - block size used in predictive level
           m - number of levels  
    Return:
    X - numpy array of shape (N, H, W, C)
    '''
    N = Y.shape[0]
    X = []

    for i in range(N):
        Xi = image_decoder(Y[i], L[i], Imin, Imax, m = m, block_size=block_size)
        X.append(Xi)

    return np.array(X)