from tkinter import YView
import numpy as np 

##############################################################################
# In this module, we will define the functions that encode and decode arrays
# the Min Max Differential method applying to flattended arrays
##############################################################################

def Encoder(X):
    '''
    Min Max Differential
    X : input images, of shape (N, ), if not it will be reshaped

    return: Y - encoded images of the same shape as X
            min_im, max_im as arrays
    '''
    N = X.shape[0]
    X = X.reshape((N, -1))

    min_im = np.min(X, axis=0)
    max_im = np.max(X, axis=0)

    Y = np.zeros(X.shape)
    Y[:, 0] = X[:,0] - min_im[0]
    dis = (X[:,:-1] - min_im[:-1]) < (X[:,:-1] - max_im[:-1])
    Y[:,1:] = (X[:,1:] - min_im[1:])*dis + (max_im[1:] - X[:,1:])*(~dis)

    Y = Y.reshape((N, 32, 32, -1))
    min_im = min_im.reshape((32,32,-1))
    max_im = max_im.reshape((32,32,-1))

    return Y, min_im, max_im         

#############################################

def Decoder(Y, min_im, max_im):
    '''
    Min Max Differential Decoder
    '''
    N = Y.shape[0]
    Y = Y.reshape((N, -1))
    min_im = min_im.ravel()
    max_im = max_im.ravel()
    D = Y.shape[1]
    
    X = np.zeros(Y.shape)
    X[:,0] = Y[:,0] + min_im[0]
    
    for d in range(1, D):
        dis = (X[:,d-1] - min_im[d-1]) < (X[:,d-1] - max_im[d-1])
        X[:,d] = (Y[:,d] + min_im[d])*dis + (max_im[d] - Y[:,d])*(~dis)

    X = X.reshape((N, 32, 32, -1))
    return X
