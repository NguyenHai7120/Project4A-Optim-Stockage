### Utilities for image compression adapted to MNIST dataset 

import numpy as np

##############################################################################
# In this module, we will define the functions that encode and decode arrays
# the Min Max Differential method applying to flattended arrays
##############################################################################

def MMD_Encoder_Flatten(X):
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

    Y = np.reshape(Y, (N, 28, 28))
    min_im = np.reshape(min_im, (28,28))
    max_im = np.reshape(max_im, (28,28))

    return Y, min_im, max_im         

#############################################

def MMD_Decoder_Flatten(Y, min_im, max_im):
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

    X = np.reshape(X, (N, 28, 28))
    return X

############# Median Encoding ############################
def Median_Encoder(X):
    median = np.median(X, axis = 0, keepdims = True)
    median = median.astype('uint8')
    Y = X - median

    return Y, median

def Median_Decoder(Y, median):
    X = Y + median
    return X
###########################################################



def Delta_Encoder(X):
    '''
    Delta encoder
    '''
    Y = np.copy(X)
    Y[:,1:,:] = X[:,1:,:] - X[:,:-1,:]

    return Y 


def Delta_Decoder(Y):
    X = np.cumsum(Y, axis = 1)
    return X 

