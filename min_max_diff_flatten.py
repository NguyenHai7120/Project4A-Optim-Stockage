import numpy as np 

##############################################################################
# In this module, we will define the functions that encode and decode arrays
# the Min Max Differential method applying to flattended arrays
##############################################################################

def Encoder(X, start_by_min = True):
    '''
    Min Max Differential
    X : input images, of shape (N, ), if not it will be reshaped

    return: Y - encoded images of the same shape as X
            min_im, max_im as arrays
    '''
    N = X.shape[0]
    X = X.reshape((N, -1))
    D = X.shape[1]
    min_im = np.min(X, axis=0)
    max_im = np.max(X, axis=0)

    Y = np.zeros(X.shape)
    is_min = start_by_min
    Y[:, 0] = X[:,0] - min_im[0]
    dis = (X[:,1:] - min_im[1:]) > (X[:,1:] - max_im[1:])
    for d in range(1,D):
        dis = (X[:,d-1] - min_im[d-1]) > (max_im[d] - X[:,d])
        Y[:,d] = (X[:,d] - min_im[d])*dis + (max_im[d] - X[:,d])*

