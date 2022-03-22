import numpy as np

def Delta_Encoder(X):
    '''
    Delta encoder
    '''
    Y = np.copy(X)
    Y[:,1:,:,:] = X[:,1:,:,:] - X[:,:-1,:,:]

    return Y 


def Delta_Decoder(Y):
    X = np.cumsum(Y, axis = 1)
    return X 