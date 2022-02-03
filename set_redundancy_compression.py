import numpy as np

def MMD_Encoder(X):
    '''
    Function to encode the min-max differential extraction
    Input: X - set of originals images, of shape N x H x W x C 
    Output: Y - set of extraction datas (images) of shape N x H x W x C 
            max_im
            min_im
    '''
    max_im = np.max(X, axis = 0)
    min_im = np.min(X, axis = 0)
    Y = np.minimum(max_im - X, X - min_im)

    return max_im, min_im, Y


def MMD_Decoder(Y, max_im, min_im):
    '''
    Function to decode the min-max differential 
    Input: Y - set of images encoded of shape N x H x W x C 
    Output: X - set of original images of shape N x H x W x C 
    '''
    X = (Y + min_im)*(2*Y < max_im - min_im) + (max_im - Y)*(2*Y >= max_im - min_im)
    return X

    
    