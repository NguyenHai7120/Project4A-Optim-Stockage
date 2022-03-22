import numpy as np

def channel_encoder(channel, max_channel, min_channel, start_by_min = True):
    H, W = channel.shape
    encoded_channel = np.zeros((H,W))
    is_min = start_by_min
    for h in range(H):
        for w in range(W):
            dis_to_min = channel[h,w] - min_channel[h,w]
            dis_to_max = max_channel[h,w] - channel[h,w]
            if is_min == True:
                encoded_channel[h,w] = dis_to_min
            if is_min == False:
                encoded_channel[h,w] = dis_to_max
            
            is_min = dis_to_min <= dis_to_max
            
    return encoded_channel

def channel_decoder(encoded_channel, max_channel, min_channel, start_by_min = True):
    H, W = encoded_channel.shape
    decoded_channel = np.zeros((H,W))
    is_min = start_by_min

    for h in range(H):
        for w in range(W):
            if is_min == True:
                decoded_channel[h,w] = min_channel[h,w] + encoded_channel[h,w]
            if is_min == False:
                # print(max_channel[h,w])
                # print(encoded_channel[h,w])
                decoded_channel[h,w] = max_channel[h,w] - encoded_channel[h,w]
                # print("Decoded:", decoded_channel[h,w])

            dis_to_min = decoded_channel[h,w] - min_channel[h,w]
            dis_to_max = max_channel[h,w] - decoded_channel[h,w]
            is_min = dis_to_min <= dis_to_max

    return decoded_channel

def MMD_Encoder(X, start_by_min = True):
    '''
    Function to encode the min-max differential extraction
    Input: X - set of originals images, of shape N x H x W x C 
           where N : nb of images, H : height, W : width and C : nb of channels (3)
    Output: Y - set of extraction datas (images) of shape N x H x W x C 
            max_im
            min_im
    Notes: As default, we start by subtracting the min image

    This is un vectorized implementation
    '''
    max_im = np.max(X, axis = 0)
    min_im = np.min(X, axis = 0)

    # Y = np.minimum(max_im - X, X - min_im)
    Y = np.zeros(X.shape)
    N, H, W, C = np.shape(X)

    for im in range(N):
        for c in range(C):
            Y[im,:,:,c] = channel_encoder(X[im,:,:,c], max_im[:,:,c], min_im[:,:,c], start_by_min= start_by_min)

    return max_im, min_im, Y


def MMD_Decoder(Y, max_im, min_im, start_by_min = True):
    '''
    Function to decode the min-max differential 
    Input: Y - set of images encoded of shape N x H x W x C 
                where N : nb of images, H : height, W : width and C : nb of channels (3)
           max_im : maximum image over pixel
           min_im : minimum image over pixel

    Output: X - set of original images of shape N x H x W x C 

    This is unvertorized version
    '''
    # X = (Y + min_im)*(2*Y < max_im - min_im) + (max_im - Y)*(2*Y >= max_im - min_im)
    X = np.zeros(Y.shape)
    N, H, W, C = np.shape(Y)

    for im in range(N):
        for c in range(C):
            X[im,:,:,c] = channel_decoder(Y[im,:,:,c], max_im[:,:,c], min_im[:,:,c], start_by_min= start_by_min)
    return X

    
    