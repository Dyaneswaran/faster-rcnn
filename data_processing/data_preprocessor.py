import numpy as np

def preprocess_image(config, img):
    '''
    Input - 

    config - Config file contains Preprocessing Parameters.
    img - Input Image (BGR format).

    Output - 

    img - Preprocessed Image.
    '''
    
    ### Mean Reduction 0-Mean ###

    (red_channel_mean, green_channel_mean, blue_channel_mean) = config.img_channel_mean
    
    img = img.astype(np.float32)
    img[:,:,0] -= blue_channel_mean
    img[:,:,1] -= green_channel_mean
    img[:,:,2] -= red_channel_mean

    ### Standard Scaling 1-Variance ###

    std_variance = config.img_scaling_factor
    img /= std_variance

    return img


