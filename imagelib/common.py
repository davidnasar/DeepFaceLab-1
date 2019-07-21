import numpy as np

def normalize_channels(img, target_channels):
    img_shape_len = len(img.shape)
    if img_shape_len == 2:
        h, w = img.shape
        c = 0
    elif img_shape_len == 3:
        h, w, c = img.shape
    else:
        raise ValueError("normalize: incorrect image dimensions.")
        
    if c == 0 and target_channels > 0:
        img = img[...,np.newaxis]        
    if c == 1 and target_channels > 1:
        img = np.repeat (img, target_channels, -1)   
    if c > target_channels:         
        img = img[...,0:target_channels]
        c = target_channels
        
    return img