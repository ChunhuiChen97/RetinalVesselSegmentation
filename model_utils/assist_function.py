# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import h5py

def load_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        return f['image'][()]

def write_hdf5(arr, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('image', data = arr, dtype = arr.dtype)

#group a set of images row per columns
def group_imgs(data, per_row):
    assert data.shape[0] % per_row == 0
    assert (data.shape[3] == 1 or data.shape[3] == 3)
    all_strip = []
    for i in range(int(data.shape[0]/per_row)):
        strip = data[i*per_row]
        for j in range(i*per_row + 1, (i+1) * per_row):
            strip = np.concatenate((strip, data[j]), axis = 1)
        all_strip.append(strip)
    totimg = all_strip[0]
    for i in range(1, len(all_strip)):
        totimg = np.concatenate((totimg, all_strip[i]), axis = 0)
    return totimg

# visualize image(as opencv, not as matlotlib)
def visualize(data, filename):
    assert(len(data.shape) == 3) # [height, width, cahnnel]
    img = None
    if data.shape[2] == 1: # black-white style
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1: # the image is already 0-255
        img = data.astype(np.uint8)
    else:
        img = (data*255).astype(np.uint8)
    cv.imwrite(filename+'.png', img)
    return img

# prepare the mask for U-net
def masks_prepare(masks):
    assert(len(masks.shape) == 4) # 4D array
    assert(masks.shape[3] == 1) # one channel
    img_h = masks.shape[1]
    img_w = masks.shape[2]
    masks = np.reshape(masks, (masks.shape[0], img_h * img_w))
    new_masks = np.empty((masks.shape[0], img_h * img_w, 2))
    
    for i in range(masks.shape[0]):
        for j in range(img_h * img_w):
            if masks[i,j] == 0:
                new_masks[i,j,0] = 1
                new_masks[i,j,1] = 0
            else:
                new_masks[i,j,0] = 0
                new_masks[i,j,1] = 1
    return new_masks

# form identification(prediction) of pixels to image patches
def iden_to_imgs(y_pred, patch_height, patch_width, mode = 'original'):
    assert(len(y_pred.shape) == 4) # n_patches, patch_height * patch_width, 2 classes
    assert(y_pred.shape[3] == 1) # 2 classes, 0 / 1, one channel
    predicted_img_patches = np.empty((y_pred.shape)) # n_patches, patch_height * patch_width
    if mode == 'original':
        predicted_img_patches = y_pred
        
    elif mode == 'threshold':
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                for k in range(y_pred.shape[2]):
                    for l in range(y_pred.shape[3]):
                        if y_pred[i, j, k, l] >= 0.5:
                            predicted_img_patches[i, j, k, l] = 1
                        else:
                            predicted_img_patches[i, j, k ,l] = 0
    else:
        print('mode '+ mode + " can not be recognized, it should be 'threshold' or 'original'")
        exit()

    return predicted_img_patches