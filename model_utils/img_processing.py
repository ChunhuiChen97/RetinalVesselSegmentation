# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np

def image_preprocessing(data):
    assert(len(data.shape) == 4)
    assert(data.shape[3] == 3 or data.shape[3] == 1)
    # if the input is colorful image, we then convert it into grayscale.
    if data.shape[3] == 3:
      imgs = rgb2gray(data)
    else:
      imgs = data
    imgs = dataset_normalize(imgs)
    imgs = CLAHE_equalize(imgs)
    imgs = gamma_adjust(imgs, 1.4)
    # imgs = 255 - truncated_linear_stretch(imgs)
    imgs = imgs/255.0
    return imgs

def rgb2gray(rgb):
    assert(len(rgb.shape) == 4)
    assert(rgb.shape[3] == 3) # channels == 3
    gray_imgs = rgb[:,:,:, 1] # extract green channel. it is b-g-r sequence
    gray_imgs = np.reshape(gray_imgs, (rgb.shape[0], rgb.shape[1], rgb.shape[2], 1))
    return gray_imgs

def dataset_normalize(imgs):
    assert(len(imgs.shape) == 4)
    assert(imgs.shape[3] == 1)
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean)/imgs_std # global normalization
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = (((imgs_normalized[i] - np.min(imgs_normalized[i]))
                            /(np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255)
    return imgs_normalized

def hist_equalize(imgs):
    assert(len(imgs.shape) == 4)
    assert(imgs.shape[3] == 1)
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,:, :, 0] = cv.equalizeHist(np.array(imgs[i, :, :, 0], dtype = np.uint8))
    return imgs_equalized

def CLAHE_equalize(imgs):
    assert(len(imgs.shape) == 4)
    assert(imgs.shape[3] == 1)
    clahe = cv.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, :, :, 0] = clahe.apply(np.array(imgs[i, :, :, 0], dtype = np.uint8))
    return imgs_equalized

def gamma_adjust(imgs, gamma = 1.4):
    assert(len(imgs.shape) == 4)
    assert(imgs.shape[3] == 1)
    inv_gamma = 1/gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    imgs_adjusted = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_adjusted[i,:, :, 0] = cv.LUT(np.array(imgs[i, :, :, 0], dtype = np.uint8), table)
    return imgs_adjusted

def truncated_linear_stretch(imgs, truncated_value = 2, min_out = 0, max_out = 255):
    assert(len(imgs.shape) == 4)
    assert(imgs.shape[3] == 1)
    for i in range(imgs.shape[0]):
      truncated_down = np.percentile(imgs[i,...], truncated_value)
      truncated_up = np.percentile(imgs[i,...], 100 - truncated_value)
      imgs[i,...] = ((max_out - min_out)/(truncated_up - truncated_down)) * (imgs[i,...] - truncated_down)
      imgs[i,...][imgs[i,...] < min_out] = min_out
      imgs[i,...][imgs[i,...] > max_out] = max_out
    return imgs