# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import cv2 as cv
import random
import h5py
from .assist_function import load_hdf5
from .img_processing import image_preprocessing

# to select the same images
random.seed(10)

# load the original data and return image patches for training/testing
def get_data_training(Train_imgs_original,
                     Train_GT,
                     Train_masks,
                     patch_height,
                     patch_width,
                     n_subimgs,
                     inside_FOV):
    train_imgs_original = load_hdf5(Train_imgs_original)
    train_GT = load_hdf5(Train_GT)
    train_masks = load_hdf5(Train_masks)
#     visualize(group_imgs(train_imgs_original[0:20,:,:,:],5), 'imgs_train')

    train_imgs = image_preprocessing(train_imgs_original)
    train_GT = train_GT / 255.0
    # train_imgs = train_imgs[:, 9:574, :, :] # crop images into 565*565
    # train_GT = train_GT[:, 9:574, :, :]
    data_consistency_check(train_imgs, train_GT)
    
    assert(np.max(train_GT) == 1 and np.min(train_GT) == 0.0)
    print('\ntrain images/GT shape: ')
    print(train_imgs.shape, train_GT.shape)
    print('train imgs range(min-max): ' + str(np.min(train_imgs)) + '-' + str(np.max(train_imgs)))
    print('train GT are within 0-1\n')
    
    # extract training patches from the whole images
    train_imgs_patches, train_GT_patches = extract_random(train_imgs, train_GT, train_masks,
                                                          patch_height, patch_width, n_subimgs, inside_FOV)
    data_consistency_check(train_imgs_patches, train_GT_patches)
    print('\ntrain images/GT shape: ')
    print(train_imgs_patches.shape, train_GT_patches.shape)
    print('train imgs range(min-max): ' + str(np.min(train_imgs_patches)) + '-' + str(np.max(train_imgs_patches)))

    return train_imgs_patches.astype(np.float32), train_GT_patches.astype(np.float32)

def get_data_testing(Test_imgs_original,
                    Test_GT,
                    n_imgs_to_test,
                    patch_height,
                    patch_width):
    ## test
    test_imgs_original = load_hdf5(Test_imgs_original)
    test_GT = load_hdf5(Test_GT)
    test_imgs = image_preprocessing(test_imgs_original)
    test_GT = test_GT / 255.0
    test_imgs = test_imgs[0:n_imgs_to_test, :, :, :]
    test_GT = test_GT[0:n_imgs_to_test, :, :, :]
    test_imgs = paint_boarder(test_imgs, patch_height, patch_width)
    test_GT = paint_boarder(test_GT, patch_height, patch_width)
    data_consistency_check(test_imgs, test_GT)
    
    assert(np.min(test_GT) == 0 and np.max(test_GT) == 1)
    print('\ntest images/GT shape: ')
    print(test_imgs.shape, test_GT.shape)
    print('test images range(min-min): ' + str(np.min(test_imgs)) + '-' + str(np.max(test_imgs)))
    print('test GT are with 0-1\n')
    
    ## extract the test patch from the full images
    test_imgs_patches, n_patches_h, n_patches_w = extract_ordered(test_imgs, patch_height, patch_width)
    test_GT_patches, n_patches_h, n_patches_w = extract_ordered(test_GT, patch_height, patch_width)
    data_consistency_check(test_imgs_patches, test_GT_patches)
    
    print('\ntest patches images/GT shape: ')
    print(test_imgs_patches.shape, test_GT_patches.shape)
    print('test image patches range(min-max): ' + str(np.min(test_imgs_patches)) + '-' + str(np.max(test_imgs_patches)))
    print(str(n_patches_h) + ' patches in H dimension, ' + str(n_patches_w) + ' in W dimension.' )
    
    return test_imgs_patches.astype(np.float32), test_GT_patches.astype(np.float32), n_patches_h, n_patches_w

# load the original data and return the extracted pathes for testing
# return the ground truth in its original shape

def get_data_testing_overlap(Test_imgs_original, Test_GT, n_imgs_to_test,
                             patch_height, patch_width, stride_height, stride_width):
    test_imgs_original = load_hdf5(Test_imgs_original)
    test_GT = load_hdf5(Test_GT)
    
    test_imgs = image_preprocessing(test_imgs_original)
    test_GT = test_GT / 255.0
    # extend both imgs and GT so they can be divided by the patches dimension
    test_imgs = test_imgs[0:n_imgs_to_test, :, :, :].astype(np.float32)
    test_GT = test_GT[0:n_imgs_to_test, :, :, :].astype(np.float32)
    test_imgs = paint_boarder_overlap(test_imgs, patch_height, patch_width,
                                      stride_height, stride_width)
    test_GT = paint_boarder_overlap(test_GT, patch_height, patch_width,
                                      stride_height, stride_width)
    # check test_GT are with 0-1
    assert(np.max(test_GT) == 1 and np.min(test_GT) == 0)
    
    print('\ntest_imgs shape: ')
    print(test_imgs.shape)
    print('\ntest ground truth shape: ')
    print(test_GT.shape)
    print('test images range(min-max): ' + str(np.min(test_imgs)) + '-' + str(np.max(test_imgs)))
    print('test GT are within 0-1 \n')
    
    # extract the test patches from full images
    test_img_patches = extract_ordered_overlap(test_imgs, patch_height,
                                            patch_width, stride_height, stride_width)
    test_GT_patches = extract_ordered_overlap(test_GT, patch_height,
                                            patch_width, stride_height, stride_width)
    print("\ntest PATCHES images shape: ")
    print(test_img_patches.shape)
    print('test image patches range(min-max): ' + str(np.min(test_img_patches)) + '-' + str(np.max(test_img_patches)))
    return test_img_patches.astype(np.float32), test_imgs.shape[1], test_imgs.shape[2], test_GT_patches.astype(np.float32)

# data consistency check
def data_consistency_check(imgs, GT):
    assert(len(imgs.shape) == len(GT.shape))
    assert(imgs.shape[0] == GT.shape[0])
    assert(imgs.shape[1] == GT.shape[1])
    assert(imgs.shape[2] == GT.shape[2])
    assert(imgs.shape[3] == GT.shape[3])
    assert(GT.shape[3] == 1)
    assert(imgs.shape[3] == 1 or imgs.shape[3] == 3)

# extract patches randomly in the full training imgs

def extract_random(full_imgs, full_GT, full_masks, patch_height, patch_width, n_patches, inside = True):
    if (n_patches % full_imgs.shape[0] != 0):
        print(f'n_patches: please enter a multiple integer of {full_imgs.shape[0]} !')
        exit() 
    assert(len(full_imgs.shape) == 4 and len(full_GT.shape) == 4) # 4D array
    assert(full_imgs.shape[3] == 1 or full_imgs.shape[3] == 3) # check the channel
    assert(full_GT.shape[3] == 1)
    assert(full_imgs.shape[1] == full_GT.shape[1])
    assert(full_imgs.shape[2] == full_GT.shape[2])
    patches_imgs = np.empty((n_patches, patch_height, patch_width, full_imgs.shape[3])) # slice, h, w, channel
    patches_GT = np.empty((n_patches, patch_height, patch_width, full_GT.shape[3]))
    img_h = full_imgs.shape[1]
    img_w = full_imgs.shape[2]
    # (0,0) is the center of the image
    n_patches_per_img = int(n_patches / full_imgs.shape[0])
    print('patches per full image:' + str(n_patches_per_img))
    iter_total = 0 # iter over the total number of patches
    for i in range(full_imgs.shape[0]):
        k = 0
        while k < n_patches_per_img:
            x_center = random.randint(0 + int(patch_height / 2), img_h - int(patch_height / 2))
            # print('x_center': + str(x_center))
            y_center = random.randint(0 + int(patch_width / 2), img_w - int(patch_width / 2))
            # print('y_center:' + str(y_center))
            # check if the patch is fully contained in the FOV
            if inside == True:
                if center_inside_FOV(i, x_center, y_center, full_masks) == False:
                    continue
            patch = full_imgs[i, x_center - int(patch_height / 2) : x_center + int(patch_height / 2), y_center - int(patch_width / 2) : y_center + int(patch_width / 2), :]
            patch_GT = full_GT[i, x_center - int(patch_height / 2) : x_center + int(patch_height / 2), y_center - int(patch_width / 2) : y_center + int(patch_width / 2), :]
            patches_imgs[iter_total] = patch
            patches_GT[iter_total] = patch_GT
            iter_total += 1 # full imgs
            k += 1 # per img
    return patches_imgs, patches_GT

# check if the patch is fully contained in the FOV
def center_inside_FOV(i, x, y, full_masks):
    assert(len(full_masks.shape) == 4)
    assert(full_masks.shape[3] == 1)
    
    if full_masks[i, x, y, 0] > 0:
        return True
    else:
        return False

#split all the full_imgs into pacthes without overlap
def extract_ordered(full_imgs, patch_h, patch_w):
    assert(len(full_imgs.shape) == 4) # it should be a 4D array
    assert(full_imgs.shape[3] ==1 or full_imgs.shape[3] == 3) # check the channel
    img_h = full_imgs.shape[1]
    img_w = full_imgs.shape[2]
    n_patches_h = int(img_h / patch_h) # round to the lowest int
    n_patches_w = int(img_w / patch_w)
    if (img_h % patch_h != 0):
        print('warning:' + str(n_patches_h)) + 'patches in height, with about' + str(img_h % patch_h) + 'pixels left out'
    if (img_w % patch_w != 0):
        print('warning:' + str(n_patches_w)) + 'patches in width, with about' + str(img_w % patch_w) + 'pixels left out'
    print('number of patches per image:' + str(n_patches_h * n_patches_w))
    n_patches = (n_patches_h * n_patches_w) * full_imgs.shape[0]
    patches = np.empty((n_patches, patch_h, patch_w,  full_imgs.shape[3]))
    
    iter_total = 0
    for i in range(full_imgs.shape[0]):
        for j in range(n_patches_h):
            for k in range(n_patches_w):
                patch = full_imgs[i, j * patch_h: (j + 1) * patch_h, k * patch_w : (k + 1) * patch_w, :]
                patches[iter_total] = patch
                iter_total += 1
    assert( iter_total == n_patches)
    return patches, n_patches_h, n_patches_w # array with all the full_imgs spiltted into patches

# padding pixels to full images so that they can be splitted
def paint_boarder_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert(len(full_imgs.shape) == 4) # it should be a 4D array
    assert(full_imgs.shape[3] ==1 or full_imgs.shape[3] == 3) # check the channel
    img_h = full_imgs.shape[1]
    img_w = full_imgs.shape[2]
    leftover_h = (img_h - patch_h) % stride_h 
    leftover_w = (img_w - patch_w) % stride_w
    if leftover_h != 0:
        print('\nthe side H is not compatible with the selected stride of ' + str(stride_h))
        print('img_h ' + str(img_h) + ', patch_h ' + str(patch_h) + ', stride_h ' + str(stride_h))
        print('(img_h - patch_h) % stride_h: ' + str(leftover_h))
        print('so the H dimension will be padded with additional ' + str(stride_h - leftover_h) + ' pixels')
        temp_full_imgs = np.zeros((full_imgs.shape[0], img_h + stride_h - leftover_h, img_w,  full_imgs.shape[3]))
        temp_full_imgs[0:full_imgs.shape[0], 0:img_h, 0:img_w, 0:full_imgs.shape[3]] = full_imgs
        full_imgs = temp_full_imgs
    if leftover_w != 0:
        print('\nthe side W is not compatible with the selected stride of ' + str(stride_w))
        print('img_w ' + str(img_w) + ', patch_w ' + str(patch_w) + ', stride_w ' + str(stride_w))
        print('(img_w - patch_w) % stride_w: ' + str(leftover_w))
        print('so the W dimension will be padded with additional ' + str(stride_w - leftover_w) + ' pixels')
        temp_full_imgs = np.zeros((full_imgs.shape[0], full_imgs.shape[1], img_w + stride_w - leftover_w, full_imgs.shape[3]))
        temp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:img_w, 0:full_imgs.shape[3]] = full_imgs
        full_imgs = temp_full_imgs
    print('shape of new full images:\n' + str(full_imgs.shape))
    return full_imgs

# split all the full imgs into patches with overlap

def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert(len(full_imgs.shape) == 4) # it should be a 4D array
    assert(full_imgs.shape[3] ==1 or full_imgs.shape[3] == 3) # check the channel
    img_h = full_imgs.shape[1]
    img_w = full_imgs.shape[2]
    assert((img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0)
    n_patches_img = ((img_h - patch_h) // stride_h + 1) * ((img_w - patch_w) // stride_w + 1) # // interger division
    n_patches_total = n_patches_img * full_imgs.shape[0]
    print('number of patches on H dimension: ' + str(((img_h - patch_h) // stride_h + 1)))
    print('number of patches on W dimension: ' + str(((img_w - patch_w) // stride_w + 1)))
    print('number of patches per image: ' + str(n_patches_img))
    print('number of total patches in this dataset: ' + str(n_patches_total))
    patches = np.empty((n_patches_total, patch_h, patch_w, full_imgs.shape[3]))
    iter_total = 0
    for i in range(full_imgs.shape[0]):
        for j in range((img_h - patch_h) // stride_h + 1):
            for k in range((img_w - patch_w) // stride_w + 1):
                patch = full_imgs[i, j * stride_h : (j * stride_h + patch_h), k * stride_w : (k * stride_w + patch_w), :]
                patches[iter_total] = patch
                iter_total += 1
    assert(iter_total == n_patches_total)
    return patches

# recompone full imgs using patches with overlap
def recompone_overlap(idens, img_h, img_w, stride_h, stride_w):
    assert(len(idens.shape) == 4) # 4D arrays, slice, h, w, channel
    assert(idens.shape[3] == 1 or idens.shape[3] == 3) # channels should be 1 or 3
    patch_h = idens.shape[1]
    patch_w = idens.shape[2]
    n_patches_h = (img_h - patch_h) // stride_h +1
    n_patches_w = (img_w - patch_w) // stride_w + 1
    n_patches_img = n_patches_h * n_patches_w
    print('n_patches_h: ' + str(n_patches_h))
    print('n_patches_w: ' + str(n_patches_w))
    print('n_patches_per_img: ' + str(n_patches_img))
    assert(idens.shape[0] % n_patches_img == 0)
    n_full_imgs = idens.shape[0] // n_patches_img
    print("According to the dimension inserted, there are " + str(n_full_imgs) + " ful imgs with dimension:(" + str(img_h) + ',' + str(img_w) + ')')
    full_prob = np.zeros((n_full_imgs, img_h, img_w, idens.shape[3])) # initialize propability
    full_sum = np.zeros((n_full_imgs, img_h, img_w, idens.shape[3])) # counter
    
    k = 0 # iterator for all the patches
    for i in range(n_full_imgs):
        for h in range(n_patches_h):
            for j in range(n_patches_w):
                full_prob[i, h * stride_h : h * stride_h + patch_h, j * stride_w : j * stride_w + patch_w, :] += idens[k]
                full_sum[i, h * stride_h : h * stride_h + patch_h, j * stride_w : j * stride_w + patch_w, :] += 1
                k += 1
    assert(k == idens.shape[0])
    assert(np.min(full_sum) >= 1.0) # at least one patch
    final_avg = full_prob / full_sum
    print('shape of final avg: ', final_avg.shape)
    assert(np.min(final_avg) >= 0.0)
    assert(np.max(final_avg) <= 1.0)
    return final_avg

# recompone full_imgs using patches without overlap
def recompone(data, n_patches_h, n_patches_w):
    assert(len(data.shape) == 4) # 4D arrays, slice, h, w, channel
    assert(data.shape[3] == 1 or data.shape[3] == 3) # channel check
    n_patches_img = n_patches_h * n_patches_w
    assert(data.shape[0] % n_patches_img == 0)
    n_full_imgs = data.shape[0] / n_patches_img
    patch_h = data.shape[1]
    patch_w = data.shape[2]
    n_full_imgs = int(n_full_imgs)
    # define and start recompone
    full_recompone = np.empty((n_full_imgs, n_patches_h * patch_h, n_patches_w * patch_w, data.shape[3]))
    k = 0 # iter full img
    s = 0
    while (s < data.shape[0]):
        single_recompone = np.empty((n_patches_h * patch_h, n_patches_w * patch_w, data.shape[3]))
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                single_recompone[i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w, :] =data[s]
                s += 1
        full_recompone[k] = single_recompone
        k += 1
    assert(k == n_full_imgs)
    return full_recompone

# extend/pad the full imgs becasue the split is not exact
def paint_boarder(data, patch_h, patch_w):
    assert(len(data.shape) == 4) # 4D array
    assert(data.shape[3] == 1 or data.shape[3] == 3)
    img_h = data.shape[1]
    img_w = data.shape[2]
    new_img_h = 0
    new_img_w = 0
    if (img_h % patch_h) ==0:
        new_img_h = img_h
    else:
        new_img_h = ((int(img_h) // int(patch_h)) + 1) * patch_h
    if (img_w % patch_w == 0):
        new_img_w = img_w
    else:
        new_img_w = ((int(img_w) // int(patch_w)) + 1) * patch_w
    new_img_h = int(new_img_h)
    new_img_w = int(new_img_w)
    new_data = np.zeros((data.shape[0], new_img_h, new_img_w, data.shape[3]))
    new_data[:, 0 : img_h, 0 : img_w, :] = data[:, :, :, :]
    return new_data

# only return pixels in FOV for both images and identifications(predictions)
def idens_only_FOV(predictions, labels, masks):
    assert(len(predictions.shape) == 4 and len(labels.shape) == 4)
    assert(predictions.shape[0] == labels.shape[0])
    assert(predictions.shape[2] == labels.shape[2])
    assert(predictions.shape[1] == labels.shape[1])
    assert(predictions.shape[3] == 1 and labels.shape[3] == 1)

    new_predictions = []
    new_labels = []
    for i in range(predictions.shape[0]):
        for x in range(predictions.shape[1]):
            for y in range(predictions.shape[2]):
                if inside_FOV(i, x, y, masks) == True:
                    new_predictions.append(predictions[i, x, y, :])
                    new_labels.append(labels[i, x, y, :])
    new_predictions = np.asarray(new_predictions)
    new_labels = np.asarray(new_labels)
    return new_predictions, new_labels

def inside_FOV(i, x, y, masks):
    assert(len(masks.shape) == 4)
    assert(masks.shape[3] == 1)
#     masks = masks/255.  #NOOO!! otherwise with float numbers takes forever!!
    if (x >= masks.shape[1] or y>= masks.shape[2]):
        return False
    if(masks[i, x, y, 0] > 0):
        return True
    else:
        return False

# set a black ring outside the FOV  for all imgs
def kill_boarder(data, original_imgs_boarder_masks):
    assert (len(data.shape)==4)  #4D arrays, slice, channel, height, width
    assert (data.shape[3]==1 or data.shape[3]==3)  #check the channel is 1 or 3
    height = data.shape[1]
    width = data.shape[2]
    for i in range(data.shape[0]):  #loop over the full images
        for x in range(height):
            for y in range(width):
                if inside_FOV(i,x,y,original_imgs_boarder_masks)==False:
                    data[i, x, y, :]=0.0

