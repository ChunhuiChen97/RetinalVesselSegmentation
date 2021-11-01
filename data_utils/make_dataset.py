import numpy as np
import h5py
import os
import cv2 as cv

def write_hdf5(arr, output):
    with h5py.File(output, 'w') as f:
        f.create_dataset('image', data = arr, dtype = arr.dtype)

# convert format of GT and masks from .gif to .png
'''
def format_conversion(orig_dir, dest_dir):
    files = next(os.walk(orig_dir))[2]
    print(files)
    for item in next(os.walk(orig_dir))[2]:
        img = imread(os.path.join(orig_dir, item).replace('\\', '/'))
        imsave(os.path.join(dest_dir, item[:-4]+'.png').replace('\\', '/'), img)
dir = ["test/1st_manual","test/2nd_manual","test/mask","training/1st_manual","training/mask"]
for item in dir:
    orig_dir="../Dataset/"+ item
    dest_dir="../Dataset/"+ item
    format_conversion(orig_dir, dest_dir)
'''

def get_dataset(imgs_dir, GT_dir, mask_dir,dataset,N_imgs,height,width, channels, train_test= 'null'):
    imgs = np.empty((N_imgs, height, width, channels))
    GT = np.empty((N_imgs, height, width))
    masks =np.empty((N_imgs, height, width))
    for path, subfolders, files in os.walk(imgs_dir):
        for i in range(len(files)):
            print(i)
            img_id = os.path.join(imgs_dir, files[i]).replace('\\', '/')
            print(img_id)
            img = cv.imread(img_id)
            imgs[i]=np.asarray(img)
            GT_name = files[i][0:2] + '_manual1.png'
            ground_truth_dir = os.path.join(GT_dir, GT_name).replace('\\', '/')
            print(ground_truth_dir)
            ground_truth = cv.imread(ground_truth_dir)
            ground_truth = cv.cvtColor(ground_truth, cv.COLOR_BGR2GRAY)
            print(ground_truth.shape)
            assert(np.max(ground_truth) == 255 and np.min(ground_truth) == 0)
            GT[i] = np.asarray(ground_truth)
            mask_name = ''
            if train_test == 'train':
                mask_name = files[i][0:2] + '_training_mask.png'
            elif train_test == 'test':
                mask_name = files[i][0:2] + '_test_mask.png'
            else:
                print('please specify train or test !!')
                exit()
            mask_id = os.path.join(mask_dir, mask_name).replace('\\', '/')
            print(mask_id)
            mask = cv.imread(mask_id)
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            assert(np.max(mask) == 255 and np.min(mask) == 0)
            masks[i] = np.asarray(mask)
            
    print('imgs max:',np.max(imgs))
    print('imgs min:' + str(np.min(imgs)))
    print(np.max(GT))
    assert(np.max(GT) == 255)
    assert(np.max(masks) == 255)
    assert(np.min(GT) == 0 and np.min(masks) == 0)
    print('GT and masks are correctly within pixel value range 0-255 (black-white)')

    # reshape to standard tensors
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    assert(imgs.shape == (N_imgs, channels, height, width))
    GT = np.reshape(GT, (N_imgs, 1, height, width))
    masks = np.reshape(masks, (N_imgs, 1, height, width))
    assert(GT.shape == (N_imgs, 1, height, width))
    assert(masks.shape == (N_imgs, 1, height, width))
    return imgs, GT, masks

N_imgs = 20
channels = 3
height = 584
width = 565
dataset_path = 'E:/A08-CNN project/DATABASE/DRIVE/training_testing'
if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

# training image path
orig_train_imgs_dir = '../Database/DRIVE/training/images'
train_GT_dir = '../Database/DRIVE/training/1st_manual'
train_mask_dir = '../Database/DRIVE/training/mask'
# test image path
orig_test_image_dir = '../Database/DRIVE/test/images'
test_1stGT_dir = '../Database/DRIVE/test/1st_manual'
test_2ndGT_dir = '../Database/DRIVE/test/2nd_manual'
test_mask_dir = '../Database/DRIVE/test/mask'

# get train dataset
imgs_train, GT_train, masks_train = get_dataset(orig_train_imgs_dir, train_GT_dir, train_mask_dir, 'train')
print('saving dataset')
write_hdf5(imgs_train, os.path.join(dataset_path, 'DRIVE_dataset_imgs_train.hdf5').replace('\\', '/'))
write_hdf5(GT_train, os.path.join(dataset_path, 'DRIVE_dataset_GT_train.hdf5').replace('\\', '/'))
write_hdf5(masks_train, os.path.join(dataset_path, 'DRIVE_dataset_masks_train.hdf5').replace('\\', '/'))

# get the test dataset

imgs_test, GT_test, masks_test = get_dataset(orig_test_image_dir, test_1stGT_dir, test_mask_dir, 'test')
print('saving dataset')
write_hdf5(imgs_test, os.path.join(dataset_path, 'DRIVE_dataset_imgs_test.hdf5').replace('\\', '/'))
write_hdf5(GT_test, os.path.join(dataset_path, 'DRIVE_dataset_GT_test.hdf5').replace('\\', '/'))
write_hdf5(masks_test, os.path.join(dataset_path, 'DRIVE_dataset_masks_test.hdf5').replace('\\', '/'))