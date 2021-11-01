# -*- coding: utf-8 -*-
""" import modules """
import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import os, importlib, datetime, shutil, sys
from tqdm import tqdm
import random, argparse, logging
import sklearn
from model_utils.assist_function import *
from model_utils.extract_patches import *
from torch.utils.data import DataLoader
from data_utils.dataset import H5Dataset
from pathlib import Path
import pdb

#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score

import matplotlib.pylab as pylab
params = {'legend.fontsize': 10,
         'axes.labelsize': 10,
         'axes.titlesize':10,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
pylab.rcParams.update(params)

def test(model, criterion, dataloader, patch_height, patch_width, batch_size):
    mean_loss = []
    mean_acc=[]
    batch_sum, batch_correct = 0,0
    model = model.eval()
    pred = torch.empty(0)
    for j, (test_data, test_label) in tqdm(enumerate(dataloader), total= len(dataloader)):
        test_data, test_label = test_data.permute(0,3,1,2), test_label.permute(0,3,1,2)
        if not args.use_cpu:
            test_data, test_label, pred= test_data.cuda(), test_label.cuda(), pred.cuda()
        test_pred = model(test_data)
        # pred[j*batch_size:j*batch_size+test_data.shape[0],...]
        # import pdb
        # pdb.set_trace()
        pred = torch.cat((pred,test_pred), dim = 0)
        binarized_pred = torch.where(test_pred>0.5, torch.ones_like(test_pred), torch.zeros_like(test_pred))
        batch_correct += torch.sum(binarized_pred==test_label)
        batch_sum += test_label.numel()
        #batch_acc=batch_sum/torch.Size(test_label)
        loss= criterion(test_pred, test_label)
        mean_loss.append(loss)
        # mean_loss.append(batch_acc)
    total_mean_loss = torch.mean(torch.tensor(mean_loss))
    total_mean_acc = torch.tensor(batch_correct/batch_sum)
    print("batch_correct",batch_correct)
    print("batch_sum:",batch_sum)
    return pred, total_mean_loss, total_mean_acc.type(torch.float32)
def inplace_relu(m):
    classname=m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True

def parse_args():
    """ PARAMETERS"""
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--use_cpu", action="store_true", default=False, help = "use cpu mode?")
    parser.add_argument("--gpu", type = str, default="0", help = "specify the gpu device")
    parser.add_argument("--batch_size", type=int, default = 128, help = "batch size in training")
    parser.add_argument("--model", type = str, default = "RDL_Unet", help = "model name")
    parser.add_argument("--log_dir", type = str, default=None, help="base dir")
    parser.add_argument('--notice', type = str, default="...", help="notice for this training process")
    parser.add_argument('--database', type = str, default="DRIVE", help="choose database")
    parser.add_argument("--patch_width", type = int, default = 48, help="patch width, assume equals to patch height")
    parser.add_argument("--patch_height", type = int, default = 48, help="patch width, assume equals to patch height")
    parser.add_argument("--stride_width", type = int, default = 25, help="patch width, assume equals to patch height")
    parser.add_argument("--stride_height", type = int, default = 25, help="patch width, assume equals to patch height")
    parser.add_argument("--mode", type=str, default="double", help="mode of network, single or double?")
    parser.add_argument("--dilation_rate", nargs="+", type=int, default=None, help="dilation rate")
    return parser.parse_args()
def main(args):
    """ hyper-parameter """
    os.environ["CUDA_VISIABLE_DIVICES"] = args.gpu

    def log_string(str):
        logger.info(str)
        print(str)

    """ CREATE DIR """
    experiment_dir = "logs/" + args.log_dir
    export_dir = experiment_dir + "/eval/"
    if not os.path.exists(export_dir):
        os.mkdir(export_dir) 

    """ LOG """
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s -%(levelname)s -%(message)s")
    file_handler = logging.FileHandler("%s/eval.txt" % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string("PARAMETER ...")
    log_string(args)

    """ data dir """
    if args.database == "DRIVE":
        test_imgs_dir = './Database/DRIVE/DRIVE_20_imgs_test.hdf5'
        test_labels_dir1 = './Database/DRIVE/DRIVE_20_GT1_test.hdf5'
        test_labels_dir2 = './Database/DRIVE/DRIVE_20_GT2_test.hdf5'
        test_masks_dir = './Database/DRIVE/DRIVE_20_masks_test.hdf5'
        n_imgs_to_test = 20
        n_group_visiual = 5
    elif args.database == "STARE":
        test_imgs_dir = './Database/STARE/STARE_5_imgs_test.hdf5'
        test_labels_dir1 = './Database/STARE/STARE_5_GT1_test.hdf5'
        test_labels_dir2 = './Database/STARE/STARE_5_GT1_test.hdf5'
        test_masks_dir = './Database/STARE/STARE_5_masks_test.hdf5'
        n_imgs_to_test = 5
        n_group_visiual = 5
    else:
        test_imgs_dir = './Database/CHASE/CHASE_8_imgs_test.hdf5'
        test_labels_dir1 = './Database/CHASE/CHASE_8_GT1_test.hdf5'
        test_labels_dir2 = './Database/CHASE/CHASE_8_GT2_test.hdf5'
        test_masks_dir = './Database/CHASE/CHASE_8_masks_test.hdf5'
        n_imgs_to_test = 8
        n_group_visiual = 4

    test_imgs_orig = load_hdf5(test_imgs_dir)
    test_labels_1 = load_hdf5(test_labels_dir1)
    test_labels_2 = load_hdf5(test_labels_dir2)
    test_masks = load_hdf5(test_masks_dir)
    print(type(test_masks), type(test_imgs_orig), type(test_labels_1), type(test_labels_2))
    print(test_masks.shape, test_imgs_orig.shape, test_labels_1.shape, test_labels_2.shape)

    full_img_height = test_imgs_orig.shape[1]
    full_img_width = test_imgs_orig.shape[2]
    
    # 5 imgs per row
    save_path = export_dir
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    print(full_img_height, full_img_width)

    # from Assist_function import visualize
    original_test_imgs_show = visualize(group_imgs(test_imgs_orig[0:20,:,:,:],n_group_visiual), export_dir + '/original_test_imgs')#.show()
    test_masks_show = visualize(group_imgs(test_masks[0:20,:,:,:],n_group_visiual), export_dir + '/test_masks')#.show()
    test_labels_show = visualize(group_imgs(test_labels_1[0:20,:,:,:],n_group_visiual), export_dir + '/test_labels_1')#.show()

    # split patches
    img_patches_test = None
    new_height = None
    new_width = None
    masks_test  = None
    mask_patches_test = None

    img_patches_test, new_height, new_width, label_patches_test = get_data_testing_overlap(
    Test_imgs_original = test_imgs_dir,  #original
    Test_GT = test_labels_dir1,  #labels
    n_imgs_to_test = n_imgs_to_test,
    patch_height = args.patch_height,
    patch_width = args.patch_width,
    stride_height = args.stride_height,
    stride_width = args.stride_width
    )
    test_set = H5Dataset(img_patches_test, label_patches_test)
    test_dataloader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

    """ load model """
    model_name = os.listdir(experiment_dir + "/logs")[0].split('.')[0]
    # pdb.set_trace()
    Model = importlib.import_module(model_name) # , package= experiment_dir.replace("/",".")
    model = Model.get_model(out_features=[16, 32, 64, 128], kernel_size=3, dilation_rate=args.dilation_rate, mode=args.mode)
    model.apply(inplace_relu)
    criterion = Model.get_loss()
    if not args.use_cpu:
        model = model.cuda()
    checkpoint = torch.load(str(experiment_dir) + "/checkpoints/best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    

    # y_pred = model.predict(img_patches_test, batch_size = 500)
    with torch.no_grad():
        y_pred, total_mean_loss, total_mean_acc = test(model, criterion, test_dataloader, args.patch_height, args.patch_width, args.batch_size)
        log_string(f"loss in testing sample: {total_mean_loss}, accuracy: {total_mean_acc}")
    print(y_pred.shape, img_patches_test.shape)
    y_pred=y_pred.permute(0,2,3,1)
    predicted_img_patches = iden_to_imgs(y_pred, args.patch_height, args.patch_width, mode = 'original')

    # new
    predicted_imgs = None
    test_imgs_orignal = None
    test_labels = None

    predicted_imgs = recompone_overlap(predicted_img_patches.cpu().numpy(), new_height, new_width, args.stride_height, args.stride_width)# predictions
    test_imgs_preprocessed = recompone_overlap(img_patches_test, new_height, new_width, args.stride_height, args.stride_width)
    test_labels = recompone_overlap(label_patches_test, new_height, new_width, args.stride_height, args.stride_width)

    print(type(predicted_imgs), type(test_imgs_preprocessed), type(test_labels))
    print(test_imgs_preprocessed.shape, predicted_imgs.shape, test_labels.shape)
    print(full_img_height, full_img_width)

    # back to original dimensions
    test_imgs_preprocessed = test_imgs_preprocessed[:, 0:full_img_height, 0:full_img_width, :]
    predicted_imgs = predicted_imgs[:, 0:full_img_height, 0:full_img_width, :]
    test_labels = test_labels[:, 0:full_img_height, 0:full_img_width, :]
    print("Original imgs shape: " +str(test_imgs_preprocessed.shape))
    print("Predicted imgs shape: " +str(predicted_imgs.shape))
    print("Ground truth imgs shape: " +str(test_labels.shape))

    all_preprocessed_imgs = visualize(group_imgs(test_imgs_preprocessed,n_group_visiual), export_dir + "/all_preprocessed_imgs")#.show()
    all_predictions = visualize(group_imgs(predicted_imgs,n_group_visiual), export_dir + "/all_predictions.png")#.show()
    all_GT_1 = visualize(group_imgs(test_labels,n_group_visiual), export_dir + "/all_GT_1")#.show()

    #visualize results comparing mask and prediction:
    assert (test_imgs_preprocessed.shape[0] == predicted_imgs.shape[0])
    assert (test_imgs_preprocessed.shape[0] == test_labels.shape[0])
    N_predicted = test_imgs_preprocessed.shape[0]
    group = 1
    assert (N_predicted%group==0)
    for i in range(int(N_predicted/group)):
        img_stripe = group_imgs(test_imgs_preprocessed[i*group:(i*group)+group,:,:,:],group)
        label_stripe = group_imgs(test_labels[i*group:(i*group)+group,:,:,:],group)
        prediction_stripe = group_imgs(predicted_imgs[i*group:(i*group)+group,:,:,:],group)
        total_img = np.concatenate((img_stripe,label_stripe,prediction_stripe),axis=1) # 0 is vertical, 1 is horizontal
        visualize(prediction_stripe, save_path + "/Prediction"+str(i))#.show()
        visualize(total_img, save_path + "/PreprcoessedImg_GroundTruth_Prediction"+str(i))#.show()

    #====== Evaluate the results
    print("\n\n========  Evaluate the results =======================")
    #predictions only inside the FOV
    y_prediction, y_label = idens_only_FOV(predicted_imgs,test_labels, test_masks)  #returns data only inside the FOV
    # print("Calculating results only inside the FOV:")
    # print("y predicted pixels: " +str(y_prediction.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(predicted_imgs.shape[0]*predicted_imgs.shape[1]*predicted_imgs.shape[2]) +" (584*565==329960)")
    # print("y label pixels: " +str(y_label.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(test_labels.shape[1]*test_labels.shape[2]*test_labels.shape[0])+" (584*565==329960)")
    # print("y_prediction: ", y_prediction.shape, type(y_prediction))
    # print('y_label: ',y_label.shape, type(y_label))
    
    #Area under the ROC curve
    fpr, tpr, thresholds = roc_curve(y_label, y_prediction)
    AUC_ROC = roc_auc_score(y_label, y_prediction)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print("\nArea under the ROC curve: " +str(AUC_ROC))
    rooc_curve =plt.figure()
    plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(save_path + "/ROC.png")

    #Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_label, y_prediction)
    precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision,recall)
    print("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
    prec_rec_curve = plt.figure()
    plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(save_path + "/Precision_Recall.png")

    #Confusion matrix
    threshold_confusion = 0.5
    print("\nConfusion matrix:  Costum threshold (for positive) of " +str(threshold_confusion))
    binarized_y_pred = np.where(y_prediction>threshold_confusion, 1,0)
    # y_pred = np.empty((y_prediction.shape[0]))
    # for i in range(y_prediction.shape[0]):
    #     if y_prediction[i]>=threshold_confusion:
    #         y_pred[i]=1
    #     else:
    #         y_pred[i]=0
    confusion = confusion_matrix(y_label, binarized_y_pred)
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion))!=0:
        accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
    print("Global Accuracy: " +str(accuracy))
    specificity = 0
    if float(confusion[0,0]+confusion[0,1])!=0:
        specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
    print("Specificity: " +str(specificity))
    sensitivity = 0
    if float(confusion[1,1]+confusion[1,0])!=0:
        sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
    print("Sensitivity: " +str(sensitivity))
    precision = 0
    if float(confusion[1,1]+confusion[0,1])!=0:
        precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
    print("Precision: " +str(precision))

    #Jaccard similarity index
    jaccard_index = jaccard_score(y_label, binarized_y_pred)
    print("\nJaccard similarity score: " +str(jaccard_index))

    #F1 score
    F1_score = f1_score(y_label, binarized_y_pred, labels=None, average='binary', sample_weight=None)
    print("\nF1 score (F-measure): " +str(F1_score))

    #Save the results
    file_perf = open(export_dir + '/DRIVE_performances_CHASE_126.txt', 'w')
    file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                    + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                    + "\nJaccard similarity score: " +str(jaccard_index)
                    + "\nF1 score (F-measure): " +str(F1_score)
                    +"\n\nConfusion matrix:"
                    +str(confusion)
                    +"\nACCURACY: " +str(accuracy)
                    +"\nSENSITIVITY: " +str(sensitivity)
                    +"\nSPECIFICITY: " +str(specificity)
                    +"\nPRECISION: " +str(precision)
                    )
    file_perf.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)