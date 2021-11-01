# -*- coding: utf-8 -*-
""" import modules """
import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import os, importlib, datetime, shutil
from tqdm import tqdm
import random, argparse, logging
from model_utils.assist_function import *
from model_utils.extract_patches import *
from torch.utils.data import DataLoader
from data_utils.dataset import H5Dataset
from pathlib import Path

def test(model, criterion, dataloader):
    mean_loss = []
    mean_acc=[]
    batch_sum, batch_correct = 0,0
    model = model.eval()
    for j, (test_data, test_label) in tqdm(enumerate(dataloader), total= len(dataloader)):
        if not args.use_cpu:
            test_data, test_label = test_data.permute(0,3,1,2).cuda(), test_label.permute(0,3,1,2).cuda()
        test_pred = model(test_data)
        loss= criterion(test_pred, test_label)
        mean_loss.append(loss)
        binarized_pred = torch.where(test_pred>0.5, torch.ones_like(test_pred), torch.zeros_like(test_pred))
        batch_correct += torch.sum(binarized_pred==test_label)
        batch_sum += test_label.numel()
    total_mean_loss = torch.mean(torch.tensor(mean_loss))
    total_mean_acc = torch.tensor((batch_correct)/(batch_sum))
    return total_mean_loss, total_mean_acc

def draw_training_curve(train_epoch_loss, test_epoch_loss,train_epoch_acc, test_epoch_acc, save_dir):
    N = len(train_epoch_loss)
    x_axis = torch.arange(N)
    fig = plt.figure(figsize=(15,8))
    plt.plot(x_axis, train_epoch_loss,marker = "^", color = "b", label = "train loss")
    plt.plot(x_axis, test_epoch_loss, marker = "o", color = "r", label = "test loss")
    plt.plot(x_axis, train_epoch_acc,marker = "D", color = "m", label = "train accuracy")
    plt.plot(x_axis, test_epoch_acc, marker = "*", color = "c", label = "test accuracy")
    plt.suptitle("Loss Curves in Training", fontsize = 20, y = 0.95)
    plt.legend(fontsize = 15)
    plt.ylim(0,1)
    plt.savefig(save_dir)

def parse_args():
    """ PARAMETERS"""
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--use_cpu", action="store_true", default=False, help = "use cpu mode?")
    parser.add_argument("--gpu", type = str, default="0", help = "specify the gpu device")
    parser.add_argument("--batch_size", type=int, default = 128, help = "batch size in training")
    parser.add_argument("--model", type = str, default = "RDL_Unet", help = "model name")
    parser.add_argument("--epoch", type = int, default = 30, help="number of epoch in training")
    parser.add_argument("--log_dir", type = str, default=None, help="base dir")
    parser.add_argument("--optimizer", type = str, default="Adam", help="optimizer for training")
    parser.add_argument("--lr", type = float, default=0.0001, help="learning rate in training")
    parser.add_argument("--decay_rate", type=float, default=1e-2, help="decay rate for optimizer")
    parser.add_argument('--notice', type = str, default="...", help="notice for this training process")
    parser.add_argument('--database', type = str, default="DRIVE", help="choose database")
    parser.add_argument('--num_patches', type = int, default=3200, help="num of patches for each database")
    parser.add_argument("--patch_width", type = int, default = 48, help="patch width, assume equals to patch height")
    parser.add_argument("--patch_height", type = int, default = 48, help="patch width, assume equals to patch height")
    parser.add_argument("--valid_proportion", type = float, default = 0.1, help="proportion of valid and train dataset")
    parser.add_argument("--dilation_rate", nargs="+", type=int, default=None, help="dilation rate in conv block")
    parser.add_argument("--mode", type=str, default="double", help="mode of network, single or double?")
    # parser.add_argument("--class_weights", type=list, default=[2, 1], help="class weights, [weight for vessel, weight for background")
    return parser.parse_args()

def plot_learning_curves(record, logdir):
    pd.DataFrame(record.history).plot(figsize = (15,10))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.gca().set_xlabel('Epoches', fontsize=15)
    plt.gca().set_ylabel('Value', fontsize=15)
    plt.gca().legend(fontsize=15)
    plt.gca().set_yticks(np.arange(0, 1, step=0.1))
    plt.gca().set_title('Train Process', fontsize = 18)
    plt.savefig(logdir + '/training_process.png')
    plt.show()
def inplace_relu(m):
    classname= m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True
def xavier(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
def setup_seed(seed):
     torch.manual_seed(seed) # set seed for CPU
     torch.cuda.manual_seed(seed) # set seed for current GPU
     torch.cuda.manual_seed_all(seed) # set seed for all GPU
     np.random.seed(seed)
     random.seed(seed)
     os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现。
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     """
     cudnn.benchmark = True can imporve the efficiency of model, but it will cause inreproducability.
     """
def _init_fn(worker_id): 
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)

def main(args):

    """HYPER PAREMETER"""
    os.environ["CUDA_VISIABLE_DEVICES"] = args.gpu
    def log_string(str):
        logger.info(str)
        print(str)
    """set random seed"""
    setup_seed(2021)

    """CREATE DIR"""
    timestr = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    exp_dir = Path("./logs/")
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath("checkpoints/")
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath("logs/")
    log_dir.mkdir(exist_ok=True)

    """ LOGS """
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler("%s/%s.txt" % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string("PAREMETER ...")
    log_string(args)

    """ load data """
    # get dataset_dir
    if args.database == "DRIVE":
        train_img_dataset_dir = './Database/DRIVE/DRIVE_160_imgs_train.hdf5'
        train_label_dataset_dir = './Database/DRIVE/DRIVE_160_GT_train.hdf5'
        train_mask_dataset_dir = './Database/DRIVE/DRIVE_160_masks_train.hdf5'
    elif args.database=="STARE":
        train_img_dataset_dir = './Database/STARE/STARE_120_imgs_train.hdf5'
        train_label_dataset_dir = './Database/STARE/STARE_120_GT1_train.hdf5'
        train_mask_dataset_dir = './Database/STARE/STARE_120_masks_train.hdf5'
    else:
        train_img_dataset_dir = './Database/CHASE/CHASE_160_imgs_train.hdf5'
        train_label_dataset_dir = './Database/CHASE/CHASE_160_GT2_train.hdf5'
        train_mask_dataset_dir = './Database/CHASE/CHASE_160_masks_train.hdf5'

    train_patches, train_labels = get_data_training(
        Train_imgs_original = train_img_dataset_dir,
        Train_GT = train_label_dataset_dir,  #ground truth
        Train_masks = train_mask_dataset_dir,
        patch_height = args.patch_width,
        patch_width = args.patch_width,
        n_subimgs = args.num_patches,
        inside_FOV = True #select the patches only inside the FOV  (default == True)
    )

    # split dataset
    valid_ind = random.sample(range(train_labels.shape[0]),int(np.floor(args.valid_proportion*train_labels.shape[0])))
    train_ind =  set(range(train_labels.shape[0])) - set(valid_ind)
    train_ind = list(train_ind)

    img_patches_train = train_patches[train_ind,...]
    label_patches_train = train_labels[train_ind,...]
    img_patches_valid = train_patches[valid_ind,...]
    label_patches_valid = train_labels[valid_ind,...]

    train_set = H5Dataset(img_patches_train, label_patches_train)
    test_set = H5Dataset(img_patches_valid, label_patches_valid)
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, worker_init_fn=_init_fn)
    test_dataloader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)


    """ load model """
    Model = importlib.import_module(args.model)
    shutil.copy(f"./{args.model}.py", str(exp_dir))
    shutil.copy("./train.py", str(exp_dir))    

    model = Model.get_model(out_features=[16, 32, 64, 128], kernel_size=3, dilation_rate=args.dilation_rate, mode=args.mode)
    model.apply(inplace_relu)
    model.apply(xavier)
    criterion = Model.get_loss()
    if not args.use_cpu:
        model = model.cuda()
        criterion = criterion.cuda()
    try:
        checkpoint = torch.load(str(exp_dir) + "/checkpoints/best_model.pth")
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        log_string("USE pretrained model")
    except:
        log_string("No existing model, start training from scratch...")
        start_epoch = 0

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas = (0.9,0.999), eps = 1e-08, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size = 2, gamma=0.95, last_epoch = -1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode="min", factor=0.8, patience=3,min_lr=1e-6, verbose=True)
    """ TRAINING """
    log_string("Start training ...")
    train_epoch_loss = []
    train_epoch_acc=[]
    test_epoch_loss = []
    test_epoch_acc=[]
    global_epoch, global_step = 0, 0
    best_loss = 1000.0
    for epoch in range(start_epoch, args.epoch):
        log_string("Epoch %d (%d/%s) with learning rate:%.6f" % (global_epoch+1, epoch+1,args.epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        mean_loss = []
        mean_acc = []
        model = model.train()
        # lr_scheduler.step()
        batch_sum=0
        batch_correct=0
        for step_idx,(imgs,labels) in tqdm(enumerate(train_dataloader),total=len(train_dataloader), smoothing=0.9):
            optimizer.zero_grad()
            # flow_curr = flow_curr.data.numpy()
            if not args.use_cpu:
                imgs = imgs.permute(0,3,1,2).cuda()
                labels = labels.permute(0,3,1,2).cuda()
            pred = model(imgs) #[B, D, N]
            binarized_pred = torch.where(pred>0.5,torch.ones_like(pred),torch.zeros_like(pred))
            batch_correct += torch.sum(binarized_pred==labels)
            batch_sum += labels.numel()
            # labels = labels.long()
            # import pdb
            # pdb.set_trace()
            loss = criterion(pred,labels)
            mean_loss.append(loss)
            # mean_acc.append(batch_acc)
            # import pdb
            # pdb.set_trace()
            loss.backward()
            optimizer.step()
        # import pdb
        # pdb.set_trace()    
        train_mean_loss = torch.mean(torch.tensor(mean_loss))
        train_mean_acc = torch.tensor(batch_correct/batch_sum)
        train_epoch_loss.append(train_mean_loss.data.cpu())
        train_epoch_acc.append(train_mean_acc.cpu())
        log_string("Train Instance Loss: %f, Train Instance Accuracy: %f " % (train_mean_loss, train_mean_acc))

        with torch.no_grad():
            total_test_mean_loss, total_mean_acc = test(model, criterion, test_dataloader)
            # lr_scheduler.step(total_test_mean_loss)
            test_epoch_loss.append(total_test_mean_loss.data.cpu())
            test_epoch_acc.append(total_mean_acc.data.cpu())
            if total_test_mean_loss <= best_loss:
                best_loss = total_test_mean_loss
                best_epoch = epoch + 1
                earlystopping = 0
            else:
                earlystopping += 1
                log_string(f"Count {earlystopping} of 10 for early stopping.")
            log_string("Test Instance Loss: %f, Best Test Instance Loss: %f" % (total_test_mean_loss, best_loss))
            log_string("Test Instance accuracy: %f" % (total_mean_acc))

            if total_test_mean_loss <= best_loss:
                logger.info("Saving model...")
                savepath = str(checkpoints_dir) + "/best_model.pth"
                log_string("Saved at %s" % savepath)
                state = {
                    "epoch": best_epoch,
                    "total_mean_loss": total_test_mean_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
            if earlystopping >=10:
                log_string("Early stopping...")
                break
        lr_scheduler.step(metrics=total_test_mean_loss)
    log_string("End of training...")
    draw_training_curve(train_epoch_loss,test_epoch_loss,train_epoch_acc, test_epoch_acc, save_dir=str(exp_dir)+ "/Training Process.png")

if __name__ == "__main__":
    args = parse_args()
    main(args)