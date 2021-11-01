# RetinalVesselSegmentation

master project: retinal vessel segmentation in fundus images using deep learning

## Update

2021-11-1: original version.

## Introduction

The code is the implementation in Pytorch of my master project.

## Requirement

```
torch>=1.7.0
```

## Train

such as, run the following command in terminal of Linux:

```
python train.py --model RDL_net --database STARE --log_dir STARE04 --num_patches 240000 --epoch 100 --mode double --dilation_rate 2 2 2
```

## Test

such as, run the following command in terminal of Linux:

```
python test.py --model RDL_net --database STARE --log_dir STARE04 --mode double --dilation_rate 2 2 2
```

## Others

Pretraind models (in Tensorflow), well-made databases, and source code (in Tensorflow) can be found [here](https://drive.google.com/drive/folders/1pXLICttL8osAUuXzAHz_BasGxqXSoRxU?usp=sharing).

```
https://drive.google.com/drive/folders/1pXLICttL8osAUuXzAHz_BasGxqXSoRxU?usp=sharing
```

## Author

Chunhui Chen, Chuah Joon Huang, Ali Raza.

## Citation

wait to be added.
