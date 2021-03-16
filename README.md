# Contrastive Predictive Coding
PyTorch implementation of the following papers:

A. v. d. Oord, Y. Li, and O. Vinyals, [Representation learning with contrastive predictive coding](https://arxiv.org/abs/1807.03748)

O. J. H ́enaff, A. Srinivas, J. D. Fauw, A. Razavi, C. Doersch, S. M. A. Eslami, and A. van den Oord [Data-Efficient Image Recognition with Contrastive Predictive Coding](https://arxiv.org/abs/1905.09272)

## Dependencies
* PyTorch (verified with version 1.6.0)
* tqdm
* numpy
* opencv-python (patch aug visualistaion only - not training CPC)

Included is environment.yml 

## Usage
There are two training functions, one for the unsupervised training and one for supervised training.

* Viewing all command-line options
    ```
    python train_classifier.py -h
    ```
    ```
    python train_CPC.py -h
    ```
* Training a fully supervised model
    ```
    python train_classifier.py --fully_supervised --dataset stl10 --encoder resnet18
    ```
* Training Resnet14 on STL10 with CPCV1 - Unsupervised Stage
    ```
    python  train_CPC.py --dataset stl10 --epochs 300 --crop 64-0 --encoder resnet14 --norm none --grid_size 7 --pred_steps 5 --pred_directions 1
    ```
*  Training Wideresnet-28-2 on CIFAR10 with CPCV2 (and a smaller grid size) - Unsupervised Stage
    ```
    python train_CPC.py --dataset  cifar10 --epochs 500 --crop 30-2 --encoder wideresnet-28-2 --norm layer --grid_size 5 --pred_steps 3 --pred_directions 4 --patch_aug 
    ```
*  Training Wideresnet-28-2 on CIFAR10 with CPCV2 (and a smaller grid size) - Supervised Stage with 10,000 labeled images
    ```
    python train_classifier.py --dataset cifar10 --train_size 10000 --epochs 100 --lr 0.1 --crop 30-2 --encoder wideresnet-28-2 --norm layer --grid_size 5 --pred_directions 4 --cpc_patch_aug --patch_aug --model_num 500    
    ```
