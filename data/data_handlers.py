#From:
#https://github.com/loeweX/Greedy_InfoMax/blob/master/GreedyInfoMax/vision/data/get_dataloader.py

import torch
import torchvision.transforms as transforms
import torchvision
from torchvision.transforms import transforms

import os
from os import path
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

aug = {
    "stl10": {
        "randcrop": 64,
        "rand_horizontal_flip": True,
        "grayscale": True,
        "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
        "std": [0.2683, 0.2610, 0.2687],
        "bw_mean": [0.4120],  # values for train+unsupervised combined
        "bw_std": [0.2570],
    },  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
    "cifar10": {
        "randcrop": 64,
        "rand_horizontal_flip": True,
        "grayscale": True,
        "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
        "std": [0.2683, 0.2610, 0.2687],
        "bw_mean": [0.4120],  # values for train+unsupervised combined
        "bw_std": [0.2570],
    },
    "cifar100": {
        "randcrop": 64,
        "rand_horizontal_flip": True,
        "grayscale": True,
        "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
        "std": [0.2683, 0.2610, 0.2687],
        "bw_mean": [0.4120],  # values for train+unsupervised combined
        "bw_std": [0.2570],
    }
}

def get_stl10_dataloader(batch_size, labeled=False, validate=False, download_dataset=False):
    base_folder = "data\stl10_binary"

    num_workers = 0

    # Define Transforms
    transform_train = transforms.Compose(
        [get_transforms(eval=False, aug=aug["stl10"])]
    )
    transform_valid = transforms.Compose(
        [get_transforms(eval=True, aug=aug["stl10"])]
    )

    # Get Datasets
    unsupervised_dataset = torchvision.datasets.STL10(
        base_folder, split="unlabeled", transform=transform_train, download=download_dataset
    )
    train_dataset = torchvision.datasets.STL10(
        base_folder, split="train", transform=transform_train, download=download_dataset
    )
    test_dataset = torchvision.datasets.STL10(
        base_folder, split="test", transform=transform_valid, download=download_dataset
    )

    # Get DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # create train/val split
    if validate:
        print("Use train / val split")

        training_dataset = "train" if labeled else "unlabeled"

        if training_dataset == "train":
            dataset_size = len(train_dataset)
            train_sampler, valid_sampler = create_validation_sampler(dataset_size)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
            )

        elif training_dataset == "unlabeled":
            dataset_size = len(unsupervised_dataset)
            train_sampler, valid_sampler = create_validation_sampler(dataset_size)

            unsupervised_loader = torch.utils.data.DataLoader(
                unsupervised_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
            )

        # overwrite test_dataset and _loader with validation set
        test_dataset = torchvision.datasets.STL10(
            base_folder, split=training_dataset, transform=transform_valid, download=download_dataset,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers,
        )

    else:
        print("Use (train+val) / test split")

    return (
        unsupervised_loader,
        unsupervised_dataset,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )


def create_validation_sampler(dataset_size):
    # Creating data indices for training and validation splits:
    validation_split = 0.2
    shuffle_dataset = True

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


def get_transforms(eval=False, aug=None):
    trans = []

    if aug["randcrop"] and not eval:
        trans.append(transforms.RandomCrop(aug["randcrop"]))

    if aug["randcrop"] and eval:
        trans.append(transforms.CenterCrop(aug["randcrop"]))

    if aug["rand_horizontal_flip"] and not eval:
        trans.append(transforms.RandomHorizontalFlip())

    if aug["grayscale"]:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    elif aug["mean"]:
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
    else:
        trans.append(transforms.ToTensor())

    trans = transforms.Compose(trans)
    return trans

if __name__ == "__main__":
    train_loader, _, supervised_loader, _, test_loader, _ = get_stl10_dataloader(
        batch_size=32,
        labeled=False,
        validate=False,
        download_dataset=True
    )
