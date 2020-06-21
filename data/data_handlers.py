# Based On:
# https://github.com/loeweX/Greedy_InfoMax/blob/master/GreedyInfoMax/vision/data/get_dataloader.py

import torch
import torchvision.transforms as transforms
import torchvision
from torchvision.transforms import transforms

import os
from os import path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

aug = {
    "stl10": {
        "randcrop": 64,
        "rand_horizontal_flip": True,
        "grayscale": True,
        "mean": [0.44087532, 0.42790526, 0.3867924],  # values for train+unsupervised combined
        "std": [0.26826888, 0.2610458, 0.2686684],
        "bw_mean":  [0.42709708], 
        "bw_std": [0.257203],
    }, 
    "cifar10": {
        "randcrop": False,
        "rand_horizontal_flip": True,
        "grayscale": True,
        "mean": [0.49139968, 0.48215827, 0.44653124],
        "std": [0.24703233, 0.24348505, 0.26158768],
        "bw_mean": [0.4808616],
        "bw_std": [0.23919088],
    },
    "cifar100": {
        "randcrop": False,
        "rand_horizontal_flip": True,
        "grayscale": True,
        "mean": [0.5070746, 0.48654896, 0.44091788],
        "std": [0.26733422, 0.25643846, 0.27615058],
        "bw_mean": [0.48748648],
        "bw_std": [0.25063065],
    }
}

def get_stl10_dataloader(args, labeled=False, validate=False):
    data_path = os.path.join("data", "stl10")

    num_workers = 0

    # Define Transforms
    transform_train = transforms.Compose([get_transforms(eval=False, aug=aug["stl10"])])
    transform_valid = transforms.Compose([get_transforms(eval=True, aug=aug["stl10"])])

    # Get Datasets
    unsupervised_dataset = torchvision.datasets.STL10(
        data_path, split="unlabeled", transform=transform_train, download=args.download_dataset
    )
    train_dataset = torchvision.datasets.STL10(
        data_path, split="train", transform=transform_train, download=args.download_dataset
    )
    test_dataset = torchvision.datasets.STL10(
        data_path, split="test", transform=transform_valid, download=args.download_dataset
    )

    # Get DataLoaders
    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
    )

    # create train/val split
    if validate:
        print("Use train / val split")

        training_dataset = "train" if labeled else "unlabeled"

        if training_dataset == "train":
            dataset_size = len(train_dataset)
            train_sampler, valid_sampler = create_validation_sampler(dataset_size)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers,
            )

        elif training_dataset == "unlabeled":
            dataset_size = len(unsupervised_dataset)
            train_sampler, valid_sampler = create_validation_sampler(dataset_size)

            unsupervised_loader = torch.utils.data.DataLoader(
                unsupervised_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers,
            )

        # overwrite test_dataset and _loader with validation set
        test_dataset = torchvision.datasets.STL10(
            data_path, split=training_dataset, transform=transform_valid, download=args.download_dataset,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=num_workers,
        )

    else:
        print("Use (train+val) / test split")

    return (unsupervised_loader, train_loader, test_loader)


def get_cifar_dataloader(args, cifar_classes):
    num_workers = 0

    if cifar_classes == 10:
        data_path = os.path.join("data", "cifar10")

        # Define Transforms
        transform_train = transforms.Compose([get_transforms(eval=False, aug=aug["cifar10"])])
        transform_valid = transforms.Compose([get_transforms(eval=True, aug=aug["cifar10"])])

        # Get Datasets
        unsupervised_dataset = torchvision.datasets.CIFAR10(
            data_path, train=True, transform=transform_train, download=args.download_dataset
        )
        test_dataset = torchvision.datasets.CIFAR10(
            data_path, train=False, transform=transform_valid, download=args.download_dataset
        )

    if cifar_classes == 100:
        data_path = os.path.join("data", "cifar100")

        # Define Transforms
        transform_train = transforms.Compose([get_transforms(eval=False, aug=aug["cifar100"])])
        transform_valid = transforms.Compose([get_transforms(eval=True, aug=aug["cifar100"])])

        # Get Datasets
        unsupervised_dataset = torchvision.datasets.CIFAR100(
            data_path, train=True, transform=transform_train, download=args.download_dataset
        )
        test_dataset = torchvision.datasets.CIFAR100(
            data_path, train=False, transform=transform_valid, download=args.download_dataset
        )

    # Get DataLoaders
    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
    )

    # Take subset of training data for classification training
    try:
        train_size = args.train_size

        indices = list(range(len(unsupervised_dataset)))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(
            unsupervised_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers,
        )
    except AttributeError: # args.train_size is not defined during train_CPC
        # train_loader is not needed during train_CPC
        train_loader = None 

    return (unsupervised_loader, train_loader, test_loader)


def get_cifar10_dataloader(args):
    unsupervised_loader, train_loader, test_loader = get_cifar_dataloader(args, 10)
    return (unsupervised_loader, train_loader, test_loader)


def get_cifar100_dataloader(args):
    unsupervised_loader, train_loader, test_loader = get_cifar_dataloader(args, 100)
    return (unsupervised_loader, train_loader, test_loader)


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


def calculate_normalisation(dataset):
    if dataset == "stl10":
        data_path = os.path.join("data", "stl10")
        
        # RGB
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.STL10(root=data_path, split="train+unlabeled", download=True, transform=train_transform)

        c1 = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))]) # concatenate each channel
        c2 = np.concatenate([np.asarray(train_set[i][0][1]) for i in range(len(train_set))])
        c3 = np.concatenate([np.asarray(train_set[i][0][2]) for i in range(len(train_set))])

        train_mean = [ np.mean(c1, axis=(0, 1,)), np.mean(c2, axis=(0, 1,)), np.mean(c3, axis=(0, 1,))]
        train_std = [ np.std(c1, axis=(0, 1)), np.std(c2, axis=(0, 1)), np.std(c3, axis=(0, 1))]

        # grayscale
        train_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        train_set = torchvision.datasets.STL10(root=data_path, split="train+unlabeled", download=True, transform=train_transform)

        c = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])

        grey_train_mean = [ np.mean(c, axis=(0, 1,)) ]
        grey_train_std = [ np.std(c, axis=(0, 1)) ]

        # [0.44087532, 0.42790526, 0.3867924] [0.26826888, 0.2610458, 0.2686684]
        # [0.42709708] [0.257203]

    elif dataset == "cifar10":
        data_path = os.path.join("data", "cifar10")

        # RGB
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)

        c1 = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])
        c2 = np.concatenate([np.asarray(train_set[i][0][1]) for i in range(len(train_set))])
        c3 = np.concatenate([np.asarray(train_set[i][0][2]) for i in range(len(train_set))])

        train_mean = [ np.mean(c1, axis=(0, 1,)), np.mean(c2, axis=(0, 1,)), np.mean(c3, axis=(0, 1,))]
        train_std = [ np.std(c1, axis=(0, 1)), np.std(c2, axis=(0, 1)), np.std(c3, axis=(0, 1))]

        # grayscale
        train_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform )

        c = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])

        grey_train_mean = [ np.mean(c, axis=(0, 1,)) ]
        grey_train_std = [ np.std(c, axis=(0, 1)) ]

        # [0.49139968, 0.48215827, 0.44653124] [0.24703233, 0.24348505, 0.26158768]
        # [0.4808616] [0.23919088]

    elif dataset == "cifar100":
        data_path = os.path.join("data", "cifar100")

        # RGB
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=train_transform)

        c1 = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])
        c2 = np.concatenate([np.asarray(train_set[i][0][1]) for i in range(len(train_set))])
        c3 = np.concatenate([np.asarray(train_set[i][0][2]) for i in range(len(train_set))])

        train_mean = [ np.mean(c1, axis=(0, 1,)), np.mean(c2, axis=(0, 1,)), np.mean(c3, axis=(0, 1,))]
        train_std = [ np.std(c1, axis=(0, 1)), np.std(c2, axis=(0, 1)), np.std(c3, axis=(0, 1))]

        # grayscale
        train_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=train_transform )

        c = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])

        grey_train_mean = [ np.mean(c, axis=(0, 1,)) ]
        grey_train_std = [ np.std(c, axis=(0, 1)) ]

        # [0.5070746, 0.48654896, 0.44091788] [0.26733422, 0.25643846, 0.27615058]
        # [0.48748648] [0.25063065]

    print(train_mean, train_std)
    print(grey_train_mean, grey_train_std)


if __name__ == "__main__":    
    calculate_normalisation("cifar100")
