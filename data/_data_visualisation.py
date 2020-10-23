import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as ds
from torch.utils.data import DataLoader

import os
import numpy as np
import matplotlib.pyplot as plt

dataset = input("Dataset: ")
loader_num = input("Loader (0=training,1=testing,other=unsupervised): ")
rows = 2
columns = 10
batch_size = rows * columns

# Get selected dataset
if dataset == "stl10":
    dataset_labels = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    data_path = os.path.join("./data", "stl10")

    unsupervised_dataset = ds.STL10(data_path, split="unlabeled", transform=transforms.ToTensor(), download=False)
    train_dataset = ds.STL10(data_path, split="train", transform=transforms.ToTensor(), download=False)
    test_dataset = ds.STL10(data_path, split="test", transform=transforms.ToTensor(), download=False)

    unsupervised_loader = DataLoader(unsupervised_dataset, batch_size=batch_size, shuffle=True)

elif dataset in ("cifar10", "cifar100"):
    data_path = os.path.join("./data", dataset)

    if dataset == "cifar10":
        dataset_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        ds_cifar = ds.CIFAR10
    else:
        dataset_labels = [""] * 100
        ds_cifar = ds.CIFAR100

    train_dataset = ds_cifar(data_path, train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = ds_cifar(data_path, train=False, transform=transforms.ToTensor(), download=False)
    unsupervised_loader = None
else:
    raise Exception("Dataset doesn't exist")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("Data Loaded")

# View images
if loader_num == "0":
    loader = train_loader
elif loader_num == "1":
    loader = test_loader
else:
    loader = unsupervised_loader

image_batch, labels = next(iter(loader))

fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(10, 2))

for row in range(rows):
    for col in range(columns):
        label_index = labels[row * columns + col]

        if label_index != -1:
            label = dataset_labels[label_index]
        else:
            label = "" # For unsupervised stl10 which has -1 for all labels

        img = image_batch[row * columns + col]
        img = np.moveaxis(img.numpy(),0,2)
        axes[row][col].imshow(img)
        axes[row][col].title.set_text(label)
        axes[row][col].axis("off")

plt.show()

# Count for each classes

