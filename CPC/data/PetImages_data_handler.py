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


class PetImages():
    def __init__(self):
        self.CATS = "CPC/data/PetImages/Cat"
        self.DOGS = "CPC/data/PetImages/Dog"
        self.LABELS = {self.CATS: 0, self.DOGS: 1}
        self.IMG_SIZE = 64
        self.pet_images = []
        self.catCount = 0
        self.dogCount = 0
        self.normalise = True

        # Make or load data
        if not path.exists("CPC/data/petImages.npy"):
            resp = input("Make the data (Y/N): ")
            if resp == "Y":
                self.make_npy()

    # Process the raw images and save to npy file
    def make_npy(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    lbl = np.eye(2)[self.LABELS[label]]

                    self.pet_images.append([np.array(img), lbl])

                    # set counts
                    if label == self.CATS: 
                        self.catCount += 1
                    elif label == self.DOGS:
                        self.dogCount += 1
                except Exception as e:
                    pass

        np.save("CPC/data/petImages.npy", self.pet_images)
        print(f'Cats: {self.catCount}')
        print(f'Dogs: {self.dogCount}')


    # Load the normal data from npy file into memory
    def load_data(self):
        self.pet_images = np.load("CPC/data/petImages.npy", allow_pickle=True)

    # Show a random image from the dataset   
    def show_random_image(self):
        self.load_data()

        randomImage = np.random.randint(len(self.pet_images))
        plt.imshow(self.pet_images[randomImage][0], cmap="gray")
        plt.show()

        del self.pet_images


# Iterator to generate batches of normal data
class PetImagesHandler(PetImages):
    def __init__(self, batch_size, train_proportion, test_proportion, include_labels):
        super().__init__()

        assert test_proportion + train_proportion <= 1

        self.load_data()
        
        self.include_labels = include_labels

        # Seperate training and test data
        np.random.shuffle(self.pet_images)
        self.train_data_len = int(train_proportion * len(self.pet_images))
        self.test_data_len = int(test_proportion * len(self.pet_images))

        self.train_data = self.pet_images[:self.train_data_len]
        self.test_data = self.pet_images[self.train_data_len:self.train_data_len + self.test_data_len]

        del self.pet_images

        # set values for iteration
        self.batch_size = batch_size
        self.n_batches = self.train_data_len // batch_size

        self.n = 0
        self.perm = None

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        # If it is the first iteration generate random permutation of data
        if self.n == 0:
            self.perm = np.random.permutation(self.train_data_len)

        if self.n < self.n_batches:    
            # Using the random permutation get a batch of indexes, then get data        
            indexes = self.perm[self.batch_size*self.n: self.batch_size*self.n + self.batch_size]  
            batch = self.train_data[indexes]

            self.n += 1

            batch_img = torch.Tensor([i[0] for i in batch])
            batch_img = batch_img.view(self.batch_size, 1, 64, 64)
            batch_img = batch_img / 255.0

            if self.include_labels:
                batch_lbl = torch.Tensor([i[1] for i in batch])
                return batch_img, batch_lbl

            return batch_img

        else:
            self.n = 0
            raise StopIteration

    def test_batch(self, batch_size):
        start = np.random.randint(self.test_data_len - batch_size)

        batch = self.test_data[start:start+batch_size]

        batch_img = torch.Tensor([i[0] for i in batch]).view(batch_size, 1, 64, 64)
        batch_img = batch_img / 255.0
        batch_lbl = torch.Tensor([i[1] for i in batch])

        return batch_img, batch_lbl


if __name__ == "__main__":
    data = PetImagesHandler(
        batch_size=10, 
        train_proportion=0.1, 
        test_proportion=0.05,
        include_labels=True)
    print(data)
    data.show_random_image()
        





