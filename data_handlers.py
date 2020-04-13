import os
from os import path
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

class PetImages():
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    IMG_SIZE = 256
    pet_images = []
    pet_images_patched = []
    catCount = 0
    dogCount = 0
    normalise = True

    def __init__(self):
        # Make or load data
        if not path.exists("petImages.npy") or not path.exists("petImagesPatched.npy"):
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

                    # Create normal data
                    self.pet_images.append([np.array(img), lbl])

                    # Create patched data
                    self.make_patched(img, lbl)

                    # set counts
                    if label == self.CATS: 
                        self.catCount += 1
                    elif label == self.DOGS:
                        self.dogCount += 1
                except Exception as e:
                    pass

        np.save("petImages.npy", self.pet_images)
        np.save("petImagesPatched.npy", self.pet_images_patched)
        print(f'Cats: {self.catCount}')
        print(f'Dogs: {self.dogCount}')

    # From an image create the 7x7 patches 
    def make_patched(self, img, lbl):
        processed_img = []

        for row in range(7):
            row_patches = []
            for column in range(7):
                patch = [x[column*32:column*32+64] for x in img[row*32:row*32+64]]
                row_patches.append(patch)
            processed_img.append(row_patches)

        self.pet_images_patched.append([np.array(processed_img), lbl])

    # Load the normal data from npy file into memory
    def load_normal(self):
        self.pet_images = np.load("petImages.npy", allow_pickle=True)
   
    # Load the patched data from npy file into memory
    # This is the 7x7 patches for CPC
    def load_patched(self):
        self.pet_images_patched = np.load("petImagesPatched.npy", allow_pickle=True)

    # Show a random image from the dataset   
    def show_random_image(self):
        self.load_normal()

        randomImage = np.random.randint(len(self.pet_images))
        plt.imshow(self.pet_images[randomImage][0], cmap="gray")
        plt.show()

        del self.pet_images


# Iterator to generate batches of normal data
class PetImagesNormalHandler(PetImages):
    def __init__(self, batch_size, train_proportion, test_proportion):
        super().__init__()
        self.load_normal()

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
        self.perm = []

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        # If it is the first iteration generate random permutation of data
        if self.n == 0:
            self.perm = np.random.permutation(len(self.train_data))

        if self.n < self.n_batches:
            index = self.perm[self.batch_size*self.n: self.batch_size*self.n + self.batch_size]  
            batch = self.train_data[index]

            batch_img = torch.Tensor([i[0] for i in batch]).view(self.batch_size, 1, 256, 256)
            batch_img = batch_img / 255.0
            batch_lbl = torch.Tensor([i[1] for i in batch])

            self.n += 1

            return batch_img, batch_lbl

        else:
            self.n = 0
            raise StopIteration

    def test_batch(self, batch_size):
        start = np.random.randint(self.test_data_len - batch_size)

        batch = self.test_data[start:start+batch_size]

        batch_img = torch.Tensor([i[0] for i in batch]).view(self.batch_size, 1, 256, 256)
        batch_img = batch_img / 255.0
        batch_lbl = torch.Tensor([i[1] for i in batch])

        return batch_img, batch_lbl

# Iterator to generate batches of patched data
# Defualt input parameters are for cpc training using all data unlabelled
class PetImagesCPCHandler(PetImages):
    def __init__(self, batch_size, include_labels=False, train_proportion=1, test_proportion=0):
        super().__init__()
        self.load_patched()

        # Seperate training and test data
        np.random.shuffle(self.pet_images_patched)
        self.train_data_len = int(train_proportion * len(self.pet_images_patched))
        self.test_data_len = int(test_proportion * len(self.pet_images_patched))

        self.train_data = self.pet_images_patched[:self.train_data_len]
        self.test_data = self.pet_images_patched[self.train_data_len:self.train_data_len + self.test_data_len]

        del self.pet_images_patched

        # set values for iteration
        self.batch_size = batch_size
        self.n_batches = self.train_data_len // batch_size
        self.include_lables = include_labels

        self.n = 0
        self.perm = []

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        # If it is the first iteration generate random permutation of data
        if self.n == 0:
            self.perm = np.random.permutation(len(self.train_data))

        if self.n < self.n_batches:
            index = self.perm[self.batch_size*self.n: self.batch_size*self.n + self.batch_size]  
            batch = self.train_data[index]

            batch_img = torch.Tensor([i[0] for i in batch]).view(self.batch_size, 7, 7, 1, 64, 64)
            batch_img = batch_img / 255.0

            self.n += 1

            if self.include_lables:
                batch_lbl = torch.Tensor([i[1] for i in batch])

                return batch_img, batch_lbl
            else:
                return batch_img
        else:
            self.n = 0
            raise StopIteration

    def test_batch(self, batch_size):
        start = np.random.randint(self.test_data_len - batch_size)

        batch = self.test_data[start:start+batch_size]

        batch_img = torch.Tensor([i[0] for i in batch]).view(self.batch_size, 7, 7, 1, 64, 64)
        batch_img = batch_img / 255.0
        batch_lbl = torch.Tensor([i[1] for i in batch])

        return batch_img, batch_lbl
        

if __name__ == "__main__":
    data = PetImagesNormalHandler(batch_size=10, train_proportion=0.1, test_proportion=0.05)

    for batch_img, batch_lbl in tqdm(data):
        pass
        

        



