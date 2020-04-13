import os
from os import path
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

class PetImagesHandler():
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

    # Get normal data for supervised learning
    def get_normal_data(self, proportion):
        self.load_normal()
        if proportion <= 0 or proportion > 1:
            raise ValueError("0< proportion <= 1")

        np.random.shuffle(self.pet_images)

        x = torch.Tensor([i[0] for i in self.pet_images]).view(-1, self.IMG_SIZE, self.IMG_SIZE)
        X = x / 255.0
        y = torch.Tensor([i[1] for i in self.pet_images])

        # seperate training and test data
        val_size = int(len(X)*proportion)
        train_X = X[:-val_size]
        train_y = y[:-val_size]
        test_X = X[-val_size:]
        test_y = y[-val_size:]

        return train_X, train_y, test_X, test_y

    # Show a random image from the dataset   
    def show_random_image(self):
        self.load_normal()

        randomImage = np.random.randint(len(self.pet_images))
        plt.imshow(self.pet_images[randomImage][0], cmap="gray")
        plt.show()


# Iterator to generate batches of patched data
class PetImagesCPCHandler(PetImagesHandler):
    def __init__(self, batch_size, include_labels=False):
        super().__init__()
        self.load_patched()

        self.batch_size = batch_size
        self.n_batches = len(self.pet_images_patched) // batch_size
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
            self.perm = np.random.permutation(len(self.pet_images_patched))

        if self.n < self.n_batches:
            index = self.perm[self.batch_size*self.n: self.batch_size*self.n + self.batch_size]  
            batch = self.pet_images_patched[index]

            batchImg = torch.Tensor([i[0] for i in batch]).view(self.batch_size, 7, 7, 1, 64, 64)
            batchImg = batchImg / 255.0

            self.n += 1

            if self.include_lables:
                batchLbl = torch.Tensor([i[1] for i in batch])

                return batchImg, batchLbl
            else:
                return batchImg
        else:
            self.n = 0
            raise StopIteration
        

if __name__ == "__main__":
    handler = PetImagesCPCHandler(batch_size=10, include_labels=True)

    for batchImg, batchLbl in handler:
        print(batchImg.shape)
        print(batchLbl)

        break

        



