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
    pet_images_labelled = []
    pet_images_unlabelled = []
    catCount = 0
    dogCount = 0
    normalise = True

    def __init__(self):
        # Make or load labelled Data
        if not path.exists("petImagesLabelled.npy") or not path.exists("petImagesUnlabelled.npy"):
            resp = input("Make the labelled data (Y/N): ")
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

                    # Create labelled data
                    self.pet_images_labelled.append([np.array(img), lbl])

                    # Create unlabelled data
                    self.make_patches(img)

                    # set counts
                    if label == self.CATS: 
                        self.catCount += 1
                    elif label == self.DOGS:
                        self.dogCount += 1
                except Exception as e:
                    pass

        np.save("petImagesLabelled.npy", self.pet_images_labelled)
        np.save("petImagesUnlabelled.npy", self.pet_images_unlabelled)
        print(f'Cats: {self.catCount}')
        print(f'Dogs: {self.dogCount}')

    # From an image create the 7x7 patches 
    def make_patches(self, img):
        processed_img = []

        for row in range(7):
            row_patches = []
            for column in range(7):
                patch = [x[column*32:column*32+64] for x in img[row*32:row*32+64]]
                row_patches.append(patch)
            processed_img.append(row_patches)

        self.pet_images_unlabelled.append(np.array(processed_img))

    # Load the labelled data from npy file into memory
    def load_labelled(self):
        self.pet_images_labelled = np.load("petImagesLabelled.npy", allow_pickle=True)
   
    # Load the unlabelled data from npy file into memory
    # This is the 7x7 patches for CPC
    def load_unlabelled(self):
        self.pet_images_unlabelled = np.load("petImagesUnlabelled.npy", allow_pickle=True)

    # Get labelled data for supervised learning
    def get_labelled_data(self, proportion):
        np.random.shuffle(self.pet_images_labelled)
        extract_length = proportion * len(self.pet_images_labelled)
        pet_images_labelled_extract = self.pet_images_labelled[:extract_length]

        x = torch.Tensor([i[0] for i in pet_images_labelled_extract]).view(-1, self.IMG_SIZE, self.IMG_SIZE)
        X = x / 255.0
        y = torch.Tensor([i[1] for i in pet_images_labelled_extract])

        # seperate training and test data
        VAL_PCT = 0.05
        val_size = int(len(X)*VAL_PCT)
        train_X = X[:-val_size]
        train_y = y[:-val_size]
        test_X = X[-val_size:]
        test_y = y[-val_size:]

        return train_X, train_y, test_X, test_y

    # Show a random image from the dataset   
    def show_random_image(self):
        self.load_labelled()

        randomImage = np.random.randint(len(self.pet_images_labelled))
        plt.imshow(self.pet_images_labelled[randomImage][0], cmap="gray")
        plt.show()


# Iterator to generate batches of unlabelled data
class PetImagesCPCHandler(PetImagesHandler):

    def __init__(self, batch_size):
        super().__init__()
        self.load_unlabelled()

        self.batch_size = batch_size
        self.n_batches = len(self.pet_images_unlabelled) // batch_size

        self.n = 0
        self.perm = []

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        # If it is the first iteration generate random permutation of data
        if self.n == 0:
            self.perm = np.random.permutation(len(self.pet_images_unlabelled))

        if self.n < self.n_batches:
            index = self.perm[self.batch_size*self.n: self.batch_size*self.n + self.batch_size]  

            batch = self.pet_images_unlabelled[index]
            batch = torch.tensor(batch).view(self.batch_size, 7, 7, 1, 64, 64)
            batch = batch / 255.0

            self.n += 1

            return batch
        else:
            self.n = 0
            raise StopIteration
        




