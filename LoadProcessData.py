import os
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
    BUILD = False
    petImagesData = []
    catCount = 0
    dogCount = 0

    def __init__(self):
        if self.BUILD:
            self.make_data()
        else:
            self.load_data()

    # Process the raw images and save as npy
    def make_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    lbl = np.eye(2)[self.LABELS[label]]
                    self.petImagesData.append([np.array(img), lbl])

                    if label == self.CATS: 
                        self.catCount += 1
                    elif label == self.DOGS:
                        self.dogCount += 1
                except Exception as e:
                    pass

        np.random.shuffle(self.petImagesData)
        np.save("petImages.npy", self.petImagesData)
        loaded = True
        print(f'Cats: {self.catCount}')
        print(f'Dogs: {self.dogCount}')

    # Load the data from npy to memory
    def load_data(self):
        self.petImagesData = np.load("petImages.npy", allow_pickle=True)

    # Retrieve labelled data for supervised learning
    def retrieve_labelled_data(self):
        x = torch.Tensor([i[0] for i in self.petImagesData]).view(-1, self.IMG_SIZE, self.IMG_SIZE)
        X = x / 255.0
        y = torch.Tensor([i[1] for i in self.petImagesData])

        # seperate training and test data
        VAL_PCT = 0.1
        val_size = int(len(X)*VAL_PCT)
        train_X = X[:-val_size]
        train_y = y[:-val_size]
        test_X = X[-val_size:]
        test_y = y[-val_size:]

        return train_X, train_y, test_X, test_y

    # Generate 
    def retrieve_unlabelled_data(self):
        pass

    def showRandomImage(self):
        randomImage = np.random.randint(len(self.petImagesData))
        plt.imshow(self.petImagesData[randomImage][0], cmap="gray")
        plt.show()

if __name__ == "__main__":
    dogsvcats = PetImagesHandler()
    dogsvcats.showRandomImage()
