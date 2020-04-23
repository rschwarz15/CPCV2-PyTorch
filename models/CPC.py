import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

import os
import cv2
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.pixelCNN import PixelCNN

# Workaround for Python 3.8 Error -> RuntimeError: error in LoadLibraryA
# Fix being released with torch 1.5
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

# Run everything on the first GPU available
device = torch.device("cuda:0") 

class CPC_encoder(nn.Module):
    def __init__(self, classifier = False):
        super().__init__()

        self.classifier = classifier

        self.enc = models.mobilenet_v2(num_classes=2)
        
        # Modify for one channel input
        self.enc.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Change last activation function from ReLU6 to Hardtanh
        # self.enc.features[18][2] = nn.Hardtanh()

        # ????? Remove dropout in classifier ?????
        self.enc.classifier = nn.Sequential(nn.Linear(1280, 2, bias=True))

    def forward(self, x):
        batch_size = x.shape[0]
        # ENCODER + CLASSIFIER
        if self.classifier:
            # input x = batch_size * 7 * 7 * 1 * 16 * 16
            x = x.view(batch_size, 49, 1, 16, 16) # x = batch_size * 49 * 1 * 16 * 16

            # For each image find the mean encoding vector of all 49 patches
            img_mean_encodings = torch.tensor([]).to(device)

            for img in range(batch_size):
                z = self.enc.features(x[img]).mean([2, 3]).view(49, 1280) # z = 49 * 1280

                # mean the 7x7 encodings
                mean = torch.mean(z, dim=0).view(1, 1280) # mean = 1 * 1280

                img_mean_encodings = torch.cat([img_mean_encodings, mean], 0) # encodings = batch_size * 1280 
            
            classification = self.enc.classifier(img_mean_encodings) # classification = batch_size * 2
            return F.softmax(classification, dim=1)

        # ENCODER
        else:
            return self.enc.features(x).mean([2, 3])


class CPC(nn.Module):

    def __init__(self, pred_steps, neg_samples):
        super().__init__()

        self.pred_steps = pred_steps
        self.neg_sample = neg_samples
        self.pred_size = 1280 # size of the output vector from MobileNetV2

        # Define Encoder Network (Reshaped MobileNetV2)
        self.enc = CPC_encoder()

        # Define Autoregressive Network
        #self.ar = PixelCNN()

        # Define Predictive Networkyo
        self.Wk  = nn.ModuleList([nn.Linear(self.pred_size, self.pred_size) for i in range(pred_steps)])

    def forward(self, x):
        # input x = batch_size * 7 * 7 * 1 * 16 * 16
        batch_size = x.shape[0]
        x = x.view(batch_size, 49, 1, 16, 16) # x = batch_size * 49 * 1 * 16 * 16

        ### FIND ALL ENCODING VECTORS
        # For each image find encoding vector for all 49 patches
        encodings = torch.tensor([]).to(device)

        for img in range(batch_size):
            z = self.enc(x[img]) # z = 49 * 1 * 1280
            encodings = torch.cat([encodings, z.view(1,7,7,1280)], 0) # encodings = batch_size * 7 * 7 * 1280 

        ### FIND ALL CONTEXT VECTORS
        # reshape encodings to batch_size * 1280 * 7 * 7 for ar parse
        encodings = encodings.permute(0,3,1,2) 
        contexts = self.ar(encodings) # contexts = batch_size * 1280 * 7 * 7

        # reshape contexts and encodings back to batch_size * 7 * 7 * 1280
        contexts = contexts.permute(0,2,3,1)
        encodings = encodings.permute(0,2,3,1)

        ### FIND ALL PREDICTED VECTORS AND DETERMINE LOSS
        total_loss, number_preds, number_correct = 0, 0, 0

        # For each prediction find the loss
        for img in range(batch_size):
            for step in range(self.pred_steps):
                for row in range(6-step):
                    for col in range(7):
                        c = contexts[img][row][col]
                        pred_encoding = self.Wk[step](c)
                        pred_row = row + step + 1

                        loss, correct = self.info_nce_loss(pred_encoding, encodings, img, pred_row, col)
                        total_loss += loss
                        number_correct += correct
                        number_preds += 1

        loss_per_pred = total_loss / number_preds
        acc = number_correct / number_preds
        return loss_per_pred, acc

    # Contrastive loss function
    # Takes as input:
    #   - predicted encoding
    #   - all other encodings
    #   - the img, row and col of the predicted encoding
    # Returns:
    #   - loss as determined by infoNCE 
    #   - accuracy of argmax being the positive sample
    def info_nce_loss(self, pred_encoding, encodings, img, row, col):
        batch_size = encodings.shape[0]
        
        # Create list of encodings (1 * positive sample + neg_sample * negative samples)
        encodings_matrix = torch.tensor([]).to(device)

        # Add target encoding in first position of encodings matrix
        target_encoding = encodings[img][row][col].view(1,1280)
        encodings_matrix = torch.cat([encodings_matrix, target_encoding], dim=0)

        # Get neg_sample * negative samples (20% being from same image)
        n = 0
        same_img_encodings = self.neg_sample // 5
        while n < self.neg_sample:
            # Get the 20% from the same image - ensure it is not the same patch
            if n < same_img_encodings:
                img_num = img

                row_num = np.random.randint(7)
                col_num = np.random.randint(7)

                while row_num == row and col_num == col:
                    row_num = np.random.randint(7)
                    col_num = np.random.randint(7)
                
            # Get 80% from other images
            else:
                img_num = np.random.randint(batch_size)

                while img_num == img:
                    img_num = np.random.randint(batch_size)
                
                row_num = np.random.randint(7)
                col_num = np.random.randint(7)
            
            # Add negative encoding to encodings matrix
            negative_encoding = encodings[img_num][row_num][col_num].view(1,1280)
            encodings_matrix = torch.cat([encodings_matrix, negative_encoding], dim=0)

            n += 1
        
        # Calculate all the dot products (reshape to 1D)
        dots = torch.mm(pred_encoding.view(1,1280), encodings_matrix.t()).view(self.neg_sample+1)

        # Calculate loss
        cross_entropy_loss = -1 * F.log_softmax(dots, dim=0)[0]  

        # Calculate correctness
        if torch.argmax(dots) == 0:
            correct = 1
        else:
            correct = 0

        # print(pred_encoding)
        # print(target_encoding)
        # print(negative_encoding)

        return cross_entropy_loss, correct


# Testing
if __name__ == "__main__":
    x = torch.randn(8, 7, 7, 1, 16, 16).to(device)
    net = CPC(pred_steps=3, neg_samples=5).to(device)

    torch.backends.cudnn.enabled = False # temporary fix

    loss, acc = net(x)
    print(loss)
    print(acc)

    loss.backward()
