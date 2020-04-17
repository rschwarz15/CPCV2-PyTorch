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

# Workaround for Python 3.8 Error -> RuntimeError: error in LoadLibraryA
# Fix being released with torch 1.5
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')


class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()

        # Get Mobile Net
        self.model = models.mobilenet_v2(num_classes=2)

        # Modify for one channel input
        # Originally: nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):   
        x = self.model(x)
        return F.softmax(x, dim=1)


class CPC(nn.Module):

    def __init__(self, pred_steps, neg_samples):
        super().__init__()

        self.device = torch.device("cuda:0")
        self.pred_steps = pred_steps
        self.neg_sample = neg_samples
        self.hidden_size = 256
        self.pred_size = 1280

        # Define Encoder Network (Reshaped MobileNetV2)
        self.enc = CPC_encoder()

        # Define Autoregressive Network
        self.ar = nn.GRU(self.pred_size, self.hidden_size, num_layers=1, bidirectional=False, batch_first=True)

        # Define Predictive Network
        self.Wk  = nn.ModuleList([nn.Linear(self.hidden_size, self.pred_size) for i in range(pred_steps)])

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def forward(self, x, hidden):
        # input x = batch_size * 7 * 7 * 1 * 64 * 64
        batch_size = x.shape[0]
        x = x.view(batch_size, 49, 1, 64, 64) # x = batch_size * 49 * 1 * 64 * 64

        ### FIND ALL ENCODING VECTORS
        # For each image find encoding vector for all 49 patches
        encodings = torch.tensor([]).to(self.device)

        for img in range(batch_size):
            z = self.enc(x[img]) # z = 49 * 1 * 1280
            encodings = torch.cat([encodings, z.view(1,7,7,1,1280)], 0) # encodings = batch_size * 7 * 7 * 1 * 1280 

        ### FIND ALL CONTEXT VECTORS
        # For each image traverse each column to find context vector for first 6 rows
        contexts = torch.tensor([]).to(self.device)

        for img in range(batch_size):
            img_contexts = torch.tensor([]).to(self.device)

            for col in range(7):
                # reset hidden vector at start of each column
                h = hidden
                col_contexts = torch.tensor([]).to(self.device)

                for row in range(6):
                    c, h = self.ar(encodings[img][row][col].view(1,1,1280), h) # c = 1 * 1 * 256
                    col_contexts = torch.cat([col_contexts, c.view(1,1,1,256)], 0) # col_contexts = 6 * 1 * 1 * 256
                    
                img_contexts = torch.cat([img_contexts, col_contexts], 1) # img_contexts = 6 * 7 * 1 * 256

            contexts = torch.cat([contexts, img_contexts.view(1,6,7,1,256)], 0) # contexts = batch_size * 6 * 7 * 1 * 256

        ### FIND ALL PREDICTED VECTORS AND DETERMINE LOSS
        total_loss, number_preds, number_correct = 0, 0, 0

        # For each prediction find the loss
        for img in range(batch_size):
            for step in range(self.pred_steps):
                for row in range(6-step):
                    for col in range(7):
                        c = contexts[img][row][col]
                        pred = self.Wk[step](c)
                        pred = F.hardtanh(pred)[0]
                        
                        loss, correct = self.info_nce_loss(pred, encodings, img, row, col)
                        
                        total_loss += loss
                        number_correct += correct
                        number_preds += 1

        loss_per_pred = total_loss / number_preds
        acc = number_correct / number_preds
        return loss_per_pred, acc

    def info_nce_loss(self, pred, encodings, img, row, col):
        batch_size = encodings.shape[0]

        target_encoding = encodings[img][row][col][0]

        # Calculate dot products with target_encoding and other_encodings
        dots = torch.tensor([]).to(self.device)
        predicted_dot = torch.dot(pred, target_encoding).view(1)
        dots = torch.cat([dots, predicted_dot], 0)

        # Get neg_sample other_encodings from other images
        n = 0
        while n < self.neg_sample:
            other_img = np.random.randint(batch_size)

            # Don't get encodings from the same image
            # Might change this to allow different encodings from same image?
            if other_img == img:
                continue
            
            other_encoding = encodings[img][np.random.randint(7)][np.random.randint(7)][0]
            other_dot = torch.dot(pred, other_encoding).view(1)
            dots = torch.cat([dots, other_dot], 0)

            n += 1
        
        # Calculate loss
        cross_entropy_loss = -1 * F.log_softmax(dots, dim=0)[0]        

        # Calculate correctness
        if torch.argmax(dots) == 0:
            correct = 1
        else:
            correct = 0

        return cross_entropy_loss, correct


class CPC_encoder(nn.Module):
    def __init__(self, classifier = False):
        super().__init__()

        self.device = torch.device("cuda:0")
        self.classifier = classifier

        self.enc = models.mobilenet_v2(num_classes=2)
        
        # Modify for one channel input
        self.enc.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Change last activation function from ReLU6 to Hardtanh
        self.enc.features[18][2] = nn.Hardtanh()

        # Remove dropout in classifier
        self.enc.classifier = nn.Sequential(nn.Linear(1280, 2, bias=True))

    def forward(self, x):
        batch_size = x.shape[0]
        # ENCODER + CLASSIFIER
        if self.classifier:
            # input x = batch_size * 7 * 7 * 1 * 64 * 64
            x = x.view(batch_size, 49, 1, 64, 64) # x = batch_size * 49 * 1 * 64 * 64

            # For each image find the mean encoding vector of all 49 patches
            img_mean_encodings = torch.tensor([]).to(self.device)

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


device = torch.device("cuda:0")

if __name__ == "__main__":
    #x = torch.randn(5, 7, 7, 1, 64, 64).to(device)
    #net = CPC_encoder(batch_size = 5, classifier=True).to(device)

    x = torch.randn(5, 1, 256, 256).to(device)
    net = MobileNetV2()
    print(net)

    #classification = net(x)
    #print(classification)
