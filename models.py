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
        self.model = models.mobilenet_v2()

        # Modify for one channel input, originally:
        # nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Reshape the classifier linear layer for number of outputs, originally:
        # nn.Linear(in_features=1280, out_features=1000, bias=True)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=2, bias=True)

    def forward(self, x):   
        x = self.model.features(x).mean([2, 3])
        return x


class CDC(nn.Module):

    def __init__(self, batch_size=5, pred_steps=5, set_size=5):
        super().__init__()

        self.device = torch.device("cuda:0")
        self.batch_size = batch_size
        self.pred_steps = pred_steps
        self.set_size = set_size
        self.hidden_size = 256
        self.pred_size = 1280

        # Define Encoder Network (Reshaped MobileNetV2)
        self.enc = CDC_encoder()

        # Define Autoregressive Network
        self.ar = nn.GRU(self.pred_size, self.hidden_size, num_layers=1, bidirectional=False, batch_first=True)

        # Define Predictive Network
        self.Wk  = nn.ModuleList([nn.Linear(self.hidden_size, self.pred_size) for i in range(pred_steps)])

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def forward(self, x, hidden):
        # x = batch_size * 7 * 7 * 1 * 64 * 64
        # Reshape x so that all patches of an image can be encoded at once
        x = x.view(self.batch_size, 49, 1, 64, 64) 

        ### FIND ALL ENCODING VECTORS
        # For each image find encoding vector for all 49 patches
        encodings = torch.tensor([]).to(self.device)

        for img in range(self.batch_size):
            z = self.enc(x[img]) # z = 49 * 1 * 1280
            encodings = torch.cat([encodings, z.view(1,7,7,1,1280)], 0) # encodings = batch_size * 7 * 7 * 1 * 1280 

        ### FIND ALL CONTEXT VECTORS
        # For each image traverse each column to find context vector for first 6 rows
        contexts = torch.tensor([]).to(self.device)

        for img in range(self.batch_size):
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
        for img in range(self.batch_size):
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
        target_encoding = encodings[img][row][col][0]

        # Calculate dot products with target_encoding and other_encodings
        dots = torch.tensor([]).to(self.device)
        predicted_dot = torch.dot(pred, target_encoding).view(1)
        dots = torch.cat([dots, predicted_dot], 0)

        # Get set_size - 1 other_encodings from other images
        n = 0
        while n < self.set_size - 1:
            other_img = np.random.randint(self.batch_size)

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


class CDC_encoder(nn.Module):
    def __init__(self, classifier = False):
        super().__init__()
        self.classifier = classifier

        self.enc = models.mobilenet_v2()
        
        # Modify for one channel input
        self.enc.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Change last activation function from ReLU6 to Hardtanh
        self.enc.features[18][2] = nn.Hardtanh()

        # Modify Classifier - 2 outputs + remove dropout
        self.enc.classifier[1] = nn.Linear(in_features=1280, out_features=2, bias=True)
        del self.enc.classifier[0]

    def forward(self, x):
        # If it's acting as a classifier include the classifier layer
        if self.classifier:
            return self.enc(x)
        # Otherwise just act as an encoder
        else:
            return self.enc.features(x).mean([2, 3])
   

if __name__ == "__main__":
    net = CDC_encoder()
    print(net)
