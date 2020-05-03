# For running testing as well
if __name__ != "__main__":
    from CPC.models.PixelCNN import PixelCNN_Autoregressor
    from CPC.models.mobileNetV2_CPC import mobilenet_v2
else:
    from PixelCNN import PixelCNN_Autoregressor
    from mobileNetV2_CPC import mobilenet_v2

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Workaround for Python 3.8 Error -> RuntimeError: error in LoadLibraryA
# Fix being released with torch 1.5
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

# Set device to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CPCEncoder(nn.Module):
    def __init__(self, use_classifier = False):
        super().__init__()

        self.use_classifier = use_classifier

        self.enc = mobilenet_v2(num_classes=2)

    def forward(self, x):
        batch_size = x.shape[0]
        # ENCODER + CLASSIFIER
        if self.use_classifier:
            # input x = batch_size * 7 * 7 * 1 * 16 * 16
            x = x.view(batch_size, 49, 1, 16, 16) # x = batch_size * 49 * 1 * 16 * 16

            # For each image find the mean encoding vector of all 49 patches
            img_mean_encodings = torch.tensor([]).to(device)

            for img in range(batch_size):
                z = self.enc.features(x[img])
                z = nn.functional.adaptive_avg_pool2d(z, 1).reshape(z.shape[0], -1)
                z = z.view(49, 1280) # z = 49 * 1280

                # mean the 7x7 encodings
                mean = torch.mean(z, dim=0).view(1, 1280) # mean = 1 * 1280

                img_mean_encodings = torch.cat([img_mean_encodings, mean], 0) # encodings = batch_size * 1280 
            
            classification = self.enc.classifier(img_mean_encodings) # classification = batch_size * 2
            return F.softmax(classification, dim=1)

        # ENCODER
        else:
            x = self.enc.features(x)
            x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
            return x


class CPC(nn.Module):

    def __init__(self, pred_steps, neg_samples):
        super().__init__()

        self.pred_steps = pred_steps
        self.neg_sample = neg_samples
        self.pred_size = 1280 # size of the output vector from MobileNetV2

        # Define Encoder Network (Reshaped MobileNetV2)
        self.enc = CPCEncoder()

        # Define Autoregressive Network
        self.ar = PixelCNN_Autoregressor(weight_init = False, in_channels=1280)

        # Define Predictive Networks
        self.Wk  = nn.ModuleList([nn.Linear(self.pred_size, self.pred_size) for _ in range(pred_steps)])

    def forward(self, x):
        # input x = batch_size * 7 * 7 * 1 * 16 * 16
        batch_size = x.shape[0]
        x = x.view(batch_size * 49, 1, 16, 16) # (batch_size * 49) * 1 * 16 * 16

        ### FIND ALL ENCODING VECTORS
        encodings = self.enc(x).view(batch_size, 7, 7, 1280)

        ### FIND ALL CONTEXT VECTORS
        # reshape encodings to batch_size * 1280 * 7 * 7 for ar parse
        encodings = encodings.permute(0,3,1,2).contiguous()
        
        contexts = self.ar(encodings) # batch_size * 1280 * 7 * 7

        # reshape contexts and encodings back to batch_size * 7 * 7 * 1280
        contexts = contexts.permute(0,2,3,1).contiguous()
        encodings = encodings.permute(0,2,3,1).contiguous()

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
    #   - whether or not argmax is the positive sample
    def info_nce_loss(self, pred_encoding, encodings, img, row, col):
        batch_size = encodings.shape[0]
        
        # Create list of encodings (1 * positive sample + neg_sample * negative samples)
        encodings_matrix = torch.tensor([]).to(device)

        # Add target encoding in first position of encodings matrix
        target_encoding = encodings[img][row][col].view(1,1280)
        encodings_matrix = torch.cat([encodings_matrix, target_encoding], dim=0)

        # Get neg_sample * negative samples
        for i in range(self.neg_sample):
            img_num = np.random.randint(batch_size)
            row_num = np.random.randint(7)
            col_num = np.random.randint(7)

            # Ensure we don't select the same patch as what is being predicted
            if img_num == img:
                while row_num == row and col_num == col:
                    row_num = np.random.randint(7)
                    col_num = np.random.randint(7)
            
            # Add negative encoding to encodings matrix
            negative_encoding = encodings[img_num][row_num][col_num].view(1,1280)
            encodings_matrix = torch.cat([encodings_matrix, negative_encoding], dim=0)
        
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
    net = CPC(pred_steps=5, neg_samples=16).to(device)

    loss, acc = net(x)
    print(loss)
    print(acc)

    loss.backward()
