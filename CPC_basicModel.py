import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

import numpy as np
from tqdm import tqdm
import time
import os
import cv2
import matplotlib.pyplot as plt

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

    def __init__(self, batch_size, pred_steps):
        super().__init__()

        self.batch_size = batch_size
        self.pred_steps = pred_steps
        self.hidden_size = 256
        self.pred_size = 1280

        # Define Encoder Network (Reshaped I/O MobileNetV2)
        self.enc = models.mobilenet_v2()
        # Modify for one channel input
        self.enc.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Delete classifier
        # Requires forward function to be called as: self.enc.features(x).mean([2, 3])
        # Outputs a 1280D Vector
        del self.enc.classifier

        # Define Autoregressive Network
        self.ar = nn.GRU(self.pred_size, self.hidden_size, num_layers=1, bidirectional=False, batch_first=True)

        # Define Predictive Networks
        self.Wk  = nn.ModuleList([nn.Linear(self.hidden_size, self.pred_size) for i in range(pred_steps)])

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def forward(self, x, h):
        # x = 49 * 1 * 64 * 64
        z = self.enc.features(x).mean([2, 3]).view(-1,1,1280) # z = 49 * 1280
        
        # Extract encoding of one patch
        zt = z[0].view(1,1,1280)
        c, h = self.ar(zt, h)

        # TO DO:
        # Loop through "past sequence" to create context vector

        # Predict required steps    

        return z, c, h


if __name__ == "__main__":
    net = CDC(1,0)
    
    x = torch.randn(49,1,64,64)
    h = net.init_hidden()

    z,c,h = net(x, h)
    print(h.shape)
    pass

