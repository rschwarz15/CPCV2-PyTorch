from CPC.models.PixelCNN import PixelCNN
from CPC.models.MobileNetV2_Encoder import MobileNetV2_Encoder    
from CPC.models.Resnet_Encoder import ResNet_Encoder
from CPC.models.GIM_InfoNCE_Loss import InfoNCE_Loss

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CPC(nn.Module):

    def __init__(self, pred_steps, neg_samples, enc_model="resnet34"):
        super().__init__()
        self.batch_size = None
        self.pred_steps = pred_steps
        self.neg_samples = neg_samples

        # Define Encoder Network
        if enc_model == "resnet34":
            self.enc = ResNet_Encoder(resnet=34, num_classes=2)
            self.pred_size = 256
        elif enc_model == "resnet50":
            self.enc = ResNet_Encoder(resnet=50, num_classes=2)
            self.pred_size = 1024
        elif enc_model == "mobilenet_v2":
            self.enc = MobileNetV2_Encoder(num_classes=2)
            self.pred_size = 1280
        else:
            raise Exception("Undefined encoder choice")

        # Define Autoregressive Network
        self.ar = PixelCNN(in_channels=self.pred_size)

        # Define Predictive and Loss Network
        self.pred_loss = InfoNCE_Loss(in_channels=self.pred_size, 
                                        out_channels=self.pred_size, 
                                        negative_samples=neg_samples, 
                                        pred_steps=pred_steps)

    def forward(self, x):
        # Input x is of shape (batch_size, 1, 64, 64)

        if self.batch_size is None:
            self.batch_size = x.shape[0]

        # Find all encoding vectors
        self.encodings = self.enc(x) # (batch_size, 7, 7, 256)

        # Find all context vectors
        # permute encodings to (batch_size, pred_size, 7, 7) for ar parse
        # then premute back to (batch_size, 7, 7, pred_size) 
        self.encodings = self.encodings.permute(0,3,1,2).contiguous() # (batch_size, pred_size, 7, 7)
        self.contexts = self.ar(self.encodings) # (batch_size, pred_size, 7, 7)

        # Find Contrastive Loss
        loss = self.pred_loss(self.encodings, self.contexts)

        return loss

if __name__ == "__main__":
    from PixelCNN import PixelCNN
    from MobileNetV2_Encoder import MobileNetV2_Encoder    
    from Resnet_Encoder import ResNet_Encoder
    from InfoNCE_Loss import InfoNCE_Loss

    net = CPC(pred_steps=5, 
            neg_samples=16,
            enc_model="resnet34"
            ).to(device)

    x = torch.randn(32, 7, 7, 1, 16, 16).to(device)
    optimizer = optim.Adam(net.parameters(), lr=2e-4) # lr as in paper

    for epoch in range(10):
            net.zero_grad()
            loss = net(x)
            loss.backward()
            optimizer.step()


