from CPC.models.PixelCNN import PixelCNN
from CPC.models.MobileNetV2_Encoder import MobileNetV2_Encoder    
from CPC.models.Resnet_Encoder import ResNet_Encoder
from CPC.models.InfoNCE_Loss import InfoNCE_Loss

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
            self.enc = ResNet_Encoder(resnet=34, num_classes=10)
            self.pred_size = 256
        elif enc_model == "resnet50":
            self.enc = ResNet_Encoder(resnet=50, num_classes=10)
            self.pred_size = 1024
        elif enc_model == "mobilenet_v2":
            self.enc = MobileNetV2_Encoder(num_classes=10)
            self.pred_size = 1280
        else:
            raise Exception("Undefined encoder choice")

        # Define Autoregressive Network
        self.ar = PixelCNN(in_channels=self.pred_size)

        # Define Predictive Network
        # bias = False: because the paper infers Wk is just a matrix mult
        self.W_k = nn.ModuleList(
            nn.Linear(self.pred_size, self.pred_size, bias=False) 
            for _ in range(self.pred_steps)
        )

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
        self.contexts = self.contexts.permute(0,2,3,1).contiguous() # (batch_size, 7, 7, pred_size) 
        self.encodings = self.encodings.permute(0,2,3,1).contiguous() # (batch_size, 7, 7, pred_size) 

        # Find predictions and loss
        loss = 0
        for step in range(self.pred_steps):
            # XXX GIM skips overlap in first step 

            c = self.contexts[:,:7-step-1].contiguous().view(-1, self.pred_size)
            step_predictions = self.W_k[1](c)
            step_targets = self.encodings[:,step+1:,:,:].contiguous().view(-1, self.pred_size)

            # Find Contrastive Loss
            loss += InfoNCE_Loss(self.encodings, step_predictions, step_targets, self.neg_samples)

        loss = loss / self.pred_steps

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


