from models.PixelCNN import PixelCNN
from models.MobileNetV2_Encoder import MobileNetV2_Encoder    
from models.ResNetV2_Encoder import PreActResNetN_Encoder
from models.WideResNet_Encoder import Wide_ResNet_Encoder
from models.InfoNCE_Loss import InfoNCE_Loss

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class CPC(nn.Module):

    def __init__(self, args):
        super().__init__()

        # Define Encoder Network
        if args.encoder in ("resnet18", "resnet34"):
            self.enc = PreActResNetN_Encoder(args, use_classifier=False)
            self.pred_size = 256
        elif args.encoder in ("resnet50", "resent101", "resnet152"):
            self.enc = PreActResNetN_Encoder(args, use_classifier=False)
            self.pred_size = 1024
        elif args.encoder[:10] == "wideresnet":
            parameters = args.encoder.split("-")
            depth = int(parameters[1])
            widen_factor = int(parameters[2])
            self.enc = Wide_ResNet_Encoder(args, depth, widen_factor, use_classifier=False)
            self.pred_size = 64 * widen_factor
        elif args.encoder == "mobilenetV2":
            self.enc = MobileNetV2_Encoder(args)
            self.pred_size = 1280

        # Define Autoregressive Network
        self.ar = PixelCNN(in_channels=self.pred_size)

        # Define Predictive and Loss Network
        self.pred_loss = InfoNCE_Loss(args, in_channels=self.pred_size)

    def forward(self, x):
        # Input x is of shape (batch_size, 1, 64, 64)

        # Find all encoding vectors
        self.encodings = self.enc(x) # (batch_size, 7, 7, pred_size)

        # Find all context vectors
        # permute encodings to (batch_size, pred_size, 7, 7) for ar parse
        self.encodings = self.encodings.permute(0,3,1,2).contiguous() # (batch_size, pred_size, 7, 7)
        self.contexts = self.ar(self.encodings) # (batch_size, pred_size, 7, 7)

        # Find Contrastive Loss
        loss = self.pred_loss(self.encodings, self.contexts)

        return loss


