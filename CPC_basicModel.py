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

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class CDC(nn.Module):
    def __init__(self, batch_size, seq_len):
        super().__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = 256
        self.pred_size = 128

        # Define Encoder Network
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv2d(64, self.pred_size, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(self.pred_size),
            nn.ReLU()
        )

        # Define Autoregressive Network
        self.ar = nn.GRU(self.pred_size, self.hidden_size, num_layers=1, bidirectional=False, batch_first=True)

        # Define Predictive Networks
        self.Wk  = nn.ModuleList([nn.Linear(self.hidden_size, self.pred_size) for i in range(seq_len)])

    def forward(self, x, hidden):
        # Encode each patch
        # x = batch_size * 7 * 7 * 64 * 64
        z = self.enc(x)

        return z

    def init_hidden(self):
        #return torch.zeros(2*1, batch_size, 40).cuda()
        return torch.zeros(1, self.batchsize, self.hidden_size)

if __name__ == "__main__":
    pass

