# Derived from:
# https://github.com/jzbontar/pixelcnn-pytorch/blob/master/main.py
# https://github.com/rampage644/wavenet/blob/master/wavenet/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

# Run everything on the first GPU available
device = torch.device("cuda:0") 

class PixelCNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


