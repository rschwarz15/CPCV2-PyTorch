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

        # Define Encoder Network (Reshaped MobileNetV2)
        self.enc = models.mobilenet_v2()
        # Modify for one channel input
        self.enc.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Delete classifier
        # Requires forward function to be called as: self.enc.features(x).mean([2, 3])
        # Outputs a 1280D Vector
        del self.enc.classifier

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
        encodings = torch.tensor([])

        for img in range(self.batch_size):
            z = self.enc.features(x[img]).mean([2, 3]) # z = 49 * 1 * 1280
            encodings = torch.cat([encodings, z.view(1,7,7,1,1280)], 0) # encodings = batch_size * 7 * 7 * 1 * 1280 

        ### FIND ALL CONTEXT VECTORS
        # For each image traverse each column to find context vector for first 6 rows
        contexts = torch.tensor([])

        # for each image in batch
        for img in range(self.batch_size):
            img_contexts = torch.tensor([])
            # for each column
            for col in range(7):
                # reset hidden vector at start of each column
                h = hidden
                col_contexts = torch.tensor([])

                # for each row
                for row in range(6):
                    c, h = self.ar(encodings[img][row][col].view(1,1,1280), h) # c = 1 * 1 * 256
                    col_contexts = torch.cat([col_contexts, c.view(1,1,1,256)], 0) # col_contexts = 6 * 1 * 1 * 256
                    
                img_contexts = torch.cat([img_contexts, col_contexts], 1) # img_contexts = 6 * 7 * 1 * 256

            contexts = torch.cat([contexts, img_contexts.view(1,6,7,1,256)], 0) # contexts = batch_size * 6 * 7 * 1 * 256

        ### FIND ALL PREDICTED VECTORS
        # For each image and prediction length build a 7x7 tensor of predictions, pad with rows of zeros
        preds = torch.tensor([])
        for img in range(self.batch_size):
            step_preds = torch.tensor([])

            for pred_step in range(self.pred_steps):
                zeros = torch.zeros(pred_step+1, 7, 1, 1280)
                c = contexts[img][:6-pred_step]
                p = self.Wk[pred_step](c)
                p = torch.cat([zeros, p]) # 7 * 7 * 1 * 1280

                step_preds = torch.cat([step_preds, p.view(1,7,7,1,1280)], 0) # step_preds = pred_steps * 7 * 7 * 1 * 1280

            preds = torch.cat([preds, step_preds.view(1,self.pred_steps,7,7,1,1280)], 0) # preds = batch_size * pred_steps * 7 * 7 * 1 * 1280

        return encodings, preds

if __name__ == "__main__":
    net = CDC(5,5)

    # simulate passing a batch of 5 images
    x = torch.randn(5,7,7,1,64,64)
    h = net.init_hidden()

    encodings, predictions = net(x, h)

    pass

