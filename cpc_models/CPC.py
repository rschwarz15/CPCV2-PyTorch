from cpc_models.InfoNCE_Loss import InfoNCE_Loss

import torch
import torch.nn as nn

class CPC(nn.Module):

    def __init__(self, encoderNet, arNet, pred_directions, pred_steps, neg_samples):
        super().__init__()
        
        self.pred_directions = pred_directions
        assert 1 <= pred_directions <= 4

        # Define Encoder Network
        self.enc = encoderNet

        # Define Autoregressive Network
        self.ar = arNet

        # Define Predictive + Loss Networks
        self.pred_loss = nn.ModuleList(
            InfoNCE_Loss(pred_steps=pred_steps, neg_samples=neg_samples, in_channels=encoderNet.pred_size)
            for _ in range(self.pred_directions)
        )

    def forward(self, x):
        # Input x = (batch_size, grid_size, grid_size, channels, patch_size, patch_size)

        # Find all encoding vectors
        self.encodings = self.enc(x) # (batch_size, grid_size, grid_size, pred_size)

        # Permute encodings to (batch_size, pred_size, grid_size, grid_size) for ar network
        self.encodings = self.encodings.permute(0,3,1,2).contiguous() # (batch_size, pred_size, grid_size, grid_size)

        # For each direction find context vectors and contrastive loss
        loss = 0
        for i in range(self.pred_directions):
            # rotate encoding 90 degrees clockwise
            if i > 0:
                self.encodings = self.encodings.transpose(2,3).flip(3)

            # Find all context vectors
            self.contexts = self.ar(self.encodings) # (batch_size, pred_size, grid_size, grid_size)

            # Find contrastive loss
            loss += self.pred_loss[i](self.encodings, self.contexts)

        return loss


