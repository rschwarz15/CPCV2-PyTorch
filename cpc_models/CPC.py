from cpc_models.InfoNCE_Loss import InfoNCE_Loss

import torch
import torch.nn as nn

class CPC(nn.Module):
    """PyTorch implementation of Contrastive Predictive Coding as in:

    A. v. d. Oord, Y. Li, and O. Vinyals, 
    Representation learning with contrastive predictive coding
    <https://arxiv.org/abs/1807.03748>

    O. J. H ́enaff, A. Srinivas, J. D. Fauw, A. Razavi, C. Doersch, S. M. A. Eslami, and A. van den Oord 
    Data-Efficient Image Recognition with Contrastive Predictive Coding
    <https://arxiv.org/abs/1905.09272>

    Args:
        encoderNet (nn.Module): instance of PyTorch model to be used for encoder network
        arNet (nn.Module): instance of PyTorch model to be used for autoregerssive (context) network
        pred_directions (int): number of directions to perform predictions
        pred_steps (int): number of steps into the future to perform predictions
        neg_samples (int): number of negative samples to be used for contrastive loss
    """

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
            InfoNCE_Loss(pred_steps=pred_steps, neg_samples=neg_samples, in_channels=encoderNet.encoding_size)
            for _ in range(self.pred_directions)
        )

    def forward(self, x):
        # Input x = (batch_size, grid_size, grid_size, channels, patch_size, patch_size)

        # Find all encoding vectors
        self.encodings = self.enc(x) # (batch_size, grid_size, grid_size, encoding_size)

        # Permute encodings to (batch_size, encoding_size, grid_size, grid_size) for ar network
        self.encodings = self.encodings.permute(0,3,1,2).contiguous() # (batch_size, encoding_size, grid_size, grid_size)

        # For each direction find context vectors and contrastive loss
        loss = 0
        for i in range(self.pred_directions):
            # rotate encoding 90 degrees clockwise
            if i > 0:
                self.encodings = self.encodings.transpose(2,3).flip(3)

            # Find all context vectors
            self.contexts = self.ar(self.encodings) # (batch_size, encoding_size, grid_size, grid_size)

            # Find contrastive loss
            loss += self.pred_loss[i](self.encodings, self.contexts)
        loss /= self.pred_directions

        return loss


