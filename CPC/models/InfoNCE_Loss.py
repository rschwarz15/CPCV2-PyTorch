# https://github.com/loeweX/Greedy_InfoMax/blob/master/GreedyInfoMax/vision/models/InfoNCE_Loss.py

from CPC.models.model_utils import makeDeltaOrthogonal
#from model_utils import makeDeltaOrthogonal

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def InfoNCE_Loss(encodings, predictions, targets, neg_samples):  
    batch_size = encodings.shape[0]  
    pred_size = encodings.shape[3]
    number_of_preds = predictions.shape[0]
    
    # Calculate the dot product with target encoding
    positive_dot = torch.mm(predictions, targets.t()).diag().view(number_of_preds, 1)

    # Create matrix of negative encodings
    neg_encodings_matrix = torch.tensor([]).to(device)
    for i in range(neg_samples):
        img_num = np.random.randint(batch_size)
        row_num = np.random.randint(7)
        col_num = np.random.randint(7)
        
        negative_encoding = encodings[img_num][row_num][col_num].view(1,pred_size)
        neg_encodings_matrix = torch.cat([neg_encodings_matrix, negative_encoding], dim=0)

    # Calculate the dot product with negative encodings
    negative_dots = torch.mm(predictions, neg_encodings_matrix.t())
    
    # Collect all dot products
    dots = torch.cat([positive_dot, negative_dots], dim=1)

    # Calculate loss
    target = torch.tensor([0] * number_of_preds).to(device)
    cross_entropy_loss = F.cross_entropy(dots, target)

    # Calculate correctness
    # if torch.argmax(dots) == 0:
    #     correct = 1
    # else:
    #     correct = 0

    return cross_entropy_loss
