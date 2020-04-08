import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm
from models import CDC
from data_handlers import PetImagesCPCHandler

# 5 5 7 7 1 1280
def info_nce(encodings, predictions, set_size):
    total_loss = torch.tensor([0.0]).to(device)
    batch_size = predictions.shape[0]
    pred_steps = predictions.shape[1]
    
    # Predicted Encoding dot product will be the first element
    target = torch.tensor([0]).to(device)

    for img in range(batch_size):
        for step in range(pred_steps):
            for row in range(step+1,7):
                for col in range(7):
                    num = 0
                    den = 0

                    target_encoding = encodings[img][row][col][0]
                    predicted_encoding = predictions[img][step][row][col][0]

                    # Find dot products of predicted_encoding and encodings from other images
                    dots = torch.tensor([]).to(device)
                    predicted_dot = torch.dot(predicted_encoding, target_encoding).view(1)
                    dots = torch.cat([dots, predicted_dot], 0)

                    # Get set_size - 1 encodings from other images
                    n = 0
                    incorrect_encodings = torch.tensor([]).to(device)
                    while n < set_size - 1:
                        other_img = np.random.randint(batch_size)

                        # Don't get encodings from the same image
                        # Might change this to allow different encodings from same image?
                        if other_img == img:
                            continue
                        
                        other_encoding = encodings[img][np.random.randint(7)][np.random.randint(7)][0]
                        other_dot = torch.dot(other_encoding, target_encoding).view(1)
                        dots = torch.cat([dots, other_dot], 0)

                        n += 1
                    
                    loss = -1 * F.log_softmax(dots, dim=0)[0]
                    total_loss += loss

    print(predicted_encoding)
    print(target_encoding)
    print(other_encoding)

    # return sum of loss per image
    return total_loss / batch_size

if __name__ == "__main__":
    device = torch.device("cuda:0")
    batch_size = 5
    pred_steps = 5
    set_size = 5

    # Initialise data handler, network and optimizer
    data = PetImagesCPCHandler(batch_size=batch_size)
    net = CDC(batch_size=batch_size, pred_steps=pred_steps).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    h = net.init_hidden().to(device)

    for batch in data:
        enc, pred = net(batch.to(device), h)
        loss = info_nce(enc, pred, set_size=set_size)
        loss.backward()
        optimizer.step()

        print(loss)
        print()
        print()

# Observations:
# Previously the last output of MobileNetV2 was a RELU6 function
# What this meant was that the predictor learnt to maximise all values which would increase dot product and decrease loss
# To counter this, I have changed the last activation function of the encoder to a hardtanh so that positives and negatives are involved

# What now happened was that the predictor learnt to correctly determine pos/neg and then maximised the values
# To address this I introduced a sigmoid function to the predictor

# Now it is just zeroing the prediction...



    
        


