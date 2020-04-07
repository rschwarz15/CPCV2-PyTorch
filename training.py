import torch
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

    for img in range(batch_size):
        for step in range(pred_steps):
            for row in range(step+1,7):
                for col in range(7):
                    num = 0
                    den = 0

                    target_encoding = encodings[img][row][col][0]
                    predicted_encoding = predictions[img][step][row][col][0]

                    # Dot Products to be softmaxed then cross-entropied
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

                    sm = F.softmax(dots, dim=0)
                    target = torch.eye(set_size)[0].to(device)
                    loss = F.binary_cross_entropy_with_logits(sm, target)
                    total_loss += loss

    print(predicted_encoding)
    print(target_encoding)
    print(other_encoding)
    return total_loss

if __name__ == "__main__":
    device = torch.device("cuda:0")
    batch_size = 5

    # Initialise data handler, network and optimizer
    data = PetImagesCPCHandler(batch_size=batch_size)
    net = CDC(batch_size=batch_size, pred_steps=5).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    h = net.init_hidden().to(device)

    for batch in data:
        enc, pred = net(batch.to(device), h)
        loss = info_nce(enc, pred, 1)
        loss.backward()
        optimizer.step()

        print(loss)
        print("\n\n")
        


