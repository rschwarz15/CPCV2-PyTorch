import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm
from models import CDC
from data_handlers import PetImagesCPCHandler

if __name__ == "__main__":
    device = torch.device("cuda:0")
    batch_size = 10
    pred_steps = 5
    set_size = 5

    # Initialise data handler, network and optimizer
    data = PetImagesCPCHandler(batch_size=batch_size)
    net = CDC(batch_size=batch_size, pred_steps=pred_steps).to(device)
    optimizer = optim.Adam(net.parameters(), lr=2e-4)

    h = net.init_hidden().to(device)

    for i, batch in enumerate(data):
        loss, acc = net(batch.to(device), h)
        loss.backward()
        optimizer.step()

        print(f'iteration {i}: loss = {float(loss)}, acc={acc}')




    
        


