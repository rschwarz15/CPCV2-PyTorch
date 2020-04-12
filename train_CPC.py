import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm
from models import CDC
from data_handlers import PetImagesCPCHandler

if __name__ == "__main__":
    PATH = "./TrainedModels/trained_cpc_encoder"
    device = torch.device("cuda:0")
    batch_size = 5
    pred_steps = 1
    set_size = 5

    # Initialise data handler, network and optimizer
    data = PetImagesCPCHandler(batch_size=batch_size)
    net = CDC(batch_size=batch_size, 
              pred_steps=pred_steps, 
              set_size=set_size
             ).to(device)
    optimizer = optim.Adam(net.parameters(), lr=2e-4)

    h = net.init_hidden().to(device)

    # Get the batches and train the network
    for i, batch in enumerate(data):
        loss, acc = net(batch.to(device), h)
        loss.backward()
        optimizer.step()

        print(f'iteration {i}: loss = {float(loss)}, acc={acc}')

        if i == 10:
            break
    
    # Save the encoder
    torch.save(net.enc.state_dict(), PATH)

        
    





    
        


