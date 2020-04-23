import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from models import CPC
from data_handlers import PetImagesCPCHandler

encoder_path = "./TrainedModels/trained_cpc_encoder.pt"
full_cpc_path = "./TrainedModels/trained_full_cpc.pt"
device = torch.device("cuda:0")

if __name__ == "__main__":
    batch_size = 8
    pred_steps = 3
    neg_samples = 10
    epochs = 3
    torch.backends.cudnn.enabled = False # temporary fix

    # Initialisations
    net = CPC(pred_steps=pred_steps, 
              neg_samples=neg_samples
             ).to(device)
    data = PetImagesCPCHandler(batch_size=batch_size)        
    optimizer = optim.Adam(net.parameters(), lr=2e-4)

    # Load saved network
    LOAD_NET = False
    if LOAD_NET:
        net.load_state_dict(torch.load(full_cpc_path, map_location=device))

    # Train the network
    for epoch in range(epochs):
        for i, batch in enumerate(data):
            loss, acc = net(batch.to(device))
            loss.backward()
            optimizer.step()

            print(f'iteration {i}: loss = {float(loss)}, acc={acc}')
        
    # Save the full network and the encoder
    torch.save(net.state_dict(), full_cpc_path)
    torch.save(net.enc.state_dict(), encoder_path)

        
    





    
        


