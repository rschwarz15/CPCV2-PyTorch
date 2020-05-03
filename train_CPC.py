from CPC.models.CPC import CPC
from CPC.data.data_handlers import PetImagesCPCHandler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

encoder_path = "./TrainedModels/trained_cpc_encoder.pt"
full_cpc_path = "./TrainedModels/trained_full_cpc.pt"

if __name__ == "__main__":
    # Set device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    batch_size = 50 # paper uses 32 GPUs each with minibatch of 16
    pred_steps = 5 # as in paper
    neg_samples = 16 # this is the defualt used in GIM
    epochs = 3
    torch.backends.cudnn.enabled = False # temporary fix

    # Initialisations
    net = CPC(pred_steps=pred_steps, 
              neg_samples=neg_samples
             ).to(device)
    data = PetImagesCPCHandler(batch_size=batch_size)        
    optimizer = optim.Adam(net.parameters(), lr=2e-4) # lr as in paper

    # Load saved network
    LOAD_NET = False
    if LOAD_NET:
        net.load_state_dict(torch.load(full_cpc_path, map_location=device))

    # Train the network
    iter_per_epoch = len(data)
    best_loss = 1000
    best_acc = 0
    for epoch in range(epochs):
        for i, batch in enumerate(data):
            loss, acc = net(batch.to(device))
            loss.backward()
            optimizer.step()

            if loss < best_loss:
                best_loss = loss

            if acc > best_acc:
                best_acc = acc

            print(f'epoch {epoch} iteration {i}/{iter_per_epoch}: loss={round(float(loss),4)}, acc={round(acc*100,4)}% (best_loss= {round(float(best_loss),4)}, best_acc={round(best_acc*100,4)}%)')
        
    # Save the full network and the encoder
    torch.save(net.state_dict(), full_cpc_path)
    torch.save(net.enc.state_dict(), encoder_path)

        
    





    
        


