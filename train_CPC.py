from CPC.models.GIM_CPC import CPC
from CPC.data.data_handlers import PetImagesHandler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm

encoder_path = "./CPC//TrainedModels/trained_cpc_encoder"
full_cpc_path = "./CPC/TrainedModels/trained_full_cpc"

if __name__ == "__main__":
    # Set device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    batch_size = 32 # paper uses 32 GPUs each with minibatch of 16
    pred_steps = 5 # as in paper
    neg_samples = 16 # this is the defualt used in GIM
    epochs = 10

    # Initialisations
    net = CPC(
        pred_steps=pred_steps, 
        neg_samples=neg_samples,
        enc_model="resnet34"
        ).to(device)

    data = PetImagesHandler(
        batch_size=batch_size, 
        train_proportion=1, 
        test_proportion=0,
        include_labels=False
        )        

    optimizer = optim.Adam(net.parameters(), lr=2e-4) # lr as in paper

    # Load saved network
    LOAD_NET = True
    trained_epochs = 0
    if LOAD_NET:
        trained_epochs = 125
        net.load_state_dict(torch.load(f"{full_cpc_path}_{trained_epochs}.pt"))

    # Train the network
    iter_per_epoch = len(data)
    print_interval = 5

    for epoch in range(epochs):
        prev_time = time.time()
        avg_loss = 0

        for i, batch in enumerate(data):
            net.zero_grad()
            loss = net(batch.to(device))
            loss.backward()
            optimizer.step()
            avg_loss += float(loss)
            
            if i % print_interval == 0:
                div = 1 if ( i == 0 ) else print_interval
                avg_time = (time.time() - prev_time) / (div)
                prev_time = time.time()
                print(f'Epoch {epoch}/{epochs-1}, Iteration {i}/{iter_per_epoch-1}, Loss: {round(float(loss),4)}, Time (s): {round(avg_time, 1)} ')

        print(f'Epoch {epoch}/{epochs-1}, Iteration {i}/{i}, Loss: {round(float(avg_loss),4)}, Time (s): {round(avg_time, 1)} ')
            
    # Save the full network and the encoder
    torch.save(net.state_dict(), f"{full_cpc_path}_{trained_epochs+epochs}.pt")
    torch.save(net.enc.state_dict(), f"{encoder_path}_{trained_epochs+epochs}.pt")

        
    





    
        


