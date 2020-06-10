from CPC.models.GIM_CPC import CPC
from CPC.data.data_handler import get_stl10_dataloader

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
    epochs = 20

    # Initialisations
    net = CPC(
        pred_steps=pred_steps, 
        neg_samples=neg_samples,
        enc_model="resnet34"
        ).to(device)

    unsupervised_loader, _, _, _, _, _ = get_stl10_dataloader(batch_size)
    optimizer = optim.Adam(net.parameters(), lr=2e-4) # lr as in paper

    # Load saved network
    LOAD_NET = True
    trained_epochs = 0
    if LOAD_NET:
        trained_epochs = 30
        net.load_state_dict(torch.load(f"{full_cpc_path}_{trained_epochs}.pt"))

    # Train the network
    iter_per_epoch = len(unsupervised_loader)
    print_interval = 100
    print_interval_stats = False # if False then tqdm is displayed
    epoch_loss_batches = 200

    for epoch in range(trained_epochs+1, trained_epochs+epochs+1):
        prev_time = time.time()
        epoch_loss = 0

        for i, (batch, lbl) in enumerate(tqdm(unsupervised_loader, disable=print_interval_stats, dynamic_ncols=True)):
            net.zero_grad()
            loss = net(batch.to(device))
            loss.backward()
            optimizer.step()

            # Total loss of last n batches
            if i >= iter_per_epoch - epoch_loss_batches:
                epoch_loss += float(loss)

            if (i+1) % print_interval == 0 and print_interval_stats:
                avg_time = (time.time() - prev_time) / print_interval
                prev_time = time.time()
                print(f'Epoch {epoch}/{epochs+trained_epochs}, Iteration {i+1}/{iter_per_epoch}, Loss: {round(float(loss),4)}, Time(s): {round(avg_time, 2)}')

        print(f'Epoch {epoch}/{epochs+trained_epochs}, Epoch Loss: {round(float(epoch_loss/epoch_loss_batches),4)}')
            
    # Save the full network and the encoder
    torch.save(net.state_dict(), f"{full_cpc_path}_{epoch}.pt")
    torch.save(net.enc.state_dict(), f"{encoder_path}_{epoch}.pt")


        
    





    
        


