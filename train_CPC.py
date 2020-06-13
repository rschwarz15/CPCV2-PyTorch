from models.CPC import CPC
from data.data_handlers import get_stl10_dataloader
from argparser.train_CPC_argparser import argparser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm
import os


def train():
    iter_per_epoch = len(unsupervised_loader)
    print_interval = 100
    epoch_loss_batches = 200

    for epoch in range(args.trained_epochs+1, args.trained_epochs+args.epochs+1):
        prev_time = time.time()
        epoch_loss = 0

        for i, (batch, lbl) in enumerate(tqdm(unsupervised_loader, disable=args.print_option, dynamic_ncols=True)):
            net.zero_grad()
            loss = net(batch.to(args.device))
            loss = torch.mean(loss, 0) # take mean over all GPUs
            loss.backward()
            optimizer.step()

            # Total loss of last n batches
            if i >= iter_per_epoch - epoch_loss_batches:
                epoch_loss += float(loss)

            if ( (i+1) % print_interval == 0 or i == 0 ) and args.print_option:
                if i == 0:
                    div = 1
                elif i+1 == print_interval:
                    div = print_interval - 1
                else:
                    div = print_interval

                avg_time = (time.time() - prev_time) / div
                prev_time = time.time()

                # Print interval statistics
                print(
                    'Epoch {}/{}, Iteration {}/{}, Loss: {:.4f}, Time(s): {:.2f}'.format(
                        epoch,
                        args.trained_epochs + args.epochs,
                        i+1,
                        iter_per_epoch,
                        loss,
                        avg_time
                    )
                )

        # Results at end of epoch         
        print(
            'Epoch {}/{}, Epoch Loss: {:.4f}'.format(
                epoch,
                args.trained_epochs + args.epochs,
                epoch_loss/epoch_loss_batches,
            )
        )
        f.write(f'{epoch},{epoch_loss/epoch_loss_batches:.4f}\n')

  
def distribute_over_GPUs(args, net):
    num_GPU = torch.cuda.device_count()
    args.batch_size = args.batch_size * num_GPU
    print(f"Running on {num_GPU} GPU(s)")

    net = nn.DataParallel(net).to(args.device)

    return net
      

if __name__ == "__main__":
    args = argparser()

    cpc_path = os.path.join("TrainedModels", args.dataset, "trained_cpc")
    encoder_path = os.path.join("TrainedModels", args.dataset, "trained_encoder")
    log_path = os.path.join("TrainedModels", args.dataset, f"training_log_{args.encoder}_{args.trained_epochs+args.epochs}.txt")
    f = open(log_path, "w")

    # Initialisations
    net = CPC(args)
    net = distribute_over_GPUs(args, net)
    
    unsupervised_loader, _, _, _, _, _ = get_stl10_dataloader(args)
    optimizer = optim.Adam(net.parameters(), lr=args.lr) 

    # Load saved network
    if args.trained_epochs:
        net.load_state_dict(torch.load(f"{cpc_path}_{args.encoder}_{args.trained_epochs}.pt"))

    # Train the network
    ext = ""
    try:
        train()
    except KeyboardInterrupt:
        print("\nEnding Program on Keyboard Interrupt")
        ext = "_incomplete"
            
    # Save the full network and the encoder
    torch.save(net.state_dict(), f"{cpc_path}_{args.encoder}_{args.trained_epochs+args.epochs}{ext}.pt")
    torch.save(net.enc.state_dict(), f"{encoder_path}_{args.encoder}_{args.trained_epochs+args.epochs}{ext}.pt")
    f.close()


        
    





    
        


