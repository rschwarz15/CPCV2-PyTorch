from models.CPC import CPC
from data.data_handlers import *
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
    epoch_loss_batches = 200

    for epoch in range(args.trained_epochs+1, args.trained_epochs+args.epochs+1):
        prev_time = time.time()
        epoch_loss = 0

        for i, (batch, _) in enumerate(tqdm(unsupervised_loader, disable=args.print_option, dynamic_ncols=True)):
            net.zero_grad()
            loss = net(batch.to(args.device))
            loss = torch.mean(loss, 0) # take mean over all GPUs
            loss.backward()
            optimizer.step()

            # Total loss of last n batches
            if i >= iter_per_epoch - epoch_loss_batches:
                epoch_loss += float(loss)

            if ( (i+1) % args.print_interval == 0 or i == 0 ) and args.print_option == 1:
                if i == 0:
                    div = 1
                elif i+1 == args.print_interval:
                    div = args.print_interval - 1
                else:
                    div = args.print_interval

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

  
def distribute_over_GPUs(args, net):
    num_GPU = torch.cuda.device_count()

    if num_GPU == 0:
        raise Exception("No point training without GPU")

    args.batch_size = args.batch_size * num_GPU
    print(f"Running on {num_GPU} GPU(s)")

    net = nn.DataParallel(net).to(args.device)

    return net
      

if __name__ == "__main__":
    args = argparser()

    cpc_path = os.path.join("TrainedModels", args.dataset, "trained_cpc")
    encoder_path = os.path.join("TrainedModels", args.dataset, "trained_encoder")

    # Define Network
    net = CPC(args)
    if args.trained_epochs:
        net.load_state_dict(torch.load(f"{cpc_path}_{args.encoder}_{args.trained_epochs}.pt"))
    net = distribute_over_GPUs(args, net)
            
    # Freeze classifier layer - save memory
    for name, param in net.named_parameters():
        if "classifier" in name:
            param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr) 

    # Get selected dataset
    if args.dataset == "stl10":
        unsupervised_loader, _, _ = get_stl10_dataloader(args)
    elif args.dataset == "cifar10":
        unsupervised_loader, _, _ = get_cifar10_dataloader(args)
    elif args.dataset == "cifar100":
        unsupervised_loader, _, _ = get_cifar100_dataloader(args)
    
    # Train the network
    ext = ""
    try:
        train()
    except KeyboardInterrupt:
        print("\nEnding Program on Keyboard Interrupt")
        ext = "_incomplete"
            
    # Save the full network and the encoder
    net = net.module # unwrap DataParallel
    torch.save(net.state_dict(), f"{cpc_path}_{args.encoder}_{args.trained_epochs+args.epochs}{ext}.pt")
    torch.save(net.enc.state_dict(), f"{encoder_path}_{args.encoder}_{args.trained_epochs+args.epochs}{ext}.pt")


        
    





    
        


