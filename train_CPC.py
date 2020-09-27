from cpc_models.CPC import CPC
from cpc_models.MobileNetV2_Encoder import MobileNetV2_Encoder    
from cpc_models.ResNetV2_Encoder import PreActResNetN_Encoder
from cpc_models.WideResNet_Encoder import Wide_ResNet_Encoder
from data.data_handlers import *
from argparser.train_CPC_argparser import argparser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import time
import numpy as np
from tqdm import tqdm


def train():
    iter_per_epoch = len(unsupervised_loader)
    epoch_loss_batches = round(0.9 * iter_per_epoch)

    for epoch in range(args.trained_epochs+1, args.trained_epochs+args.epochs+1):
        prev_time = time.time()
        epoch_loss = 0

        for i, (batch, _) in enumerate(tqdm(unsupervised_loader, disable=args.print_option, dynamic_ncols=True)):       
            net.zero_grad()
            loss = net(batch.to(args.device))
            loss = torch.mean(loss, dim=0)  # take mean over all GPUs
            loss.backward()
            optimizer.step()

            # Total loss of last 10% of batches
            if i >= epoch_loss_batches:
                epoch_loss += float(loss)

            if ((i+1) % args.print_interval == 0 or i == 0) and args.print_option == 1:
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
                epoch_loss/(iter_per_epoch-epoch_loss_batches),
            )
        )

        # Save net at every 100th epoch
        if epoch % 100 == 0 and epoch != args.trained_epochs+args.epochs:
            save(net, epoch)


def distribute_over_GPUs(args, net):
    num_GPU = torch.cuda.device_count()

    if num_GPU == 0:
        raise Exception("No point training without GPU")

    args.batch_size = args.batch_size * num_GPU
    print(f"Running on {num_GPU} GPU(s)")

    net = nn.DataParallel(net).to(args.device)

    return net


def save(net, epochs):
    saveNet = net.module  # unwrap DataParallel
    torch.save(saveNet.state_dict(),
               f"{cpc_path}_{args.encoder}_crop{args.crop}{colour}_grid{args.grid_size}_{args.norm}Norm_{args.pred_directions}dir_aug{args.patch_aug}_{epochs}{args.model_name_ext}.pt")
    torch.save(saveNet.enc.state_dict(),
               f"{encoder_path}_{args.encoder}_crop{args.crop}{colour}_grid{args.grid_size}_{args.norm}Norm_{args.pred_directions}dir_aug{args.patch_aug}_{epochs}{args.model_name_ext}.pt")


if __name__ == "__main__":
    args = argparser()

    cpc_path = os.path.join("TrainedModels", args.dataset, "trained_cpc")
    encoder_path = os.path.join("TrainedModels", args.dataset, "trained_encoder")
    colour = "_colour" if (not args.gray) else ""

    # Define Encoder Network
    if args.encoder in ("resnet18", "resnet34"):
        enc = PreActResNetN_Encoder(args, use_classifier=False)
    elif args.encoder in ("resnet50", "resent101", "resnet152"):
        enc = PreActResNetN_Encoder(args, use_classifier=False)
    elif args.encoder[:10] == "wideresnet":
        parameters = args.encoder.split("-")
        depth = int(parameters[1])
        widen_factor = int(parameters[2])
        enc = Wide_ResNet_Encoder(args, depth, widen_factor, use_classifier=False)
    elif args.encoder == "mobilenetV2":
        enc = MobileNetV2_Encoder(args)

    # Define Network
    net = CPC(enc, args.pred_directions, args.pred_steps, args.neg_samples)
    if args.trained_epochs:
        net.load_state_dict(torch.load(
            f"{cpc_path}_{args.encoder}_crop{args.crop}{colour}_grid{args.grid_size}_{args.norm}Norm_{args.pred_directions}dir_aug{args.patch_aug}_{args.trained_epochs}{args.model_name_ext}.pt"))
    
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
    print(f"Dataset: {args.dataset}, Encoder: {args.encoder}, Colour: {not args.gray}, Crop: {args.crop}, Grid Size: {args.grid_size}, Norm: {args.norm}, Pred Directions: {args.pred_directions}, Patch Aug: {args.patch_aug}")
    try:
        train()
    except KeyboardInterrupt:
        print("\nEnding Program on Keyboard Interrupt")
        print("\nSaving current model...")
        args.model_name_ext += "_incomplete"

    # Save the full network and the encoder
    save(net, f"{args.trained_epochs+args.epochs}")
