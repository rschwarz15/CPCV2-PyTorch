from cpc_models.MobileNetV2_Encoder import MobileNetV2_Encoder
from cpc_models.ResNetV2_Encoder import PreActResNetN_Encoder
from cpc_models.WideResNet_Encoder import Wide_ResNet_Encoder

from baseline_models.MobileNetV2 import MobileNetV2
from baseline_models.ResNetV2 import PreActResNetN
from baseline_models.WideResNet import Wide_ResNet

from data.data_handlers import *
from argparser.train_classifier_argparser import argparser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np
from tqdm import tqdm


# Process a batch, return accuracy and loss
def fwd_pass(x, y, train=False):
    
    # Run the network
    if train:
        net.zero_grad()
        net.train()
        outputs = net(x)
    else:
        net.eval()
        with torch.no_grad():
            outputs = net(x)

    # Compute accuracy
    matches = [torch.argmax(i) == j for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)

    # Compute loss
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()

    return loss, acc


def train():
    best_acc = 0
    best_epoch = 0
    for epoch in range(1, args.epochs+1):
        epoch_loss = 0
        epoch_acc = 0

        for batch_img, batch_lbl in tqdm(train_loader, dynamic_ncols=True):
            loss, acc = fwd_pass(batch_img.to(args.device), batch_lbl.to(args.device), train=True)
            epoch_loss += loss
            epoch_acc += acc

        epoch_loss /= len(train_loader)
        epoch_acc /= len(train_loader)

        # Get learning rate
        lr = round(optimizer.param_groups[0]['lr'], 10)
        
        # at epoch intervals test the performance
        if epoch % args.test_interval == 0:
            test_loss, test_acc = test()

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                
            print(f"Epoch: {epoch}/{args.epochs} (lr={lr})\n"
                  f"Train: {epoch_loss:.4f}, {epoch_acc*100:.2f}%\n"
                  f"Test:  {test_loss:.4f}, {test_acc*100:.2f}%")
        else:
            print(f"Epoch: {epoch}/{args.epochs} (lr={lr})\n"
                  f"Train: {epoch_loss:.4f}, {epoch_acc*100:.2f}%")
        
        if args.sched_plateau:
            scheduler.step(epoch_loss)
            
            # break training once learning rate is stepped for the third time
            if lr == args.lr / 1e3:
                test_loss, test_acc = test()
                print(f"Test:  {test_loss:.4f}, {test_acc*100:.2f}%")
                print("Training Finished Early on Plateau")
                break
        else:
            scheduler.step()

    print(f"Best Accuracy: {best_acc*100:.2f} - epoch {best_epoch}")


def test():
    total_test_acc = 0
    total_test_loss = 0

    for batch_img, batch_lbl in tqdm(test_loader, dynamic_ncols=True):
        loss, acc = fwd_pass(batch_img.to(args.device), batch_lbl.to(args.device), train=False)
        total_test_acc += acc
        total_test_loss += loss

    return total_test_loss / len(test_loader), total_test_acc / len(test_loader)


if __name__ == "__main__":
    args = argparser()
    print(f"Running on {args.device}")

    # Get selected dataset
    if args.dataset == "stl10":
        _, train_loader, test_loader = get_stl10_dataloader(args, labeled=True)
    elif args.dataset == "cifar10":
        _, train_loader, test_loader = get_cifar10_dataloader(args)
    elif args.dataset == "cifar100":
        _, train_loader, test_loader = get_cifar100_dataloader(args)

    # Define network and optimizer for given train_selection
    if not args.fully_supervised:
        print("Training CPC Classifier")

        # Load the CPC trained encoder (with classifier layer activated)
        if args.encoder[:6] == "resnet":
            net = PreActResNetN_Encoder(args, use_classifier=True)
        elif args.encoder[:10] == "wideresnet":
            parameters = args.encoder.split("-")
            depth = int(parameters[1])
            widen_factor = int(parameters[2])
            net = Wide_ResNet_Encoder(args, depth, widen_factor, use_classifier=True)
        else: # args.encoder == "mobilenetV2"
            net = MobileNetV2_Encoder(args, use_classifier=True)

        colour = "_colour" if (not args.gray) else ""
        encoder_path = os.path.join("TrainedModels", args.dataset, "trained_encoder")
        encoder_path = f"{encoder_path}_{args.encoder}_crop{args.crop}{colour}_grid{args.grid_size}_{args.norm}Norm_{args.pred_directions}dir_aug{args.cpc_patch_aug}_{args.model_num}.pt"
        
        net.load_state_dict(torch.load(encoder_path))
        net.to(args.device)
        print(f"Loaded Model:\n{encoder_path}")

        # Freeze encoder layers
        for name, param in net.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        if args.sgd:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9)
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)

    else:
        print("Training Fully Supervised")

        # Load the network
        if args.encoder[:6] == "resnet":
            net = PreActResNetN(args).to(args.device)
        elif args.encoder[:10] == "wideresnet":
            parameters = args.encoder.split("-")
            depth = int(parameters[1])
            widen_factor = int(parameters[2])
            net = Wide_ResNet(args, depth, widen_factor).to(args.device)
        elif args.encoder == "mobilenetV2":
            net = MobileNetV2(num_classes=args.num_classes).to(args.device)
        else:
            raise Exception("Invalid choice of encoder")
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # define scheduler based on argument inputs
    if args.sched_plateau:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_gamma)
    elif len(args.sched_milestones) == 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.sched_step_size, gamma=args.lr_gamma)
    else:
        milestones = args.sched_milestones.split(',')
        for i in range(0, len(milestones)): 
            milestones[i] = int(milestones[i]) 
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)

    loss_function = nn.CrossEntropyLoss()

    try:
        train()
    except KeyboardInterrupt:
        print("\nEnding Program on Keyboard Interrupt")
