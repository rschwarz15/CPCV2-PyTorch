from models.MobileNetV2_Encoder import MobileNetV2_Encoder
from models.MobileNetV2 import MobileNetV2
from models.ResNetV2_Encoder import PreActResNetN_Encoder
from models.ResNetV2 import PreActResNetN
from models.WideResNet_Encoder import Wide_ResNet_Encoder
from models.WideResNet import Wide_ResNet

from data.data_handlers import *
from argparser.train_classifier_argparser import argparser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

# Process a batch, return accuracy and loss
def fwd_pass(X, y, train=False):
    # Run the network
    if train:
        net.zero_grad()
        net.train()
        outputs = net(X)

    if not train:
        net.eval()
        with torch.no_grad():
            outputs = net(X)

    # Compute accuracy
    matches = [torch.argmax(i) == j for i, j in zip(outputs,y)]
    acc = matches.count(True)/len(matches)

    # Compute loss
    loss = loss_function(outputs, y) 

    if train:
        loss.backward()
        optimizer.step()

    return loss, acc 


# Train net
def train():
    best_acc = 0
    best_epoch = 0
    for epoch in range(1, args.epochs+1):

        for batch_img, batch_lbl in tqdm(train_loader, dynamic_ncols=True):
            loss, acc = fwd_pass(batch_img.to(args.device), batch_lbl.to(args.device), train=True)    

        # at epoch intervals test the performance
        if epoch % args.test_interval == 0:
            test_loss, test_acc = test()

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                
            print(f"Epoch: {epoch}/{args.epochs}\n"
                f"Train: {loss:.4f}, {acc*100:.2f}%\n"
                f"Test:  {test_loss:.4f}, {test_acc*100:.2f}%")
        else:
            print(f"Epoch: {epoch}/{args.epochs}\n"
                f"Train: {loss:.4f}, {acc*100:.2f}%")

        scheduler.step()
        
        # for param_group in optimizer.param_groups:
        #     print(param_group['lr'])
    
    print(f"Best Accuracy: {best_acc*100:.2f} - epoch {best_epoch}")


# Process test data to find test loss/accuracy
def test():
    total_test_acc = 0
    total_test_loss = 0

    # Process all of the test data
    for batch_img, batch_lbl in tqdm(test_loader, dynamic_ncols=True):
        loss, acc = fwd_pass(batch_img.to(args.device), batch_lbl.to(args.device))  
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

        if args.model_num == -1:
            raise Exception("For Training CPC model_num needs to be set")

        # Load the CPC trained encoder (with classifier layer activated)
        if args.encoder[:6] == "resnet":
            net = PreActResNetN_Encoder(args, use_classifier=True).to(args.device)
        elif args.encoder[:10] == "wideresnet":
            parameters = args.encoder.split("-")
            depth = parameters[1]
            widen_factor = parameters[2]
            net = Wide_ResNet_Encoder(args, depth, widen_factor, use_classifier=True)
        elif args.encoder == "mobielnetV2":
            net = MobileNetV2_Encoder(args, use_classifier=True).to(args.device)
        
        encoder_path = os.path.join("TrainedModels", args.dataset, "trained_encoder")
        net.load_state_dict(torch.load(f"{encoder_path}_{args.encoder}_{args.norm}Norm_{args.model_num}.pt"))        
        net = net.to(args.device)

        # Freeze encoder layers
        for name, param in net.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        #optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9)
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

        optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # define scheduler based on argument inputs
    if len(args.sched_milestones) == 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.sched_step_size, gamma=0.1)
    else:
        milestones = args.sched_milestones.split(',')
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    loss_function = nn.NLLLoss()

    try:
        train()
    except KeyboardInterrupt:
        print("\nEnding Program on Keyboard Interrupt")

    





    