from CPC.models.CPC import CPC
from CPC.models.MobileNetV2_Encoder import MobileNetV2_Encoder
from CPC.models.mobileNetV2 import MobileNetV2
from CPC.models.Resnet_Encoder import ResNet_Encoder
from CPC.models.Resnet import ResNet
from CPC.data.data_handler import get_stl10_dataloader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

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
def train(epochs):
    best_acc = 0
    for epoch in range(epochs):

        for batch_img, batch_lbl in tqdm(train_loader, dynamic_ncols=True):
            loss, acc = fwd_pass(batch_img.to(device), batch_lbl.to(device), train=True)    

        test_loss, test_acc = test()

        if test_acc > best_acc:
            best_acc = test_acc

        print(f"Epoch: {epoch}\n"
               f"Train: {round(float(loss),4)}, {round(float(acc), 4)}\n"
                f"Test:  {round(float(test_loss),4)}, {round(float(test_acc),4)}")

        scheduler.step()
        
        # for param_group in optimizer.param_groups:
        #     print(param_group['lr'])
    
    print(f"Best Accuracy: {round(float(best_acc),4)}")


# Process test data to find test loss/accuracy
def test():
    total_test_acc = 0
    total_test_loss = 0

    # Process all of the test data
    for batch_img, batch_lbl in tqdm(test_loader, dynamic_ncols=True):
        loss, acc = fwd_pass(batch_img.to(device), batch_lbl.to(device))  
        total_test_acc += acc
        total_test_loss += loss

    return total_test_loss / len(test_loader), total_test_acc / len(test_loader)

if __name__ == "__main__":
    PATH = "./CPC/TrainedModels/"
    IMG_SIZE = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_selection = 1
    epochs = 300
    batch_size = 100  

    # Intitialise data handler
    _, _, train_loader, _, test_loader, _ = get_stl10_dataloader(batch_size, labeled=True)

    if train_selection == 0:
        print("Training CPC Classifier")

        # Load the CPC trained encoder (with classifier layer activated)
        net = ResNet_Encoder(resnet=34, num_classes=2, use_classifier=True).to(device)
        net.load_state_dict(torch.load(PATH + f"trained_cpc_encoder_{50}.pt"))

        net.classifier = nn.Sequential(nn.Linear(256, 10)) # forgot to change in cpc training
        
        net = net.to(device)

        # Freeze encoder layers
        for name, param in net.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)

    elif train_selection == 1:
        print("Training Resnet")

        # Load the network
        net = ResNet(resnet=34, num_classes=30).to(device)
        optimizer = optim.Adam(net.parameters(), lr=1e-4)
    
    # Train chosen network
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,4], gamma=0.1)
    loss_function = nn.NLLLoss()

    train(epochs=epochs)
    
    # Print final test accuracy
    #val_loss, val_acc = test()
    #print(f"Final Accuracy: {round(val_acc*100, 2)}%")





    