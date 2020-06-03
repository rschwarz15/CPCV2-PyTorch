from CPC.models.CPC import CPC
from CPC.models.MobileNetV2_Encoder import MobileNetV2_Encoder
from CPC.models.mobileNetV2 import MobileNetV2
from CPC.models.Resnet_Encoder import ResNet_Encoder
from CPC.models.Resnet import ResNet
from CPC.data.data_handlers import PetImagesHandler

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
    matches  = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs,y)]
    acc = matches.count(True)/len(matches)

    # Compute loss
    loss = loss_function(outputs, y) 

    if train:
        loss.backward()
        optimizer.step()

    return loss, acc 

# Train net
def train(data_handler, epochs, test_size):
    best_acc = 0
    for epoch in range(epochs):

        for batch_img, batch_lbl in tqdm(data_handler):
            loss, acc = fwd_pass(batch_img.to(device), batch_lbl.to(device), train=True)    

        val_loss, val_acc = test(size=test_size)

        if val_acc > best_acc:
            best_acc = val_acc

        print(f"Epoch: {epoch}\n"
               f"Train: {round(float(loss),4)}, {round(float(acc), 4)}\n"
                f"Test:  {round(float(val_loss),4)}, {round(float(val_acc),4)}")

        scheduler.step()
        
        # for param_group in optimizer.param_groups:
        #     print(param_group['lr'])
    
    print(f"Best Accuracy: {round(float(best_acc),4)}")


# Process test data to find validation loss/accuracy
def test(size):
    # Need to parse in small batches in order to not incur memory error
    test_img, test_lbl = data_handler.test_batch(batch_size=size)
    batch_size = 50
    total_val_acc = 0
    total_val_loss = 0

    # In batches process the test data
    for i in range(size // batch_size):
        batch_test_img = test_img[batch_size*i:batch_size*(i+1)].to(device)
        batch_test_lbl = test_lbl[batch_size*i:batch_size*(i+1)].to(device)
        val_acc, val_loss = fwd_pass(batch_test_img, batch_test_lbl)
        total_val_acc += val_acc * batch_size
        total_val_loss += val_loss * batch_size

        del batch_test_img, batch_test_lbl
    
    # Deal with leftover test samples
    leftover = size % batch_size

    if leftover != 0:
        batch_test_img = test_img[batch_size*(i+1):].to(device)
        batch_test_lbl = test_lbl[batch_size*(i+1):].to(device)
        val_acc, val_loss = fwd_pass(batch_test_img, batch_test_lbl)
        total_val_acc += val_acc * leftover
        total_val_loss += val_loss * leftover

        del batch_test_img, batch_test_lbl


    return total_val_acc / size, total_val_loss / size


if __name__ == "__main__":
    PATH = "./CPC/TrainedModels/"
    IMG_SIZE = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_selection = 0
    epochs = 20
    batch_size = 35  

    # Intitialise data handler
    data_handler = PetImagesHandler(
            batch_size=batch_size,
            train_proportion=0.057,
            test_proportion=0.05,
            include_labels=True
            )

    if train_selection == 0:
        print("Training CDC Classifier")

        # Load the CPC trained encoder (with classifier layer activated)
        net = ResNet_Encoder(resnet=34, num_classes=2, use_classifier=True).to(device)
        net.load_state_dict(torch.load(PATH + "trained_cpc_encoder_125.pt"))
        net = net.to(device)

        # Freeze encoder layers
        for name, param in net.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)

    elif train_selection == 1:
        print("Training MobileNet")

        # Load the network
        net = ResNet(resnet=34, num_classes=2).to(device)
        optimizer = optim.Adam(net.parameters(), lr=1e-4)
    
    # Train chosen network
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,5], gamma=0.1)
    loss_function = nn.MSELoss()

    train(data_handler=data_handler, epochs=epochs, test_size=1000)
    
    # Print final test accuracy
    val_loss, val_acc = test(size=1000)
    print(f"Final Accuracy: {round(val_acc*100, 2)}%")

# 1%  of ImageNet = 140  images per class (train_proportion = 0.012)
# 2%  of ImageNet = 280  images per class (train_proportion = 0.023)
# 5%  of ImageNet = 700  images per class (train_proportion = 0.057)
# 10% of ImageNet = 1400 images per class (train_proportion = 0.113)
# 20% of ImageNet = 2800 images per class (train_proportion = 0.225)
# 50% of ImageNet = 7000 images per class (train_proportion = 0.562)




    