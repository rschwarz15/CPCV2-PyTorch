import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm
from models import CPC_encoder, MobileNetV2
from data_handlers import PetImagesCPCHandler, PetImagesNormalHandler

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
    for epoch in range(epochs):

        for batch_img, batch_lbl in tqdm(data_handler):
            loss, acc = fwd_pass(batch_img.to(device), batch_lbl.to(device), train=True)    

        val_loss, val_acc = test(size=test_size)

        print(f"Epoch: {epoch}\n"
               f"Train: {round(float(loss),4)}, {round(float(acc), 4)}\n"
                f"Test:  {round(float(val_loss),4)}, {round(float(val_acc),4)}")


# Process test data to find validation loss/accuracy
def test(size):
    # Need to parse in small batches in order to not incur memory error
    test_img, test_lbl = data_handler.test_batch(size=size)
    batch_size = 20
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
    PATH = "./TrainedModels/"
    IMG_SIZE = 256
    device = torch.device("cuda:0")

    train_selection = 0
    epochs = 50
    batch_size = 35  

    if train_selection == 0:
        print("Training CDC Encoder")

        # Load the network
        net = CPC_encoder(classifier=True).to(device)
        net.load_state_dict(torch.load(PATH + "trained_cpc_encoder", map_location=device))

        # Freeze encoder layers
        for name, param in net.named_parameters():
            if "enc.classifier" not in name:
                param.requires_grad = False

        # for name, param in net.named_parameters():
        #     print(name, param.requires_grad)
        
        # Intitialise data handler, optimizer and loss_function
        data_handler = PetImagesCPCHandler(batch_size=batch_size,
                                            include_labels=True,
                                            train_proportion=0.056,
                                            test_proportion=0.05)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=5e-2)

    elif train_selection == 1:
        print("Training MobileNet")

        # Load the network
        net = MobileNetV2().to(device)

        # Intitialise data handler, optimizer and loss_function
        data_handler = PetImagesNormalHandler(batch_size=batch_size, 
                                                train_proportion=0.028, 
                                                test_proportion=0.05)
        optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # Train chosen network
    loss_function = nn.MSELoss()
    train(data_handler=data_handler, epochs=epochs, test_size=100)

    # Print final test accuracy
    val_loss, val_acc = test(size=1000)
    print(f"Final Accuracy: {round(val_acc*100, 2)}%")

# 1%  of ImageNet = 140  images per class (train_proportion = 0.0056)
# 5%  of ImageNet = 700  images per class (train_proportion = 0.028)
# 10% of ImageNet = 1400 images per class (train_proportion = 0.056)
# 50% of ImageNet = 7000 images per class (train_proportion = 0.28)
# 100% of ImageNet = 14000 images per class (train_proportion = 0.56)




    