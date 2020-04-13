import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm
from models import CPC_encoder, MobileNetV2
from data_handlers import PetImagesCPCHandler, PetImagesNormalHandler

# Run batch test, return accuracy and loss
def fwd_pass(X, y, train=False):
    if train:
        net.train()
        net.zero_grad()

    if not train:
        net.eval()

    # Run the network on a batch
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
def train(data_handler, epochs, batch_size):
    for epoch in range(epochs):

        for batch_img, batch_lbl in tqdm(data_handler):
            acc, loss = fwd_pass(batch_img.to(device), batch_lbl.to(device), train=True)    

        print(f"{epoch},{round(float(acc),2)},{round(float(loss), 4)}")

        # print(f"{epoch},\
        #         {round(float(acc),2)},{round(float(loss), 4)},\
        #         {round(float(val_acc),2)},{round(float(val_loss),4)}")

if __name__ == "__main__":
    PATH = "./TrainedModels/"
    IMG_SIZE = 256
    device = torch.device("cuda:0")

    train_selection = 1
    epochs = 3

    if train_selection == 0:
        print("Training CDC Encoder")
        batch_size = 16  

        # Load the network
        net = CPC_encoder(batch_size=batch_size, classifier=True).to(device)
        net.load_state_dict(torch.load(PATH + "trained_cpc_encoder", map_location=device))

        # Freeze encoder layers
        # ...

        # Intitialise data handler, optimizer and loss_function
        data_handler = PetImagesCPCHandler(batch_size=batch_size,
                                            include_labels=True,
                                            train_proportion=0.1,
                                            test_proportion=0.05)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        loss_function = nn.MSELoss()

        # Train CPC encoder for classification
        train(10)
    elif train_selection == 1:
        print("Training MobileNet")
        batch_size = 64  

        # Load the network
        net = MobileNetV2().to(device)

        # Intitialise data handler, optimizer and loss_function
        data_handler = PetImagesNormalHandler(batch_size=batch_size, 
                                                train_proportion=0.1, 
                                                test_proportion=0.05)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        loss_function = nn.MSELoss()

        # Train MobileNet
        train(data_handler=data_handler, epochs=epochs, batch_size=batch_size)

# TO DO:
# Complete training of cdc encoder + classifer
# include validation accuracy and loss


    



    