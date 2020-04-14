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

        val_acc, val_loss = test(size=32)

        print(f"Epoch: {epoch}\n"
                f"Train: {round(float(acc),2)}, {round(float(loss), 4)}\n"
                f"Test:  {round(float(val_acc),2)}, {round(float(val_loss),4)}")


def test(size):
        test_img, test_lbl = data_handler.test_batch(size=size)
        val_acc, val_loss = fwd_pass(test_img.to(device,), test_lbl.to(device))

        return val_acc, val_loss


if __name__ == "__main__":
    PATH = "./TrainedModels/"
    IMG_SIZE = 256
    device = torch.device("cuda:0")

    train_selection = 1
    epochs = 20

    if train_selection == 0:
        print("Training CDC Encoder")
        batch_size = 16  

        # Load the network
        net = CPC_encoder(batch_size=batch_size, classifier=True).to(device)
        net.load_state_dict(torch.load(PATH + "trained_cpc_encoder", map_location=device))

        # Freeze encoder layers
        for name, param in net.named_parameters():
            if "enc.classifier" in name:
                param.requires_grad = False
        
        # Intitialise data handler, optimizer and loss_function
        data_handler = PetImagesCPCHandler(batch_size=batch_size,
                                            include_labels=True,
                                            train_proportion=0.1,
                                            test_proportion=0.05)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)
        loss_function = nn.MSELoss()

        # Train CPC encoder for classification
        train(data_handler=data_handler, epochs=epochs, batch_size=batch_size)
    elif train_selection == 1:
        print("Training MobileNet")
        batch_size = 32  

        # Load the network
        net = MobileNetV2().to(device)

        # Intitialise data handler, optimizer and loss_function
        data_handler = PetImagesNormalHandler(batch_size=batch_size, 
                                                train_proportion=0.95, 
                                                test_proportion=0.05)
        optimizer = optim.Adam(net.parameters(), lr=0.1)
        loss_function = nn.MSELoss()

        # Train MobileNet
        train(data_handler=data_handler, epochs=epochs, batch_size=batch_size)



    



    