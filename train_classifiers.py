import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm
from models import CDC_encoder, MobileNetV2
from data_handlers import PetImagesHandler

PATH = "./TrainedModels/"
IMG_SIZE = 256
device = torch.device("cuda:0")

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

# Test the network on a random batch of test data
def test(size=32):
    start = np.random.randint(len(test_X) - size)
    X, y = test_X[start:start+size], test_y[start:start+size]

    #X, y = test_X[:size], test_y[:size]
    val_acc, val_loss = fwd_pass(X.view(-1, 1, IMG_SIZE, IMG_SIZE).to(device), y.to(device))
    return val_acc, val_loss

# Train the network
def train(EPOCHS, log=True):
    BATCH_SIZE = 32

    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,IMG_SIZE,IMG_SIZE).to(device)
            batch_y = train_y[i:i+BATCH_SIZE].to(device)

            acc, loss = fwd_pass(batch_X, batch_y, train=True)

        val_acc, val_loss = test(size=BATCH_SIZE)
        print(f"{epoch},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)}")


if __name__ == "__main__":
    # Load the network
    net = CDC_encoder(classifier=True).to(device)
    net.load_state_dict(torch.load(PATH + "trained_cpc_encoder", map_location=device))

    # Freeze encoder layers
    

    data_handler = PetImagesHandler()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    train_X, train_y, test_X, test_y = data_handler.get_labelled_data(0.1)


    #train(10)
    



    