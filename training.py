import torch
from models import CDC
from data_handlers import PetImagesCPCHandler

if __name__ == "__main__":
    batch_size = 5
    data = PetImagesCPCHandler(batch_size=batch_size)
    net = CDC(batch_size=batch_size, pred_steps=5)
    h = net.init_hidden()

    for batch in data:
        encodings, predictions = net(batch, h)
        print(encodings.shape)
        print(predictions.shape)
        break



