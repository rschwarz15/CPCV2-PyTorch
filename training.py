import torch
import random
from tqdm import tqdm
from models import CDC
from data_handlers import PetImagesCPCHandler

# 5 5 7 7 1 1280
def info_nce(encodings, predictions, set_size):
    loss = 0
    batch_size = predictions.shape[0]
    pred_steps = predictions.shape[1]

    for img in range(batch_size):
        for step in range(pred_steps):
            for row in range(step+1,7):
                for col in range(7):
                    num = 0
                    den = 0

                    correct_encoding = encodings[img][row][col][0]
                    predicted_encoding = predictions[img][step][row][col][0]

                    # Get set_size - 1 encodings from other images
                    n = 0
                    incorrect_encodings = torch.tensor([]).to(device)
                    while n < set_size:
                        other_img = random.randint(0, batch_size-1)

                        # Don't get encodings from the same image
                        # Might change this to allow different encodings from same image?
                        if other_img == img:
                            continue
                        
                        other_encoding = encodings[img][random.randint(0,6)][random.randint(0,6)][0]

                        print(torch.dot(other_encoding, correct_encoding))
                        print(torch.dot(correct_encoding, correct_encoding))

                        n += 1

                    return

if __name__ == "__main__":
    device = torch.device("cuda:0")
    batch_size = 5

    # Initialise data handler and network
    data = PetImagesCPCHandler(batch_size=batch_size)
    net = CDC(batch_size=batch_size, pred_steps=5).to(device)

    h = net.init_hidden().to(device)

    for batch in data:
        enc, pred = net(batch.to(device), h)
        info_nce(enc, pred, 5)

        break


