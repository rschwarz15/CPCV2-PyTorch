# This has been moved to within the models forward call to improve efficiency

import torch
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0")

def info_nce(encodings, predictions, set_size):
    total_loss = torch.tensor([0.0]).to(device)
    batch_size = predictions.shape[0]
    pred_steps = predictions.shape[1]
    number_of_preds = 0
    number_correct = 0

    for img in range(batch_size):
        for step in range(pred_steps):
            for row in range(step+1,7):
                for col in range(7):
                    number_of_preds += 1

                    target_encoding = encodings[img][row][col][0]
                    predicted_encoding = predictions[img][step][row][col][0]

                    # Find dot products of predicted_encoding and encodings from other images
                    dots = torch.tensor([]).to(device)
                    predicted_dot = torch.dot(predicted_encoding, target_encoding).view(1)
                    dots = torch.cat([dots, predicted_dot], 0)

                    # Get set_size - 1 encodings from other images
                    n = 0
                    while n < set_size - 1:
                        other_img = np.random.randint(batch_size)

                        # Don't get encodings from the same image
                        # Might change this to allow different encodings from same image?
                        if other_img == img:
                            continue
                        
                        other_encoding = encodings[img][np.random.randint(7)][np.random.randint(7)][0]
                        other_dot = torch.dot(predicted_encoding, other_encoding).view(1)
                        dots = torch.cat([dots, other_dot], 0)

                        n += 1
                    
                    # Calculate loss
                    cross_entropy_loss = -1 * F.log_softmax(dots, dim=0)[0]
                    #mse_loss = F.mse_loss(predicted_encoding, target_encoding)
    
                    total_loss += cross_entropy_loss

                    # Calculate accuracy
                    if torch.argmax(dots) == 0:
                        number_correct += 1
                         
    #print(predicted_encoding)
    #print(target_encoding)
    #print(other_encoding)
    #print(dots)

    # return sum of loss per image
    loss_per_pred = total_loss / number_of_preds
    acc = number_correct / number_of_preds
    return loss_per_pred, acc