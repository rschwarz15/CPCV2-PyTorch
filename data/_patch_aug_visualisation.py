import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import PIL.ImageOps as PIO
import PIL.ImageEnhance as PIE
import random
import time

from image_preprocessing import Patchify, PatchifyAugment

def get_crop(image, crop_height, crop_width):

    #max_x = image.shape[1] - crop_width
    #max_y = image.shape[0] - crop_height
    #x = np.random.randint(0, max_x)
    #y = np.random.randint(0, max_y)
    x = 0
    y = 0

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop


def show_patches(x, fig_num):
    grid_size = x.shape[0]
    g = plt.figure(fig_num, figsize=(grid_size, grid_size))
    gs1 = gridspec.GridSpec(grid_size, grid_size)
    gs1.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.
    
    for i in range(grid_size * grid_size):
        n = i // grid_size
        m = i % grid_size

        ax1 = plt.subplot(gs1[i])
        plt.axis('off')

        if gray:
            patch = x[n][m][0]
            plt.imshow(patch, cmap="gray")
        else:
            patch = x[n][m].permute(1,2,0)
            plt.imshow(patch)

    g.show()
    g.savefig(f'./_hidden/images/LivieOutput{fig_num}.jpeg')


if __name__ == "__main__":
    IMG_SIZE = 256
    CROP_SIZE = 256
    GRID_SIZE = 7
    PATCH_SIZE = int(CROP_SIZE / (GRID_SIZE + 1) * 2)
    gray = False
    grayLabel = "_Gray" if gray else ""
    path = os.path.join("./_hidden/images/Livie.jpeg")

    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = get_crop(img, CROP_SIZE, CROP_SIZE)
    img = torch.Tensor(img).view(CROP_SIZE, CROP_SIZE, -1)
    img = img.permute(2,0,1)
    img = img/255

    # Show entire image
    f = plt.figure(1)
    plt.axis('off')
    if gray:
        plt.imshow(img.view(CROP_SIZE, CROP_SIZE), cmap="gray")
    else:
        plt.imshow(img.permute(1,2,0))
    f.savefig(f'./_hidden/images/LivieCrop{grayLabel}.jpeg')
    f.show()

    # Show patchified image
    p = Patchify(grid_size=GRID_SIZE)
    img_pathcified = p(img)
    show_patches(img_pathcified, 2)

    # Do data augmentation
    start = time.time()
    batches = 10
    iterations = 16 * batches
    
    pa = PatchifyAugment(gray=gray, grid_size=GRID_SIZE)
    img_augmented = pa(img)

    for i in range(iterations-1):
        img_augmented = pa(img)

    finish = time.time()
    show_patches(img_augmented, 3)

    print(f"Time taken: {finish - start}")
    print(str((finish - start) / iterations) + "s/iter")
    print(str((finish - start) * 6250 / batches / 60) + "min for entire stl10")
    input()
