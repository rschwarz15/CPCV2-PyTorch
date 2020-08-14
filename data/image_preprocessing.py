import torch
import kornia
import kornia.augmentation as K
import numpy as np

def create_auto_augment(device):
    return None


def preprocess(x, train, args):
    # Input x = (batch_size, 1, img_size, img_size)
    
    img_size = x.shape[3]
    patch_size = float(img_size / (args.grid_size + 1) * 2)

    # Not all grid sizes are compatable, ensure that patch_size is a whole number
    if patch_size.is_integer():
        patch_size = int(patch_size)
    else:
        raise Exception("The specified grid size did not fit the image")
    
    # Patchify to (batch_size, grid_size, grid_size, 1, patch_size, patch_size)
    x = (
        x.unfold(2, patch_size, patch_size // 2)
        .unfold(3, patch_size, patch_size // 2)
        .permute(0, 2, 3, 1, 4, 5)
        .contiguous()
    )

    # Flatten to (batch_size * grid_size * grid_size, 1, patch_size, patch_size)
    # This is the shape that the CPC encoder is expecting
    x = x.view(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5])

    # Apply patch based data augmentation as per CPC V2 (ignoring color based augs)
    # Only apply during training, not during testing
    if args.patch_based_aug and train:
        for i in range(x.shape[0]):
            # 1) Randomly Choose two of the 16 transformations that are use in AutoAugment
            rand = np.random.randint(0, len(auto_augment_transformations), 2)
            transform1 = auto_augment_transformations[rand[0]]
            transform2 = auto_augment_transformations[rand[1]]

            x[i] =  transform2(transform1(x[i]))

    return x
