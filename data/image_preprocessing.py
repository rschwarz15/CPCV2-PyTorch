import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import PIL.ImageOps as PIO
import PIL.ImageEnhance as PIE
import random


class patchify(object):     
    """
    Converts a tensor (C x H x W) to a grid of tensors (grid_size x grid_size x C x patch_size x patch_size)
    """
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.patch_size = None

    def __call__(self, x):
        # Calculate the size of the patches
        if self.patch_size is None:         
            # Patchifying requires a square input image
            if x.shape[1] != x.shape[2]:
                raise Exception("Patchifying requires a square input image")
            
            patch_size = float(x.shape[2] / (self.grid_size + 1) * 2)

            # Not all grid sizes are compatable, ensure that patch_size is a whole number
            if patch_size.is_integer():
                self.patch_size = int(patch_size)
            else:
                raise Exception("The specified grid size did not fit the image")

        # Input x = (1, img_size, img_size)
        # Patchify to (grid_size, grid_size, 1, patch_size, patch_size)
        x = (
            x.unfold(1, self.patch_size, self.patch_size // 2)
            .unfold(2, self.patch_size, self.patch_size // 2)
            .permute(1, 2, 0, 3, 4)
            .contiguous()
        )

        return x

    def __repr__(self):
        return self.__class__.__name__ + '(grid_size={0})'.format(self.grid_size)


class patchify_augment(patchify):
    """ 
    Extends Patchify
    Converts a tensor (C x H x W) to a grid of tensors (grid_size x grid_size x C x patch_size x patch_size)
    Then for each patch applies 2 of the AutoAugment transformations
    Returns a tensor of (grid_size x grid_size x C x patch_size x patch_size)
    """

    def __init__(self, grid_size):
        super(patchify_augment, self).__init__(grid_size=grid_size)
            
        self.transformations = [
            self.ShearX, 
            self.ShearY, 
            self.TranslateX, 
            self.TranslateY, 
            self.Rotate, 
            PIO.autocontrast, 
            PIO.invert,
            PIO.equalize, 
            self.Solarize, 
            self.Posterize, 
            self.Contrast, 
            self.Brightness, 
            self.Sharpness, 
            self.Cutout
        ]

    def __call__(self, x):
        # Patchify using parent class
        x = super(patchify_augment, self).__call__(x) 
        
        self.patch_dim = (self.patch_size, self.patch_size)

        # For each path apply augmentation
        for patch_row in range(self.grid_size):
            for patch_col in range(self.grid_size):
                # Convert patch from tensor to PIL image
                patch_PIL = transforms.ToPILImage()(x[patch_row][patch_col])

                # Randomly choose two of the 16 transformations from AutoAugment
                # Minus PIE.Color, since we're dealing with greyscale
                for _ in range(2):
                    rand = random.randint(0, len(self.transformations))
                    if rand == len(self.transformations):
                        # If rand == 15 then perform SamplePairing
                        other_patch_row = random.randint(0, self.grid_size - 1)
                        other_patch_col = random.randint(0, self.grid_size - 1)
                        other_patch_PIL = transforms.ToPILImage()(x[other_patch_row][other_patch_col])
                        patch_PIL = self.SamplePairing(patch_PIL, other_patch_PIL)
                    else:
                        # Otherwise choose transform from transformations array
                        transform = self.transformations[rand]
                        patch_PIL = transform(patch_PIL)

                # Randomly apply elastic deformation and shearing with p = 0.2
                ### Currently not doing this

                # Convert PIL back to tensor
                x[patch_row][patch_col] = transforms.ToTensor()(patch_PIL)

        return x

    def __repr__(self):
        return self.__class__.__name__ + '(grid_size={0})'.format(self.grid_size)

    # The following transformation functions are largely based on:
    # https://github.com/tensorflow/models/blob/master/research/autoaugment/augmentation_transforms.py
    def ShearX(self, pil_img):
        level = random.random() * 0.6 - 0.3  # [-0.3,0.3] As in AutoAugment
        return pil_img.transform(self.patch_dim, Image.AFFINE, (1, level, 0, 0, 1, 0))


    def ShearY(self, pil_img):
        level = random.random() * 0.6 - 0.3  # [-0.3,0.3] As in AutoAugment
        return pil_img.transform(self.patch_dim, Image.AFFINE, (1, 0, 0, level, 1, 0))


    def TranslateX(self, pil_img):
        # Autoaugment does [-150,150] pixels which is eqiuvalent to ~1/2% of 331x331 image
        # 1/4 of patch - 1/2 seems excessive?
        pixels = random.randint(int(-self.patch_size/4), int(self.patch_size/4))
        return pil_img.transform(self.patch_dim, Image.AFFINE, (1, 0, pixels, 0, 1, 0))


    def TranslateY(self, pil_img):
        # Autoaugment does [-150,150] pixels which is eqiuvalent to ~1/2% of 331x331 image
        # 1/4 of patch - 1/2 seems excessive?
        pixels = random.randint(int(-self.patch_size/4), int(self.patch_size/4))
        return pil_img.transform(self.patch_dim, Image.AFFINE, (1, 0, 0, 0, 1, pixels))


    def Rotate(self, pil_img):
        degrees = random.random() * 60 - 30  # [-30, 30] as in AutoSegment
        return pil_img.rotate(degrees)


    def Posterize(self, pil_img):
        bits = random.randint(4, 8)  # [4,8] as in AutoSegment
        return PIO.posterize(pil_img, bits)


    def Solarize(self, pil_img):
        threshold = random.randint(0, 256)  # [4,8] as in AutoSegment
        return PIO.solarize(pil_img, threshold)


    def Contrast(self, pil_img):
        level = random.random() * 1.8 + 0.1  # [0.1,1.9] As in AutoAugment
        return PIE.Contrast(pil_img).enhance(level)


    def Brightness(self, pil_img):
        level = random.random() * 1.8 + 0.1  # [0.1,1.9] As in AutoAugment
        return PIE.Brightness(pil_img).enhance(level)


    def Sharpness(self, pil_img):
        level = random.random() * 1.8 + 0.1  # [0.1,1.9] As in AutoAugment
        return PIE.Sharpness(pil_img).enhance(level)


    def Cutout(self, pil_img):
        # Autoaugment does [0, 60] pixels which is eqiuvalent to ~1/5th of 331x331 image
        # 1/3 of patch otherwise they are too small
        size = random.randint(1, int(self.patch_size/3))

        # generate top_left crop coordinate - note (0,0) is upper left
        upper_coord_x = random.randint(0, self.patch_size - size)
        upper_coord_y = random.randint(0, self.patch_size - size)

        for col in range(upper_coord_x, upper_coord_x + size):
            for row in range(upper_coord_y, upper_coord_y + size):
                pil_img.putpixel((col, row), 128)

        return pil_img


    def SamplePairing(self, patch_PIL, other_patch_PIL):
        # For all pixels avg it with other patch
        for col in range(self.patch_size):
            for row in range(self.patch_size):
                avg_pixel = (patch_PIL.getpixel((col, row)) +
                            other_patch_PIL.getpixel((col, row))) / 2
                patch_PIL.putpixel((col, row), int(avg_pixel))

        return patch_PIL
