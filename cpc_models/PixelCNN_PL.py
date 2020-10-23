# From:
# https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/02134c6f4de6d1c5beb515044a62a7a5378d16eb/pl_bolts/models/vision/pixel_cnn.py
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/LICENSE
import torch.nn as nn
import torch.nn.functional as F


class PixelCNN(nn.Module):

    def __init__(self, in_channels, hidden_channels=256, num_blocks=5):
        """
        Implementation of `Pixel CNN <https://arxiv.org/abs/1606.05328>`_.
        Paper authors: Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves,
        Koray Kavukcuoglu
        Implemented by:
            - William Falcon
        Example::
            >>> from pl_bolts.models.vision import PixelCNN
            >>> import torch
            ...
            >>> model = PixelCNN(in_channels=3)
            >>> x = torch.rand(5, 3, 64, 64)
            >>> out = model(x)
            ...
            >>> out.shape
            torch.Size([5, 3, 64, 64])
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.blocks = nn.ModuleList([self.conv_block(in_channels) for _ in range(num_blocks)])

    def conv_block(self, in_channels):
        c1 = nn.Conv2d(in_channels=in_channels, out_channels=self.hidden_channels, kernel_size=(1, 1))
        act1 = nn.ReLU()
        c2 = nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=(1, 3))
        pad = nn.ConstantPad2d((0, 0, 1, 0, 0, 0, 0, 0), 1)
        c3 = nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels,
                       kernel_size=(2, 1), padding=(0, 1))
        act2 = nn.ReLU()
        c4 = nn.Conv2d(in_channels=self.hidden_channels, out_channels=in_channels, kernel_size=(1, 1))

        block = nn.Sequential(c1, act1, c2, pad, c3, act2, c4)
        return block

    def forward(self, z):
        c = z
        for conv_block in self.blocks:
            c = c + conv_block(c)

        c = F.relu(c)
        return c