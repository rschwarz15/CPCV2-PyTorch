# From https://github.com/loeweX/Greedy_InfoMax
# @article{lowe2019putting,
#   title={Putting An End to End-to-End: Gradient-Isolated Learning of Representations},
#   author={L{\"o}we, Sindy and O'Connor, Peter and Veeling, Bastiaan S},
#   journal={arXiv preprint arXiv:1905.11786},
#   year={2019}
# }
import torch
import torch.nn as nn

import pixelCNN

class PixelCNN_Autoregressor(torch.nn.Module):
    def __init__(self, in_channels, weight_init, pixelcnn_layers=4, **kwargs):
        super().__init__()

        layer_objs = [
            pixelCNN.PixelCNNGatedLayer.primary(
                in_channels, in_channels, 3, mask_mode="only_vert", **kwargs
            )
        ]
        layer_objs = layer_objs + [
            pixelCNN.PixelCNNGatedLayer.secondary(
                in_channels, in_channels, 3, mask_mode="only_vert", **kwargs
            )
            for _ in range(1, pixelcnn_layers)
        ]

        self.stack = pixelCNN.PixelCNNGatedStack(*layer_objs)
        self.stack_out = nn.Conv2d(in_channels, in_channels, 1)

        if weight_init:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                if m is self.stack_out:
                    # nn.init.kaiming_normal_(
                    #     m.weight, mode="fan_in", nonlinearity="relu"
                    # )
                    makeDeltaOrthogonal(
                        m.weight, nn.init.calculate_gain("relu")
                    )
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="tanh"
                    )
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                m.momentum = 0.3

    def forward(self, input):
        _, c_out, _ = self.stack(input, input)  # Bc, C, H, W
        c_out = self.stack_out(c_out)

        assert c_out.shape[1] == input.shape[1]

        return c_out

def makeDeltaOrthogonal(weights, gain):
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    q = genOrthgonal(dim)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    with torch.no_grad():
        weights[:, :, mid1, mid2] = q[: weights.size(0), : weights.size(1)]
        weights.mul_(gain)

def genOrthgonal(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q