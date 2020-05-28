# https://github.com/loeweX/Greedy_InfoMax/blob/master/GreedyInfoMax/vision/models/Resnet_Encoder.py

from CPC.models.model_utils import makeDeltaOrthogonal
#from model_utils import makeDeltaOrthogonal

import torch.nn as nn
import torch.nn.functional as F
import torch
import time

class PreActBlockNoBN(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlockNoBN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride
                )
            )

    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += shortcut
        return out


class PreActBottleneckNoBN(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneckNoBN, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        # self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride
                )
            )

    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(out))
        out = self.conv3(F.relu(out))
        out += shortcut
        return out


class ResNet_Encoder(nn.Module):
    def __init__(
        self,
        resnet=50,
        num_classes=2,
        weight_init=False,
        num_blocks=[3, 4, 6, 6, 6, 6, 6],
        filter=[64, 128, 256, 256, 256, 256, 256],
        encoder_num=0,
        patch_size=16,
        input_dims=1
    ):
        super(ResNet_Encoder, self).__init__()

        if resnet == 34:
            block = PreActBlockNoBN
        elif resnet == 50:
            block = PreActBottleneckNoBN
        else:
            raise Exception("Undefined resnet choice")

        self.encoder_num = encoder_num

        self.patch_size = patch_size
        self.filter = filter

        self.model = nn.Sequential()

        if encoder_num == 0:
            self.model.add_module(
                "Conv1",
                nn.Conv2d(
                    input_dims, self.filter[0], kernel_size=5, stride=1, padding=2
                ),
            )
            self.in_planes = self.filter[0]
            self.first_stride = 1
        elif encoder_num > 2:
            self.in_planes = self.filter[0] * block.expansion
            self.first_stride = 2
        else:
            self.in_planes = (self.filter[0] // 2) * block.expansion
            self.first_stride = 2

        for idx in range(len(num_blocks)):
            self.model.add_module(
                "layer {}".format((idx)),
                self._make_layer(
                    block, self.filter[idx], num_blocks[idx], stride=self.first_stride
                ),
            )
            self.first_stride = 2

        if weight_init:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                makeDeltaOrthogonal(
                    m.weight, nn.init.calculate_gain("relu")
                )
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                m.momentum = 0.3

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        #print(x.shape)
        #print(self.model)
        #start = time.time()
        out = self.model(x)
        #print(time.time() - start)
        return out


if __name__ == "__main__":
    device = torch.device("cuda:0")

    x = torch.randn(32*7*7, 1, 16, 16).to(device)
    net = ResNet_Encoder(34, 2).to(device)

    out = net(x)



