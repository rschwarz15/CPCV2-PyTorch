# Modified From:
# https://github.com/loeweX/Greedy_InfoMax/blob/master/GreedyInfoMax/vision/models/Resnet_Encoder.py

from models.model_utils import makeDeltaOrthogonal
  
import torch.nn as nn
import torch.nn.functional as F
import torch

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
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
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

class ResNet(nn.Module):
    def __init__(
        self,
        args,
        weight_init=False,
        input_dims=1,
        num_blocks=[3, 4, 6, 6, 6, 6, 6],
        filter=[64, 128, 256, 256, 256, 256, 256],
    ):
        super(ResNet, self).__init__()
        
        if args.encoder == "resnet34":
            self.block = PreActBlockNoBN
        elif args.encoder == "resnet50":
            self.block = PreActBottleneckNoBN
        else:
            raise Exception("Undefined resnet choice")

        self.filter = filter

        # Resnet Model
        self.model = nn.Sequential()

        self.model.add_module(
            "Conv1",
            nn.Conv2d(
                input_dims, self.filter[0], kernel_size=5, stride=1, padding=2
            ),
        )
        self.in_planes = self.filter[0]
        self.first_stride = 1

        for idx in range(len(num_blocks)):
            self.model.add_module(
                "layer {}".format((idx)),
                self._make_layer(
                    self.block, self.filter[idx], num_blocks[idx], stride=self.first_stride
                ),
            )
            self.first_stride = 2

        # Additional Classifier 
        self.classifier = nn.Sequential(
            #nn.Dropout(0.2),
            nn.Linear(self.filter[-1] * self.block.expansion, args.num_classes)
        )

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


    def forward(self, x, patchify=False):
        z = self.model(x)
        z = F.adaptive_avg_pool2d(z, 1).view(z.shape[0],z.shape[1])
        z = self.classifier(z)
        z = F.log_softmax(z, dim=1)
        
        return z
