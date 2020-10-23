# Based On:
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py

'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, args, block, num_blocks, num_channels):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.dataset = args.dataset

        # Grayscale or Coloured
        if args.gray:
            input_channels = 1
        else:
            input_channels = 3

        # Stem Net
        if self.dataset == "stl10":
            self.conv1 = nn.Conv2d(input_channels, self.in_planes, kernel_size=5, stride=1, padding=2, bias=False)
        elif self.dataset[:5] == "cifar":
            self.conv1 = nn.Conv2d(input_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        elif self.dataset == "imagenet": 
            # Standard ResNet Structure for ImageNet
            self.conv1 = nn.Conv2d(input_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.in_planes, args.norm)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Conv Layers
        layers_array = []
        stride = 1
        for i in range(len(num_blocks)):
            layers_array.append(self._make_layer(block, num_channels[i], num_blocks[i], stride=stride))
            if stride == 1:
                stride = 2
        self.layers = nn.Sequential(*layers_array)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(num_channels[-1]*block.expansion, args.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        if self.dataset == "imagenet":
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)

        out = self.layers(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

# Normal ResNet
def PreActResNet18(args):
    return PreActResNet(args, PreActBlock, [2,2,2,2], [64, 128, 256, 512])

def PreActResNet34(args):
    return PreActResNet(args, PreActBlock, [3,4,6,3], [64, 128, 256, 512])

def PreActResNet50(args):
    return PreActResNet(args, PreActBottleneck, [3,4,6,3], [64, 128, 256, 512])

def PreActResNet101(args):
    return PreActResNet(args, PreActBottleneck, [3,4,23,3], [64, 128, 256, 512])

def PreActResNet152(args):
    return PreActResNet(args, PreActBottleneck, [3,8,36,3], [64, 128, 256, 512])

# Modified ResNet with the fourth layer removed
def PreActResNet14(args):
    return PreActResNet(args, PreActBlock, [2, 2, 2], [64, 128, 256])

def PreActResNet28(args):
    return PreActResNet(args, PreActBlock, [3, 4, 6], [64, 128, 256])

def PreActResNet41(args):
    return PreActResNet(args, PreActBottleneck, [3, 4, 6], [64, 128, 256])

def PreActResNet92(args):
    return PreActResNet(args, PreActBottleneck, [3, 4, 23], [64, 128, 256])

def PreActResNet143(args):
    return PreActResNet(args, PreActBottleneck, [3, 8, 36], [64, 128, 256])

def PreActResNetN(args):
    if args.encoder == "resnet18":
        return PreActResNet18(args)
    elif args.encoder == "resnet34":
        return PreActResNet34(args)
    elif args.encoder == "resnet50":
        return PreActResNet50(args)
    elif args.encoder == "resnet101":
        return PreActResNet101(args)
    elif args.encoder == "resnet152":
        return PreActResNet152(args)

    elif args.encoder == "resnet14":
        return PreActResNet14(args)
    elif args.encoder == "resnet28":
        return PreActResNet28(args)
    elif args.encoder == "resnet41":
        return PreActResNet41(args)
    elif args.encoder == "resnet92":
        return PreActResNet92(args)
    elif args.encoder == "resnet143":
        return PreActResNet143(args)


