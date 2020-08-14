# Based On:
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
# For CPC Encoder the fourth residual layer is removed

'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def norm2d(planes, norm):
    if norm == "none":
        return nn.Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(planes)
    elif norm == "layer":
        return nn.GroupNorm(1, planes)
    elif norm == "instance":
        return nn.GroupNorm(planes, planes)
    else:
        raise Exception("Undefined norm choice")


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, args, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()

        # If there isn't normalisation then the conv layers need biasing
        bias = True if (args.norm == "none") else False

        self.norm1 = norm2d(in_planes, args.norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm2 = norm2d(planes, args.norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x):
        out = F.relu(self.norm1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.norm2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, args, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()

        # If there isn't normalisation then the conv layers need biasing
        bias = True if (args.norm == "none") else False

        self.norm1 = norm2d(in_planes, args.norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)
        self.norm2 = norm2d(planes, args.norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm3 = norm2d(planes, args.norm)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=bias)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x):
        out = F.relu(self.norm1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.norm2(out)))
        out = self.conv3(F.relu(self.norm3(out)))
        out += shortcut
        return out


class PreActResNet_Encoder(nn.Module):
    def __init__(self, args, use_classifier, block, num_blocks, num_channels=[64, 128, 256], input_channels=1):
        super(PreActResNet_Encoder, self).__init__()
        self.args = args
        self.in_planes = 64
        self.dataset = args.dataset
        self.use_classifier = use_classifier

        # If there isn't normalisation then the conv layers need biasing
        bias = True if (args.norm == "none") else False

        if self.dataset == "stl10":
            # From https://github.com/loeweX/Greedy_InfoMax/blob/master/GreedyInfoMax/vision/models/Resnet_Encoder.py
            # Testing showed 5x5 kernal to have better classification performance - need to retest for more epochs
            self.conv1 = nn.Conv2d(input_channels, self.in_planes, kernel_size=5, stride=1, padding=2, bias=bias)
        elif self.dataset[:5] == "cifar":
            # Testing showed 5x5 kernal to have better classification performance - need to retest for more epochs
            self.conv1 = nn.Conv2d(input_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=bias)
        elif self.dataset == "imagenet":
            # Standard ResNet Structure for ImageNet
            self.conv1 = nn.Conv2d(input_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=bias)
            self.norm1 = norm2d(self.in_planes, args.norm)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(args, block, num_channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(args, block, num_channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(args, block, num_channels[2], num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256*block.expansion, args.num_classes)

    def _make_layer(self, args, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(args, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        grid_size = self.args.grid_size

        # Run the model
        z = self.conv1(x)
        if self.dataset == "imagenet":
            z = self.norm1(z)
            z = self.relu(z)
            z = self.maxpool(z)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.avgpool(z)
        z = z.reshape(-1, grid_size, grid_size, z.shape[1]) # (batch_size, grid_size, grid_size, pred_size)

        # Use classifier if specified
        if self.use_classifier:
            # Reshape z so that each image is seperate
            z = z.view(z.shape[0], grid_size * grid_size, z.shape[3])

            # mean for each image, (batch_size, pred_size)
            z = torch.mean(z, dim=1)
            z = self.classifier(z)
            z = F.log_softmax(z, dim=1)

        return z


def PreActResNet18_Encoder(args, use_classifier):
    return PreActResNet_Encoder(args, use_classifier, PreActBlock, [2, 2, 2])


def PreActResNet34_Encoder(args, use_classifier):
    return PreActResNet_Encoder(args, use_classifier, PreActBlock, [3, 4, 6])


def PreActResNet50_Encoder(args, use_classifier):
    return PreActResNet_Encoder(args, use_classifier, PreActBottleneck, [3, 4, 6])


def PreActResNet101_Encoder(args, use_classifier):
    return PreActResNet_Encoder(args, use_classifier, PreActBottleneck, [3, 4, 23])


def PreActResNet152_Encoder(args, use_classifier):
    return PreActResNet_Encoder(args, use_classifier, PreActBottleneck, [3, 8, 36])


def PreActResNetN_Encoder(args, use_classifier):
    if args.encoder == "resnet18":
        return PreActResNet18_Encoder(args, use_classifier)
    elif args.encoder == "resnet34":
        return PreActResNet34_Encoder(args, use_classifier)
    elif args.encoder == "resnet50":
        return PreActResNet50_Encoder(args, use_classifier)
    elif args.encoder == "resnet101":
        return PreActResNet101_Encoder(args, use_classifier)
    elif args.encoder == "resnet152":
        return PreActResNet152_Encoder(args, use_classifier)
