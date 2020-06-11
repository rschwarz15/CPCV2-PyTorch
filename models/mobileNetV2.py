import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# torch.models.mobilenet_v2 with 1 input channel instead of 3
# also softmax applied for class classification use
class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Get Mobile Net
        self.model = models.mobilenet_v2(num_classes=num_classes)

        # Modify for one channel input
        # Originally: nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):   
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        
        return x
