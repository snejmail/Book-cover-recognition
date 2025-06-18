import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class MobileNetV2Custom(nn.Module):
    def __init__(self, num_classes=33):
        super(MobileNetV2Custom, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # Freeze base model
        for param in self.mobilenet_v2.parameters():
            param.requires_grad = False

        # Replace classifier
        in_features = self.mobilenet_v2.classifier[-1].in_features
        self.mobilenet_v2.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.mobilenet_v2(x)


if __name__ == "__main__":
    model = MobileNetV2Custom(num_classes=33)
    print(model)
