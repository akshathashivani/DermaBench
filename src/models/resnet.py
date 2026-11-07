import torch
import torch.nn as nn
from torchvision import models


class DermaResNet18(nn.Module):

    def __init__(self, num_classes=7, pretrained=False):

        super().__init__()

        self.model = models.resnet18(
            weights="DEFAULT" if pretrained else None
        )

        # Adapt first convolution for 28x28 images
        self.model.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.model.maxpool = nn.Identity()

        # Seven-class classification
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    model = DermaResNet18()

    dummy_input = torch.randn(4, 3, 28, 28)

    output = model(dummy_input)

    print("\n========== RESNET18 TEST ==========")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("\nResNet18 initialized successfully!")
