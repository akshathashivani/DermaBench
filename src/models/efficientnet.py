import torch
import torch.nn as nn
from torchvision import models


class DermaEfficientNetB0(nn.Module):

    def __init__(self, num_classes=7, pretrained=False):

        super().__init__()

        self.model = models.efficientnet_b0(
            weights="DEFAULT" if pretrained else None
        )

        # Adapt classifier for seven skin lesion classes
        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    model = DermaEfficientNetB0()

    dummy_input = torch.randn(4, 3, 28, 28)

    output = model(dummy_input)

    print("\n========== EFFICIENTNET-B0 TEST ==========")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )

    print("\nEfficientNet-B0 initialized successfully!")
