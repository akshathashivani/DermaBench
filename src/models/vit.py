import torch
import torch.nn as nn
from torchvision.models import VisionTransformer


class DermaViT(nn.Module):

    def __init__(self, num_classes=7, pretrained=False):

        super().__init__()

        if pretrained:
            raise ValueError(
                "Pretrained weights are not configured for this "
                "custom lightweight ViT."
            )

        self.model = VisionTransformer(
            image_size=28,
            patch_size=7,
            num_layers=4,
            num_heads=3,
            hidden_dim=192,
            mlp_dim=384,
            dropout=0.1,
            attention_dropout=0.1,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    model = DermaViT()

    dummy_input = torch.randn(4, 3, 28, 28)

    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    print("\n========== VISION TRANSFORMER TEST ==========")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )

    print("\nVision Transformer initialized successfully!")
