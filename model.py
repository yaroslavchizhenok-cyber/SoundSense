# First - learning how to train models)) Yes, CNNs included
import torch
from torch import nn


class SmallCNN(nn.Module):
    """M1: two convolution blocks -> global average pooling -> linear classifier."""

    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            # block 1: (B, 1, 64, 201) -> (B, 16, 32, 100)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # block 2: (B, 16, 32, 100) -> (B, 32, 16, 50)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)          # (B, 32, 16, 50) -> (B, 32, 1, 1)
        self.classifier = nn.Linear(32, n_classes)   # (B, 32) -> (B, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def conv_block(c_in, c_out):
    """Conv -> BatchNorm -> ReLU -> MaxPool. BatchNorm keeps each layer's outputs in a
    stable range, so a deeper network trains faster and more reliably."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),  # BatchNorm adds its own bias
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class DeepCNN(nn.Module):
    """M2 (and M3 with augmentation): four conv blocks with BatchNorm, then Dropout."""

    def __init__(self, n_classes=5, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),     # -> (B, 32, 32, 100)
            conv_block(32, 64),    # -> (B, 64, 16, 50)
            conv_block(64, 128),   # -> (B, 128, 8, 25)
            conv_block(128, 256),  # -> (B, 256, 4, 12)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)          # -> (B, 256, 1, 1)
        self.dropout = nn.Dropout(dropout)           # randomly drops 30% of features while training
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


MODELS = {"small": SmallCNN, "deep": DeepCNN}


def build_model(name, n_classes=5):
    """Create a model by name ("small" or "deep")."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose one of: {list(MODELS)}")
    return MODELS[name](n_classes=n_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick check: python model.py
    fake_batch = torch.randn(4, 1, 64, 201)
    for name in MODELS:
        m = build_model(name)
        print(f"{name:5s} | output {tuple(m(fake_batch).shape)} | {count_parameters(m):,} parameters")
