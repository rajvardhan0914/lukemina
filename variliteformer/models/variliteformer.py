import torch
import torch.nn as nn


class LightweightCNN(nn.Module):
    """
    Small CNN backbone for local feature extraction
    """

    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.features(x)


class TransformerEncoder(nn.Module):
    """
    Lightweight Transformer Encoder
    """

    def __init__(self, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        return self.transformer(x)


class VariLiteFormer(nn.Module):
    """
    Hybrid CNN + Transformer Model
    """

    def __init__(self, num_classes=2, embed_dim=128):
        super().__init__()

        self.cnn = LightweightCNN(embed_dim=embed_dim)

        self.transformer = TransformerEncoder(embed_dim=embed_dim)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)  # [B, C, H, W]

        B, C, H, W = x.shape

        # Flatten spatial dims into sequence
        x = x.view(B, C, H * W)
        x = x.permute(0, 2, 1)  # [B, sequence_len, embed_dim]

        # Transformer
        x = self.transformer(x)

        # Global pooling
        x = x.permute(0, 2, 1)
        x = self.global_pool(x).squeeze(-1)

        # Classification
        x = self.classifier(x)

        return x
