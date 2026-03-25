import torch
import torch.nn as nn
from torchvision import models


class ResNetTransformer(nn.Module):


    def __init__(self, backbone="resnet18", num_classes=2):
        super().__init__()

        # ── 1. CNN BACKBONE ──────────────────────────────────────────────────
        if backbone == "resnet18":
            cnn = models.resnet18(weights="DEFAULT")
            self.feat_dim = 512
        elif backbone == "resnet50":
            cnn = models.resnet50(weights="DEFAULT")
            self.feat_dim = 2048
        else:
            raise ValueError("backbone must be 'resnet18' or 'resnet50'")

        # Remove original FC classifier — use GAP output as feature vector
        cnn.fc = nn.Identity()
        self.cnn = cnn

        # Freeze first 20 parameter tensors for efficient transfer learning
        for param in list(self.cnn.parameters())[:20]:
            param.requires_grad = False

        # ── 2. VARILITEFORMER ATTENTION BLOCK ────────────────────────────────
        # Single lightweight encoder layer — keeps computation minimal
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feat_dim,
            nhead=4,                        # 4 heads: lightweight
            dim_feedforward=self.feat_dim,  # no expansion — parameter-efficient
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1   # single layer — computationally efficient
        )

        # ── 3. CLASSIFICATION HEAD ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        # Stage 1: Region-adaptive CNN feature extraction
        x = self.cnn(x)                 # (Batch, D)

        # Stage 2: Tokenisation — single global token
        x = x.unsqueeze(1)             # (Batch, 1, D)

        # Stage 3: VariLiteFormer attention — global contextual reasoning
        x = self.transformer(x)        # (Batch, 1, D)

        # Stage 4: Squeeze back
        x = x.squeeze(1)              # (Batch, D)

        # Stage 5: Classification
        x = self.classifier(x)        # (Batch, num_classes)

        return x
