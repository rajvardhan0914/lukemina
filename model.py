import torch
import torch.nn as nn
from torchvision import models
from config import Config


class LeukemiaClassifier(nn.Module):
    """ResNet50 based classifier for leukemia detection"""

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        super(LeukemiaClassifier, self).__init__()

        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)

        # Freeze early layers for transfer learning
        for param in list(self.resnet.parameters())[:-8]:
            param.requires_grad = False

        # Modify the final layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


class SimpleCNN(nn.Module):
    """Simple CNN for leukemia classification"""

    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model(model_name='resnet50', num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
    """Factory function to get model"""
    if model_name == 'resnet50':
        return LeukemiaClassifier(num_classes, pretrained)
    elif model_name == 'simple_cnn':
        return SimpleCNN(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
