import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import Config
from model import get_model


def test_dataset(model_path=Config.BEST_MODEL_PATH, test_dir=Config.TEST_DIR):
    """Test model on a dataset"""

    device = Config.DEVICE

    # Load model
    model = get_model('resnet50', Config.NUM_CLASSES, Config.PRETRAINED).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Transform
    test_transform = transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE,
                            shuffle=False, num_workers=4)

    # Test
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds,
                                        target_names=test_dataset.classes)

    print("=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(class_report)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('test_confusion_matrix.png')
    print("\n📊 Confusion matrix saved!")

    return accuracy, conf_matrix, class_report


if __name__ == "__main__":
    test_dataset()
