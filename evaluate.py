import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import Config
from model import get_model


def evaluate_model(model_path=Config.BEST_MODEL_PATH, test_dir=Config.VAL_DIR):
    """Comprehensive model evaluation"""

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
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    roc_auc = roc_auc_score(all_labels, all_probs)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds,
                                        target_names=test_dataset.classes)

    # Print results
    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nClassification Report:")
    print(class_report)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Plot Results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Confusion Matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes,
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    axes[0, 1].plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid()

    # Metrics Bar Plot
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
    axes[1, 0].bar(metrics.keys(), metrics.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Performance Metrics')
    axes[1, 0].set_ylim([0, 1])
    for i, (k, v) in enumerate(metrics.items()):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')

    # Prediction Distribution
    axes[1, 1].hist(all_probs[all_labels == 0], bins=20, alpha=0.7, label='Class 0')
    axes[1, 1].hist(all_probs[all_labels == 1], bins=20, alpha=0.7, label='Class 1')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150)
    print("\n📊 Evaluation plots saved!")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


if __name__ == "__main__":
    evaluate_model()
