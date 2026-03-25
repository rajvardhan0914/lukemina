import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

from configs.config import *
from variliteformer.datasets.leukemia_dataset import get_dataloaders
from variliteformer.models.resnet_transformer import ResNetTransformer


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load dataset
    # -----------------------------
    _, val_loader = get_dataloaders(DATASET_PATH, IMG_SIZE, BATCH_SIZE)

    # -----------------------------
    # Load model
    # -----------------------------
    model = ResNetTransformer(MODEL_BACKBONE, NUM_CLASSES)

    model.load_state_dict(
        torch.load(f"{CHECKPOINT_DIR}/best_{MODEL_BACKBONE}.pth", map_location=device)
    )

    model.to(device)
    model.eval()

    preds = []
    targets = []
    probs = []

    # -----------------------------
    # Inference
    # -----------------------------
    with torch.no_grad():

        for imgs, labels in val_loader:

            imgs = imgs.to(device)

            out = model(imgs)

            probabilities = torch.softmax(out, dim=1)

            p = torch.argmax(out, 1)

            preds.extend(p.cpu().numpy())
            targets.extend(labels.numpy())
            probs.extend(probabilities[:,1].cpu().numpy())

    # -----------------------------
    # Classification report
    # -----------------------------
    print("\n========= Classification Report =========\n")
    print(classification_report(targets, preds))

    # -----------------------------
    # Create output directory
    # -----------------------------
    os.makedirs(f"{OUTPUT_DIR}/graphs", exist_ok=True)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(targets, preds)

    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i][j],
                     ha="center",
                     va="center",
                     color="black")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/graphs/confusion_matrix.png")
    plt.close()

    # -----------------------------
    # ROC Curve
    # -----------------------------
    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/graphs/roc_curve.png")
    plt.close()

    # -----------------------------
    # Precision-Recall Curve
    # -----------------------------
    precision, recall, _ = precision_recall_curve(targets, probs)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(f"{OUTPUT_DIR}/graphs/pr_curve.png")
    plt.close()

    print("\nSaved evaluation graphs in:")
    print(f"{OUTPUT_DIR}/graphs/")
    print("\nEvaluation completed successfully")


if __name__ == "__main__":
    main()