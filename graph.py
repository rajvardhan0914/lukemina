import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report
)

from configs.config import *
from variliteformer.datasets.leukemia_dataset import get_dataloaders
from variliteformer.models.resnet_transformer import ResNetTransformer


os.makedirs("outputs/graphs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, val_loader = get_dataloaders(DATASET_PATH, IMG_SIZE, BATCH_SIZE)

model = ResNetTransformer(MODEL_BACKBONE, NUM_CLASSES)

model.load_state_dict(
    torch.load(f"{CHECKPOINT_DIR}/best_{MODEL_BACKBONE}.pth", map_location=device)
)

model.to(device)
model.eval()

y_true = []
y_pred = []
y_prob = []

with torch.no_grad():

    for imgs, labels in val_loader:

        imgs = imgs.to(device)

        outputs = model(imgs)

        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs[:,1].cpu().numpy())


print(classification_report(y_true, y_pred))


# -----------------------
# Confusion Matrix
# -----------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6), dpi=300)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["ALL","HEM"],
    yticklabels=["ALL","HEM"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("outputs/graphs/confusion_matrix.png")

plt.close()


# -----------------------
# ROC Curve
# -----------------------

fpr, tpr, _ = roc_curve(y_true, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6), dpi=300)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)

plt.plot([0,1],[0,1],'--',color='gray')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/graphs/roc_curve.png")

plt.close()


# -----------------------
# Precision Recall Curve
# -----------------------

precision, recall, _ = precision_recall_curve(y_true, y_prob)

plt.figure(figsize=(8,6), dpi=300)

plt.plot(recall, precision, linewidth=2)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/graphs/pr_curve.png")

plt.close()


# -----------------------
# Probability Histogram
# -----------------------

plt.figure(figsize=(8,6), dpi=300)

plt.hist(y_prob, bins=25, color="steelblue")

plt.xlabel("Prediction Probability")
plt.ylabel("Frequency")

plt.title("Prediction Probability Distribution")

plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/graphs/prob_distribution.png")

plt.close()

print("Graphs saved in outputs/graphs/")