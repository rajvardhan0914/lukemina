import torch
from sklearn.metrics import accuracy_score, f1_score


def calculate_metrics(outputs, labels):
    _, preds = torch.max(outputs, 1)

    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')

    return acc, f1