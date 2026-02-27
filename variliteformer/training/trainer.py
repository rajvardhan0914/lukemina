import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        self.best_val_acc = 0.0

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        loop = tqdm(self.train_loader, desc="Training")

        for images, labels in loop:
            images = images.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        accuracy = 100 * correct / total
        return total_loss / len(self.train_loader), accuracy

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.config.DEVICE)
                labels = labels.to(self.config.DEVICE)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        return val_loss / len(self.val_loader), accuracy