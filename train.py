import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cpu")

# =====================================
# 1. STRONG MEDICAL DATA AUGMENTATION
# =====================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset_path = "dataset/C-NMC 2019 (PKG)/C-NMC_training_data/fold_1"

train_dataset = datasets.ImageFolder(dataset_path, transform=train_transform)
val_dataset = datasets.ImageFolder(dataset_path, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("Total Samples:", len(train_dataset))

# =====================================
# 2. MODEL (ResNet18 + Transformer)
# =====================================
class LeukemiaModel(nn.Module):
    def __init__(self):
        super(LeukemiaModel, self).__init__()

        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=num_features,
                nhead=4,
                batch_first=True
            ),
            num_layers=1
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(num_features, 2)

    def forward(self, x):
        features = self.cnn(x)
        features = features.unsqueeze(1)
        features = self.transformer(features)
        features = features.squeeze(1)
        features = self.dropout(features)
        out = self.classifier(features)
        return out

model = LeukemiaModel().to(device)

# =====================================
# 3. FOCAL LOSS (BETTER THAN CE)
# =====================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()

# Class weights
targets = train_dataset.targets
class_counts = np.bincount(targets)
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum()

criterion = FocalLoss(gamma=2, alpha=class_weights)

optimizer = optim.AdamW(model.parameters(), lr=0.0003)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# =====================================
# 4. TRAINING LOOP
# =====================================
best_acc = 0

for epoch in range(15):
    print(f"\n📚 Epoch [{epoch+1}/15]")
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    scheduler.step()

    train_acc = 100 * correct / total
    print(f"Train Loss: {running_loss/len(train_loader):.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")

    # ---------------- VALIDATION ----------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("💾 Best model saved!")

print(f"\n🏆 Best Validation Accuracy Achieved: {best_acc:.2f}%")