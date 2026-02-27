import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class LeukemiaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.load_dataset()

    def load_dataset(self):
        for label_name in ["all", "hem"]:
            label_folder = os.path.join(self.root_dir, label_name)

            if not os.path.exists(label_folder):
                continue

            label = 0 if label_name == "all" else 1

            for img_name in os.listdir(label_folder):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    img_path = os.path.join(label_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        print(f"Loaded {len(self.image_paths)} images from {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


# ---------------------------------------------------
# TRANSFORMS
# ---------------------------------------------------
def get_transforms(img_size=224):

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


# ---------------------------------------------------
# DATALOADERS WITH 80/20 SPLIT
# ---------------------------------------------------
def get_dataloaders(dataset_root, batch_size=8, img_size=224):

    train_transform, val_transform = get_transforms(img_size)

    full_dataset = LeukemiaDataset(
        root_dir=dataset_root,
        transform=train_transform
    )

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Assign correct transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    print(f"Total Samples: {total_size}")
    print(f"Train Samples: {train_size}")
    print(f"Validation Samples: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader