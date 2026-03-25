import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from segmentation.nucleus_segmenter import segment_nucleus


class LeukemiaDataset(Dataset):
    """
    Custom dataset for leukemia cell classification.
    Applies nucleus segmentation before transform pipeline
    to support region-adaptive feature extraction.
    """

    def __init__(self, image_paths, labels, transform=None,
                 apply_segmentation=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.apply_segmentation = apply_segmentation

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Region-adaptive preprocessing: isolate nucleus region
        if self.apply_segmentation:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_bgr = segment_nucleus(img_bgr)
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label


def get_dataloaders(dataset_path, img_size, batch_size,
                    apply_segmentation=True):
    """
    Build stratified train/val DataLoaders with optional
    nucleus segmentation for region-adaptive processing.
    """
    image_paths = []
    labels = []

    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            image_paths.append(img_path)

            # Binary mapping: benign → 0, leukemic → 1
            if class_name.lower() == "benign":
                labels.append(0)
            else:
                labels.append(1)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = LeukemiaDataset(
        train_paths, train_labels, train_transform,
        apply_segmentation=apply_segmentation
    )
    val_dataset = LeukemiaDataset(
        val_paths, val_labels, val_transform,
        apply_segmentation=apply_segmentation
    )

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