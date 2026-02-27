import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from config import Config
from model import get_model


class GradCAM:
    """Grad-CAM implementation for visualization"""

    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.activation = None
        self.gradient = None

        # Register hooks
        self.layer.register_forward_hook(self.save_activation)
        self.layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].detach()

    def generate_cam(self, input_tensor):
        """Generate Class Activation Map"""
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        target_class = output.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Generate CAM
        weights = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activation).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze(0).squeeze(0).cpu().detach().numpy()

        return cam, target_class.item()


def visualize_gradcam(image_path, model_path=Config.BEST_MODEL_PATH):
    """Visualize model predictions with Grad-CAM"""

    device = Config.DEVICE

    # Load model
    model = get_model('resnet50', Config.NUM_CLASSES, Config.PRETRAINED).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get the last convolutional layer
    last_conv_layer = model.resnet.layer4[-1].conv3

    # Create Grad-CAM
    grad_cam = GradCAM(model, last_conv_layer)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    original_image_resized = original_image.resize((Config.INPUT_SIZE, Config.INPUT_SIZE))
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    # Generate CAM
    cam, predicted_class = grad_cam.generate_cam(image_tensor)

    # Normalize CAM
    cam_normalized = (cam - cam.min()) / (cam.max() - cam.min())

    # Resize CAM to match image size
    cam_resized = cv2.resize(cam_normalized, (Config.INPUT_SIZE, Config.INPUT_SIZE))

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)

    # Convert original image to numpy
    original_np = np.array(original_image_resized)

    # Overlay heatmap
    overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    class_names = {0: 'Normal', 1: 'Leukemia'}

    axes[0].imshow(original_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(cam_resized, cmap='hot')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Prediction: {class_names[predicted_class]}')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_output.png', dpi=150, bbox_inches='tight')
    print(f"📊 Grad-CAM visualization saved!")
    print(f"Predicted class: {class_names[predicted_class]}")

    return cam_resized, predicted_class


def batch_gradcam(directory_path, model_path=Config.BEST_MODEL_PATH, num_samples=9):
    """Generate Grad-CAM for multiple images"""

    device = Config.DEVICE

    # Load model
    model = get_model('resnet50', Config.NUM_CLASSES, Config.PRETRAINED).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get the last convolutional layer
    last_conv_layer = model.resnet.layer4[-1].conv3

    # Create Grad-CAM
    grad_cam = GradCAM(model, last_conv_layer)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Get image files
    import os
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in os.listdir(directory_path)
                  if os.path.splitext(f)[1].lower() in image_extensions][:num_samples]

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    class_names = {0: 'Normal', 1: 'Leukemia'}

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(directory_path, image_file)

        try:
            original_image = Image.open(image_path).convert('RGB')
            original_image_resized = original_image.resize((Config.INPUT_SIZE, Config.INPUT_SIZE))
            image_tensor = transform(original_image).unsqueeze(0).to(device)

            cam, predicted_class = grad_cam.generate_cam(image_tensor)
            cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-5)
            cam_resized = cv2.resize(cam_normalized, (Config.INPUT_SIZE, Config.INPUT_SIZE))
            heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(np.array(original_image_resized), 0.6, heatmap, 0.4, 0)

            axes[idx, 0].imshow(np.array(original_image_resized))
            axes[idx, 0].set_title(f'Original: {image_file}')
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(cam_resized, cmap='hot')
            axes[idx, 1].set_title('Attention Map')
            axes[idx, 1].axis('off')

            axes[idx, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[idx, 2].set_title(f'Pred: {class_names[predicted_class]}')
            axes[idx, 2].axis('off')
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    plt.tight_layout()
    plt.savefig('batch_gradcam_output.png', dpi=150, bbox_inches='tight')
    print(f"📊 Batch Grad-CAM visualization saved!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            visualize_gradcam(path)
        elif os.path.isdir(path):
            batch_gradcam(path)
    else:
        print("Usage: python gradcam.py <image_path or directory_path>")
