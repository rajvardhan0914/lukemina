import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

from configs.config import *
from variliteformer.models.resnet_transformer import ResNetTransformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):

        self.model.zero_grad()

        output = self.model(input_tensor)

        pred_class = output.argmax(dim=1)

        score = output[0, pred_class.item()]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, pred_class.item()


def run_gradcam(image_path):

    os.makedirs("outputs/gradcam", exist_ok=True)

    # Load model
    model = ResNetTransformer(MODEL_BACKBONE, NUM_CLASSES)

    model.load_state_dict(
        torch.load(f"{CHECKPOINT_DIR}/best_{MODEL_BACKBONE}.pth", map_location=device)
    )

    model.to(device)
    model.eval()

    # Target conv layer
    target_layer = model.cnn.layer4[-1]

    gradcam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))

    input_tensor = transform(image).unsqueeze(0).to(device)

    cam, pred_class = gradcam.generate(input_tensor)

    cam = cv2.resize(cam,(IMG_SIZE,IMG_SIZE))

    heatmap = cv2.applyColorMap(
        np.uint8(255*cam),
        cv2.COLORMAP_JET
    )

    image_np = np.array(image_resized)

    overlay = cv2.addWeighted(image_np,0.6,heatmap,0.4,0)

    save_path = "outputs/gradcam/gradcam_result.png"

    cv2.imwrite(save_path,overlay)

    print("GradCAM saved:",save_path)


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print("Usage: python gradcam.py image_path")

    else:

        run_gradcam(sys.argv[1])