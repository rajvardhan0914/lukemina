import torch
from torchvision import transforms
from PIL import Image
import os

from config import Config
from model import get_model


def test_single_image(image_path, model_path=Config.BEST_MODEL_PATH):
    """Test model on a single image"""

    device = Config.DEVICE

    # Load model
    model = get_model('resnet50', Config.NUM_CLASSES, Config.PRETRAINED).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    # Class names
    class_names = {0: 'Normal', 1: 'Leukemia'}

    print("=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Image: {image_path}")
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Probabilities:")
    for i, class_name in class_names.items():
        print(f"  {class_name}: {probabilities[0, i].item() * 100:.2f}%")

    return predicted_class, confidence


def test_directory(directory_path, model_path=Config.BEST_MODEL_PATH):
    """Test model on all images in a directory"""

    device = Config.DEVICE

    # Load model
    model = get_model('resnet50', Config.NUM_CLASSES, Config.PRETRAINED).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in os.listdir(directory_path)
                  if os.path.splitext(f)[1].lower() in image_extensions]

    results = []

    print(f"Testing {len(image_files)} images from {directory_path}...")

    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()

            results.append({
                'image': image_file,
                'prediction': predicted_class,
                'confidence': confidence
            })
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    class_names = {0: 'Normal', 1: 'Leukemia'}
    for result in results:
        print(f"{result['image']}: {class_names[result['prediction']]} ({result['confidence']*100:.2f}%)")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            test_single_image(path)
        elif os.path.isdir(path):
            test_directory(path)
    else:
        print("Usage: python test_model.py <image_path or directory_path>")
