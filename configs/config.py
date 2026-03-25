import torch

# Dataset
DATASET_PATH = r"C:\Projects\lukemina\ALL_IDB\segmented"

IMG_SIZE = 224
BATCH_SIZE = 32

# Model
MODEL_BACKBONE = "resnet18"
NUM_CLASSES = 2

# Training
NUM_EPOCHS = 30
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output
OUTPUT_DIR = "outputs_IDE"
CHECKPOINT_DIR = "checkpoints"