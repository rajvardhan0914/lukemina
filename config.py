import torch

class Config:
    """Configuration for leukemia classification model"""

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data paths
    TRAIN_DIR = 'dataset/train'
    VAL_DIR = 'dataset/val'
    TEST_DIR = 'dataset/C-NMC 2019 (PKG)/C-NMC_test_prelim_phase_data'

    # Model parameters
    INPUT_SIZE = 224
    NUM_CLASSES = 2  # Normal vs Leukemia
    PRETRAINED = True

    # Training hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Optimizer
    OPTIMIZER = 'adamw'

    # Scheduler
    SCHEDULER = 'cosine'
    T_MAX = EPOCHS

    # Early stopping
    PATIENCE = 5
    MIN_DELTA = 1e-3

    # Checkpoints
    CHECKPOINT_DIR = 'checkpoints'
    BEST_MODEL_PATH = 'best_model.pth'
    BEST_LEUKEMIA_MODEL_PATH = 'best_leukemia_model.pth'

    # Random seed
    SEED = 42

    # Data augmentation
    AUGMENT = True

    # Logging
    LOG_INTERVAL = 10
