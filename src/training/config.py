import torch
from pathlib import Path

# ==========================================
# PROJECT DIRECTORIES
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data" / "raw"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
OUTPUT_DIR = PROJECT_DIR / "reports" / "figures"

# Create required directories
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# DATASET CONFIGURATION
# ==========================================

NUM_CLASSES = 7
IMAGE_SIZE = 28
IMAGE_CHANNELS = 3


# ==========================================
# TRAINING CONFIGURATION
# ==========================================

BATCH_SIZE = 32
NUM_EPOCHS = 20

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


# ==========================================
# OPTIMIZER CONFIGURATION
# ==========================================

OPTIMIZER = "AdamW"
SCHEDULER = "ReduceLROnPlateau"


# ==========================================
# HARDWARE CONFIGURATION
# ==========================================

# Temporarily use CPU to avoid the Apple MPS crash

DEVICE = torch.device("cpu")


# ==========================================
# REPRODUCIBILITY
# ==========================================

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

print(f"Using device: {DEVICE}")
print("DermaBench configuration loaded successfully!")
import torch
from pathlib import Path


# Project directories
PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data" / "raw"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
OUTPUT_DIR = PROJECT_DIR / "reports" / "figures"

# Create required directories
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration
NUM_CLASSES = 7
IMAGE_SIZE = 28
IMAGE_CHANNELS = 3

# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

# Optimizer configuration
OPTIMIZER = "AdamW"
SCHEDULER = "ReduceLROnPlateau"

# Hardware configuration
DEVICE = torch.device("cpu")


# Reproducibility
RANDOM_SEED = 42

print("DermaBench configuration loaded successfully!")
