from pathlib import Path
from typing import Dict, List
import yaml
import os

def load_dataset_paths() -> Dict:
    """Load dataset paths from the central configuration file."""
    config_path = Path(__file__).parent / "dataset_paths.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

# Load dataset configuration
DATASET_PATHS = load_dataset_paths()
DATASET_ROOT = Path(DATASET_PATHS["dataset_root"])

# Dataset structure
DATASET_STRUCTURE = DATASET_PATHS["structure"]

# Class mappings
CLASS_MAPPING = {
    "real": 0,
    "fake": 1
}

# Common configuration for all splits
COMMON_CONFIG = {
    "image_size": (256, 256),  # (height, width)
    "num_workers": 4,
    "pin_memory": True,
}

# Dataset configuration
TRAIN_CONFIG = {
    **COMMON_CONFIG,
    "batch_size": 8,
    "shuffle": True,
}

# Validation settings
VAL_CONFIG = {
    **COMMON_CONFIG,
    "batch_size": 8,
    "shuffle": False,
}

# Test settings
TEST_CONFIG = {
    **COMMON_CONFIG,
    "batch_size": 8,
    "shuffle": False,
}

def get_dataset_paths() -> Dict[str, Dict[str, Path]]:
    """
    Returns a dictionary containing paths for real and fake images for each split.
    """
    paths = {}
    for split_name, split_dir in DATASET_STRUCTURE["splits"].items():
        split_path = DATASET_ROOT / split_dir
        paths[split_name] = {
            "real": split_path / DATASET_STRUCTURE["classes"]["real"],
            "fake": split_path / DATASET_STRUCTURE["classes"]["fake"]
        }
    return paths

__all__ = [
    'DATASET_ROOT',
    'DATASET_STRUCTURE',
    'CLASS_MAPPING',
    'COMMON_CONFIG',
    'TRAIN_CONFIG',
    'VAL_CONFIG',
    'TEST_CONFIG',
    'get_dataset_paths'
] 