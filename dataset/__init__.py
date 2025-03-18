from .abstract_dataset import AbstractDataset
from .unified_dataset import UnifiedDataset

LOADERS = {
    "CelebDF": UnifiedDataset,
    "FaceForensics": UnifiedDataset,
    "WildDeepfake": UnifiedDataset,
}


def load_dataset(name="CelebDF"):
    """
    Load the specified dataset using the unified dataset loader.
    Available datasets: CelebDF, FaceForensics, WildDeepfake
    """
    if name not in LOADERS:
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {list(LOADERS.keys())}")
    print(f"Loading dataset: '{name}' using unified loader...")
    return LOADERS[name]
