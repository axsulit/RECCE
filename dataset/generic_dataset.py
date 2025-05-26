import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

from config.dataset.dataset_config import (
    CLASS_MAPPING,
    COMMON_CONFIG,
    TRAIN_CONFIG,
    VAL_CONFIG,
    TEST_CONFIG,
    get_dataset_paths,
)

class GenericDataset(Dataset):
    """Generic dataset class that can handle different deepfake datasets."""
    
    def __init__(
        self,
        split: str,
        transform_cfg: Optional[Dict[str, Any]] = None,
        subset_size: Optional[int] = None,
        balance: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            transform_cfg: Optional transform configuration
            subset_size: Optional size of subset to use
            balance: Whether to balance real and fake samples
        """
        self.split = split
        self.transform_cfg = transform_cfg or COMMON_CONFIG
        self.subset_size = subset_size
        self.balance = balance
        
        # Get dataset paths
        self.paths = get_dataset_paths()[split]
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        # Apply subset if requested
        if subset_size is not None:
            self.samples = self._apply_subset(subset_size, balance)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.transform_cfg["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all image paths and their labels."""
        samples = []
        
        # Load real images
        real_path = self.paths["real"]
        if real_path.exists():
            for img_path in real_path.glob("*.jpg"):
                samples.append((img_path, CLASS_MAPPING["real"]))
        
        # Load fake images
        fake_path = self.paths["fake"]
        if fake_path.exists():
            for img_path in fake_path.glob("*.jpg"):
                samples.append((img_path, CLASS_MAPPING["fake"]))
        
        return samples
    
    def _apply_subset(
        self,
        subset_size: int,
        balance: bool
    ) -> List[Tuple[Path, int]]:
        """Apply subset selection to the dataset."""
        if balance:
            # Separate real and fake samples
            real_samples = [s for s in self.samples if s[1] == CLASS_MAPPING["real"]]
            fake_samples = [s for s in self.samples if s[1] == CLASS_MAPPING["fake"]]
            
            # Calculate samples per class
            samples_per_class = subset_size // 2
            
            # Randomly sample from each class
            real_samples = random.sample(real_samples, min(samples_per_class, len(real_samples)))
            fake_samples = random.sample(fake_samples, min(samples_per_class, len(fake_samples)))
            
            return real_samples + fake_samples
        else:
            # Randomly sample from all samples
            return random.sample(self.samples, min(subset_size, len(self.samples)))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label

def get_dataloader(
    split: str,
    transform_cfg: Optional[Dict[str, Any]] = None,
    subset_size: Optional[int] = None,
    balance: bool = True
) -> DataLoader:
    """
    Get a DataLoader for the specified split.
    
    Args:
        split: Dataset split ('train', 'val', or 'test')
        transform_cfg: Optional transform configuration
        subset_size: Optional size of subset to use
        balance: Whether to balance real and fake samples
    
    Returns:
        DataLoader for the specified split
    """
    # Get appropriate config
    if split == "train":
        config = TRAIN_CONFIG
    elif split == "val":
        config = VAL_CONFIG
    else:
        config = TEST_CONFIG
    
    # Create dataset
    dataset = GenericDataset(
        split=split,
        transform_cfg=transform_cfg or config,
        subset_size=subset_size,
        balance=balance
    )
    
    # Create dataloader
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    ) 