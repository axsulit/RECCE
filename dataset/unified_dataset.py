import torch
import numpy as np
import os
from glob import glob
from os.path import join
from PIL import Image
from dataset import AbstractDataset

SPLITS = ["train", "val", "test"]

class UnifiedDataset(AbstractDataset):
    """
    Unified Dataset loader that handles dataset-specific configurations
    while providing a common loading interface.
    Expected folder structure:
    root/
        train/
            real/
            fake/
        val/
            real/
            fake/
        test/
            real/
            fake/
    """
    def __init__(self, cfg, seed=2022, transforms=None, transform=None, target_transform=None):
        # pre-check
        if cfg['split'] not in SPLITS:
            raise ValueError(f"split should be one of {SPLITS}, but found {cfg['split']}.")
            
        super(UnifiedDataset, self).__init__(
            cfg, seed, transforms, transform, target_transform)
        
        # Handle dataset-specific configurations
        self.dataset_name = cfg.get('name', None)
        if not self.dataset_name:
            raise ValueError("Dataset name must be specified in the config")
            
        print(f"\nDataset configuration:")
        print(f"Name: {self.dataset_name}")
        print(f"Root: {self.root}")
        print(f"Split: {cfg['split']}")
        
        # Get images and targets for the specified split
        self.images, self.targets = self.__get_images(cfg['split'], cfg.get('balance', False))
        assert len(self.images) == len(self.targets), "The number of images and targets not consistent."
        print(f"Dataset '{self.dataset_name}' loaded successfully.")
        print(f"Total images: {len(self.images)}")
        print(f"Class distribution:")
        real_count = sum(1 for t in self.targets if t == 0)
        fake_count = sum(1 for t in self.targets if t == 1)
        print(f"  real: {real_count}")
        print(f"  fake: {fake_count}\n")

    def __get_images(self, split, balance=False):
        real = list()
        fake = list()
        
        # Get real images (both jpg and png)
        real.extend(glob(join(self.root, split, 'real', '*.jpg')))
        real.extend(glob(join(self.root, split, 'real', '*.png')))
        # Get fake images (both jpg and png)
        fake.extend(glob(join(self.root, split, 'fake', '*.jpg')))
        fake.extend(glob(join(self.root, split, 'fake', '*.png')))
        
        # Print search locations and results
        print(f"Searching for real images in: {join(self.root, split, 'real', '*.jpg')} and {join(self.root, split, 'real', '*.png')}")
        print(f"Searching for fake images in: {join(self.root, split, 'fake', '*.jpg')} and {join(self.root, split, 'fake', '*.png')}")
        print(f"Found {len(real)} real images and {len(fake)} fake images")
        
        if len(real) == 0 or len(fake) == 0:
            print("WARNING: No images found in one or both categories!")
            if len(real) == 0:
                print("No real images found!")
            if len(fake) == 0:
                print("No fake images found!")
        
        if balance:
            min_count = min(len(real), len(fake))
            real = np.random.choice(real, size=min_count, replace=False)
            fake = np.random.choice(fake, size=min_count, replace=False)
            print(f"After Balance | Real: {len(real)}, Fake: {len(fake)}")
        
        real_tgt = [0] * len(real)  # 0 for real
        fake_tgt = [1] * len(fake)  # 1 for fake
        
        return [*real, *fake], [*real_tgt, *fake_tgt]

    def load_item(self, path):
        """Load and preprocess a single image."""
        # Handle batch of paths
        if isinstance(path, (list, tuple)):
            images = [self._load_single_item(p) for p in path]
            # Convert list of images to tensor
            return torch.stack(images)
        # Handle single path
        return self._load_single_item(path)
        
    def _load_single_item(self, path):
        """Load and preprocess a single image file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
            
        # Load image using PIL
        image = Image.open(path).convert('RGB')
        
        # Apply transforms if any
        if self.transforms is not None:
            # Convert PIL image to numpy array for albumentations
            image = np.array(image)
            # Apply transforms with named arguments
            transformed = self.transforms(image=image)
            # Get the transformed image
            image = transformed['image']
            # Convert back to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
        # Convert PIL Image to tensor
        if isinstance(image, Image.Image):
            image = torch.from_numpy(np.array(image)).float()
            # Rearrange dimensions from HWC to CHW
            image = image.permute(2, 0, 1)
            
        return image

if __name__ == '__main__':
    import yaml
    from torch.utils import data
    import matplotlib.pyplot as plt

    def run_dataloader(display_samples=False):
        # Load config
        config_path = "../config/dataset/celeb_df.yml"
        with open(config_path) as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        config = config["train_cfg"]

        # Create dataset and dataloader
        dataset = UnifiedDataset(config)
        dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)
        
        print(f"Dataset size: {len(dataset)}")
        for i, (paths, targets) in enumerate(dataloader):
            images = torch.stack([dataset.load_item(path) for path in paths])
            print(f"Batch {i}: images shape: {images.shape}, targets: {targets}")
            
            if display_samples:
                plt.figure()
                img = images[0].permute([1, 2, 0]).numpy()
                plt.imshow(img)
                plt.title(f"Target: {targets[0]}")
                plt.show()
            
            if i >= 9:
                break

    run_dataloader(True) 