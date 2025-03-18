import numpy as np
from glob import glob
from os import listdir
from os.path import join
from dataset import AbstractDataset

SPLITS = ["train", "val", "test"]


class CelebDF(AbstractDataset):
    """
    Celeb-DF v2 Dataset with simplified folder structure:
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
        super(CelebDF, self).__init__(cfg, seed, transforms, transform, target_transform)
        print(f"Loading data from 'Celeb-DF' of split '{cfg['split']}'"
              f"\nPlease wait patiently...")
        self.categories = ['real', 'fake']
        self.root = cfg['root']
        
        # Get images and targets for the specified split
        self.images, self.targets = self.__get_images(cfg['split'], cfg.get('balance', False))
        assert len(self.images) == len(self.targets), "The number of images and targets not consistent."
        print("Data from 'Celeb-DF' loaded.\n")
        print(f"Dataset contains {len(self.images)} images.\n")

    def __get_images(self, split, balance=False):
        real = list()
        fake = list()
        
        # Get real images (both jpg and png)
        real.extend(glob(join(self.root, split, 'real', '*.jpg')))
        real.extend(glob(join(self.root, split, 'real', '*.png')))
        # Get fake images (both jpg and png)
        fake.extend(glob(join(self.root, split, 'fake', '*.jpg')))
        fake.extend(glob(join(self.root, split, 'fake', '*.png')))
        
        # print the file locations above
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


if __name__ == '__main__':
    import yaml

    config_path = "../config/dataset/celeb_df.yml"
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = config["train_cfg"]
    # config = config["test_cfg"]

    def run_dataset():
        dataset = CelebDF(config)
        print(f"dataset: {len(dataset)}")
        for i, _ in enumerate(dataset):
            path, target = _
            print(f"path: {path}, target: {target}")
            if i >= 9:
                break

    def run_dataloader(display_samples=False):
        from torch.utils import data
        import matplotlib.pyplot as plt

        dataset = CelebDF(config)
        dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)
        print(f"dataset: {len(dataset)}")
        for i, _ in enumerate(dataloader):
            path, targets = _
            image = dataloader.dataset.load_item(path)
            print(f"image: {image.shape}, target: {targets}")
            if display_samples:
                plt.figure()
                img = image[0].permute([1, 2, 0]).numpy()
                plt.imshow(img)
                # plt.savefig("./img_" + str(i) + ".png")
                plt.show()
            if i >= 9:
                break


    ###########################
    # run the functions below #
    ###########################

    run_dataset()
    # run_dataloader(False)
