# RECCE CVPR 2022

:page_facing_up: End-to-End Reconstruction-Classification Learning for Face Forgery Detection

:boy: Junyi Cao, Chao Ma, Taiping Yao, Shen Chen, Shouhong Ding, Xiaokang Yang

**Please consider citing our paper if you find it interesting or helpful to your research.**
```
@InProceedings{Cao_2022_CVPR,
    author    = {Cao, Junyi and Ma, Chao and Yao, Taiping and Chen, Shen and Ding, Shouhong and Yang, Xiaokang},
    title     = {End-to-End Reconstruction-Classification Learning for Face Forgery Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {4113-4122}
}
```

----

### Introduction

This repository is an implementation for *End-to-End Reconstruction-Classification Learning for Face Forgery Detection* presented in CVPR 2022. In the paper, we propose a novel **REC**onstruction-**C**lassification l**E**arning framework called **RECCE** to detect face forgeries. The code is based on Pytorch. Please follow the instructions below to get started.

### Motivation

Briefly, we train a reconstruction network over genuine images only and use the output of the latent feature by the encoder to perform binary classification. Due to the discrepancy in the data distribution between genuine and forged faces, the reconstruction differences of forged faces are obvious and also indicate the probably forged regions. 

### Basic Requirements
Please ensure that you have already installed the following packages.
- [Pytorch](https://pytorch.org/get-started/previous-versions/) 1.7.1
- [Torchvision](https://pytorch.org/get-started/previous-versions/) 0.8.2
- [Albumentations](https://github.com/albumentations-team/albumentations#spatial-level-transforms) 1.0.3
- [Timm](https://github.com/rwightman/pytorch-image-models) 0.3.4
- [TensorboardX](https://pypi.org/project/tensorboardX/#history) 2.1
- [Scipy](https://pypi.org/project/scipy/#history) 1.5.2
- [PyYaml](https://pypi.org/project/PyYAML/#history) 5.3.1
- [scikit-learn](https://scikit-learn.org/) (for evaluation metrics)
- [seaborn](https://seaborn.pydata.org/) (for visualization)

### Dataset Preparation
1. Download the Celeb-DF dataset from [here](https://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html)
2. Extract facial images from videos using [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)
3. Organize the dataset in the following structure:
```
celeb-df-dataset/
├── train/
│   ├── real/
│   │   └── [real face images]
│   └── fake/
│       └── [fake face images]
├── val/
│   ├── real/
│   │   └── [real face images]
│   └── fake/
│       └── [fake face images]
└── test/
    ├── real/
    │   └── [real face images]
    └── fake/
        └── [fake face images]
```

### Project Structure
```
RECCE/
├── config/
│   ├── dataset/
│   │   ├── dataset_paths.yml    # Dataset root location
│   │   └── dataset_full.yml     # Full training configuration
│   └── model/
│       └── Recce.yml           # Model and training parameters
├── dataset/
│   └── generic_dataset.py      # Dataset loading and preprocessing
├── model/
│   └── network/
│       └── Recce.py           # RECCE model architecture
├── trainer/
│   └── single_device_trainer.py # Training logic
├── train_and_evaluate.py       # Combined training and evaluation script
└── inference.py                # Inference on custom images
```

### Training and Evaluation
We provide a single script that handles both training and evaluation. The script will:
1. Train the model
2. Save the best model checkpoint
3. Evaluate on train, validation, and test splits
4. Generate comprehensive metrics and visualizations

To run the complete pipeline:
```bash
python train_and_evaluate.py --config config/model/Recce.yml
```

The results will be saved in `experiments/Recce/<exp_id>/` with:
- `config.yml`: Training configuration
- `best_model.pt`: Best model checkpoint
- `evaluation_results.yml`: All metrics for train/val/test
- Confusion matrix plots for each split
- TensorBoard logs for training monitoring

Monitor training progress:
```bash
tensorboard --logdir experiments/Recce
```

### Inference
To run inference on custom images:
```bash
python inference.py --bin path/to/best_model.pt --image_folder path/to/images --device cuda:0 --image_size 256
```

Arguments:
- `--bin`: Path to the trained model checkpoint
- `--image_folder`: Directory containing images to test
- `--device`: Device to run inference on (cpu, cuda:0, etc.)
- `--image_size`: Input image size

The script will output predictions for each image:
```
path: image1.jpg | fake probability: 0.1296 | prediction: real
path: image2.jpg | fake probability: 0.9146 | prediction: fake
```

### Acknowledgement
- We thank Qiqi Gu for helping plot the schematic diagram of the proposed method in the manuscript.
