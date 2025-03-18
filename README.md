# RECCE: Deepfake Detection Model

This repository contains the implementation of a deepfake detection model that can work with multiple datasets.

## Setup with Rye

1. Install Rye if you haven't already:
```bash
curl -sSf https://rye-up.com/get | bash
```

2. Initialize the project and install dependencies:
```bash
rye sync
```

3. Activate the virtual environment:
```bash
.venv/bin/activate
```

## Dataset Structure

All datasets should follow this structure:
```
dataset_root/
    train/
        real/
            image1.jpg
            image2.jpg
            ...
        fake/
            image1.jpg
            image2.jpg
            ...
    val/
        real/
            ...
        fake/
            ...
    test/
        real/
            ...
        fake/
            ...
```

## Configuration Files

### 1. Dataset Configuration

For each dataset, you need to modify the root values in `config/dataset/` to your actual dataset path. Here's an example for Celeb-DF:

```yaml
# config/dataset/celeb_df.yml
train_cfg:
  root: "../../final_dataset/01_celebdf_unaltered"  # Path to dataset root

test_cfg:
  root: "../../final_dataset/01_celebdf_unaltered"
```

### 2. Model Configuration

Modify the following in `config/Recce.yml` for the model settings:

```yaml
# config/model/recce.yml
name: CelebDF
file: "./config/dataset/celeb_df.yml"  # Path to dataset config
```

## Running the Model

### Training
```bash
python train.py --config config/train/recce.yml 
```

### Testing
1. Make sure you have a trained model checkpoint (`.pth` file)

2. Run the test script:
```bash
python test.py --config config/train/recce.yml --ckpt path/to/your/checkpoint.pth
```

3. For inference on custom images:
```bash
python inference.py --bin path/to/model.bin --image_folder path/to/image_folder --device cuda:0 --image_size 256
```

### Testing Options
- `--config`: Path to the config file
- `--ckpt`: Path to the model checkpoint
- `--device`: Device to run on (cuda:0, cpu)
- `--image_size`: Input image size (default: 256)

### Expected Output
The test script will output metrics like accuracy, AUC, and AP scores. For inference, you'll see:
```
path: path/to/image1.jpg           | fake probability: 0.1296      | prediction: real
path: path/to/image2.jpg           | fake probability: 0.9146      | prediction: fake
```

## Troubleshooting

1. If you get CUDA errors:
   - Make sure you have the correct CUDA version installed
   - Check if your GPU is available: `nvidia-smi`
   - Try using CPU by setting `--device cpu`

2. If you get memory errors:
   - Reduce the batch size in your config file
   - Use a smaller image size
   - Try using CPU if GPU memory is insufficient

3. If you get import errors:
   - Make sure you're in the virtual environment: `.venv/bin/activate`
   - Try reinstalling dependencies: `rye sync`