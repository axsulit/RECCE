import os
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from model.network.Recce import Recce
from trainer import SingleDeviceTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train RECCE model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create experiment directory
    exp_dir = Path("runs") / config["model"]["name"] / config["exp"]["exp_id"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / "config.yml", "w") as f:
        yaml.dump(config, f)
    
    # Initialize model
    model = Recce(num_classes=config["model"]["num_classes"]).to(config["train"]["device"])
    
    # Initialize trainer
    trainer = SingleDeviceTrainer(
        model=model,
        config=config,
        exp_dir=exp_dir,
        writer=SummaryWriter(exp_dir)
    )
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    main() 