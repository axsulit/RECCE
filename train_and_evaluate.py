import os
import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from model.network.Recce import Recce
from dataset import get_dataloader
from trainer import SingleDeviceTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate RECCE model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Directory to save results')
    return parser.parse_args()

def plot_confusion_matrix(cm, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path)
    plt.close()

def evaluate_model(model, dataloader, device, split_name="test"):
    """Evaluate model on a given dataloader."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.squeeze()
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).long()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions),
        "recall": recall_score(all_labels, all_predictions),
        "f1": f1_score(all_labels, all_predictions),
        "auc": roc_auc_score(all_labels, all_probs),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions).tolist()
    }
    
    return metrics

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create experiment directory
    exp_dir = Path(args.output_dir) / config["model"]["name"] / config["exp"]["exp_id"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / "config.yml", "w") as f:
        yaml.dump(config, f)
    
    # Set device
    device = torch.device(config["train"]["device"])
    print(f"Using device: {device}")
    
    # Initialize model
    model = Recce(num_classes=config["model"]["num_classes"]).to(device)
    
    # Initialize trainer
    trainer = SingleDeviceTrainer(
        model=model,
        config=config,
        exp_dir=exp_dir,
        writer=SummaryWriter(exp_dir)
    )
    
    # Train model
    print("\nStarting training...")
    trainer.train()
    
    # Load best model for evaluation
    checkpoint = torch.load(exp_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']}")
    
    # Create dataloaders for evaluation
    train_loader = get_dataloader("train")
    val_loader = get_dataloader("val")
    test_loader = get_dataloader("test")
    
    # Evaluate on all splits
    print("\nEvaluating model on all splits...")
    splits = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    
    all_results = {}
    for split_name, loader in splits.items():
        metrics = evaluate_model(model, loader, device, split_name)
        all_results[split_name] = metrics
        
        # Print results
        print(f"\n{split_name.capitalize()} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print("\nConfusion Matrix:")
        print(np.array(metrics['confusion_matrix']))
        
        # Plot and save confusion matrix
        plot_confusion_matrix(
            np.array(metrics['confusion_matrix']),
            exp_dir / f"{split_name}_confusion_matrix.png"
        )
    
    # Save all results
    with open(exp_dir / "evaluation_results.yml", "w") as f:
        yaml.dump(all_results, f)
    
    print(f"\nAll results saved to {exp_dir}")

if __name__ == "__main__":
    main() 