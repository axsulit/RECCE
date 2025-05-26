import os
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from dataset import get_dataloader
from trainer.abstract_trainer import AbstractTrainer
from trainer.utils import exp_recons_loss, MLLoss, AccMeter, AUCMeter, AverageMeter, center_print

class SingleDeviceTrainer(AbstractTrainer):
    """Trainer for single device training (CPU or single GPU/MPS)."""
    
    def __init__(self, model, config, exp_dir, writer):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            config: Training configuration
            exp_dir: Experiment directory for saving checkpoints and logs
            writer: TensorBoard writer
        """
        self.model = model
        self.config = config
        self.exp_dir = exp_dir
        self.writer = writer
        
        # Get configurations
        self.model_cfg = config["model"]
        self.data_cfg = config["data"]
        self.train_cfg = config["train"]
        self.config_cfg = config["config"]
        
        # Set device
        self.device = torch.device(self.train_cfg["device"])
        print(f"Using device: {self.device}")
        
        # Create dataloaders
        with open(self.data_cfg["dataset_file"]) as f:
            dataset_config = yaml.safe_load(f)
        
        self.train_loader = get_dataloader(
            "train",
            transform_cfg=dataset_config["train_cfg"]["transform"],
            subset_size=dataset_config["train_cfg"].get("subset_size"),
            balance=dataset_config["train_cfg"].get("balance", True)
        )
        
        self.val_loader = get_dataloader(
            "val",
            transform_cfg=dataset_config["val_cfg"]["transform"],
            subset_size=dataset_config["val_cfg"].get("subset_size"),
            balance=dataset_config["val_cfg"].get("balance", True)
        )
        
        # Setup optimizer and scheduler
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config_cfg["lr"],
            weight_decay=self.config_cfg["weight_decay"]
        )
        
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.config_cfg["step_size"],
            gamma=self.config_cfg["gamma"]
        )
        
        # Setup loss
        self.loss_criterion = nn.BCEWithLogitsLoss()
        
        # Training settings
        self.lambda_1 = self.config_cfg["lambda_1"]
        self.lambda_2 = self.config_cfg["lambda_2"]
        self.contra_loss = MLLoss()
        
        # Metrics
        self.acc_meter = AccMeter()
        self.loss_meter = AverageMeter()
        self.recons_loss_meter = AverageMeter()
        self.contra_loss_meter = AverageMeter()
        
        # Training parameters
        self.num_epochs = self.train_cfg["num_epochs"]
        self.val_interval = self.train_cfg["val_interval"]
        
        # Best model tracking
        self.best_acc = 0.0
    
    def train(self):
        """Train the model."""
        try:
            timer = time.time()
            grad_scaler = GradScaler()
            center_print("Training begins......")
            
            for epoch in range(self.num_epochs):
                # Training
                self.model.train()
                self.acc_meter.reset()
                self.loss_meter.reset()
                self.recons_loss_meter.reset()
                self.contra_loss_meter.reset()
                
                train_iter = tqdm(self.train_loader, desc=f"Epoch {epoch}")
                for batch_idx, (images, labels) in enumerate(train_iter):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    with autocast():
                        outputs = self.model(images)
                        
                        # BCE loss
                        outputs = outputs.squeeze()
                        loss = self.loss_criterion(outputs, labels.float())
                        outputs = torch.sigmoid(outputs)
                        
                        # Additional losses
                        recons_loss = exp_recons_loss(self.model.loss_inputs['recons'], (images, labels))
                        contra_loss = self.contra_loss(self.model.loss_inputs['contra'], labels)
                        loss += self.lambda_1 * recons_loss + self.lambda_2 * contra_loss
                    
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(self.optimizer)
                    grad_scaler.update()
                    
                    # Update metrics
                    self.acc_meter.update(outputs, labels, True)
                    self.loss_meter.update(loss.item())
                    self.recons_loss_meter.update(recons_loss.item())
                    self.contra_loss_meter.update(contra_loss.item())
                    
                    # Update progress bar
                    train_iter.set_postfix({
                        "loss": f"{self.loss_meter.avg:.4f}",
                        "acc": f"{self.acc_meter.mean_acc():.4f}"
                    })
                
                # Validation
                if (epoch + 1) % self.val_interval == 0:
                    self.validate(epoch)
                
                # Step scheduler
                self.scheduler.step()
                
                # Save checkpoint
                self.save_checkpoint(epoch)
            
            if self.writer is not None:
                self.writer.close()
            center_print("Training process ends.")
            
        except Exception as e:
            print(f"Training failed: {e}")
            raise e
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        acc_meter = AccMeter()
        auc_meter = AUCMeter()
        loss_meter = AverageMeter()
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                outputs = outputs.squeeze()
                loss = self.loss_criterion(outputs, labels.float())
                outputs = torch.sigmoid(outputs)
                
                acc_meter.update(outputs, labels, True)
                auc_meter.update(outputs, labels, True)
                loss_meter.update(loss.item())
        
        val_acc = acc_meter.mean_acc()
        val_auc = auc_meter.mean_auc()
        val_loss = loss_meter.avg
        
        print(f"\nValidation - Epoch {epoch}")
        print(f"Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        if self.writer is not None:
            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.writer.add_scalar("val/acc", val_acc, epoch)
            self.writer.add_scalar("val/auc", val_auc, epoch)
        
        # Save best model
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.save_checkpoint(epoch, best=True)
    
    def save_checkpoint(self, epoch, best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_acc": self.best_acc
        }
        
        # Save latest
        torch.save(checkpoint, self.exp_dir / "latest_model.pt")
        
        # Save best if requested
        if best:
            torch.save(checkpoint, self.exp_dir / "best_model.pt") 