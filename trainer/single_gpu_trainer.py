import os
import sys
import time
import math
import yaml
import torch
import random
import numpy as np

from tqdm import tqdm
from pprint import pprint
from torch.utils import data
from torch.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

from dataset import load_dataset
from loss import get_loss
from model import load_model
from optimizer import get_optimizer
from scheduler import get_scheduler
from trainer import AbstractTrainer, LEGAL_METRIC
from trainer.utils import exp_recons_loss, MLLoss, center_print
from trainer.utils import MODELS_PATH, AccMeter, AUCMeter, AverageMeter, Logger, Timer


class SingleGpuTrainer(AbstractTrainer):
    def __init__(self, config, stage="Train"):
        # Set device first
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        
        # Add device to config
        if 'config' not in config:
            config['config'] = {}
        config['config']['device'] = device
        
        # Set default tensor type for MPS if needed
        if device.type == 'mps':
            torch.set_default_tensor_type('torch.FloatTensor')
        
        # Initialize base settings
        self._initiated_settings()
        
        # Then call parent class initialization
        super(SingleGpuTrainer, self).__init__(config, stage)
        
        # Ensure device is set after parent initialization
        self.device = device
        np.random.seed(2021)

    def _initiated_settings(self, model_cfg=None, data_cfg=None, config_cfg=None):
        self.local_rank = 0

    def _train_settings(self, model_cfg, data_cfg, config_cfg):
        # debug mode: no log dir, no train_val operation.
        self.debug = config_cfg["debug"]
        print(f"Using debug mode: {self.debug}.")
        print("*" * 20)

        self.eval_metric = config_cfg["metric"]
        if self.eval_metric not in LEGAL_METRIC:
            raise ValueError(f"Evaluation metric must be in {LEGAL_METRIC}, but found "
                             f"{self.eval_metric}.")
        if self.eval_metric == LEGAL_METRIC[-1]:
            self.best_metric = 1.0e8

        # load training dataset
        train_dataset = data_cfg["file"]
        branch = data_cfg["train_branch"]
        name = data_cfg["name"]
        with open(train_dataset, "r") as f:
            options = yaml.load(f, Loader=yaml.FullLoader)
        train_options = options[branch]
        self.train_set = load_dataset(name)(train_options)
        # wrapped with data loader
        self.train_loader = data.DataLoader(self.train_set, shuffle=True,
                                          num_workers=data_cfg.get("num_workers", 4),
                                          batch_size=data_cfg["train_batch_size"])

        # load validation dataset
        val_options = options[data_cfg["val_branch"]]
        self.val_set = load_dataset(name)(val_options)
        # wrapped with data loader
        self.val_loader = data.DataLoader(self.val_set, shuffle=True,
                                        num_workers=data_cfg.get("num_workers", 4),
                                        batch_size=data_cfg["val_batch_size"])

        self.resume = config_cfg.get("resume", False)

        if not self.debug:
            time_format = "%Y-%m-%d...%H.%M.%S"
            run_id = time.strftime(time_format, time.localtime(time.time()))
            self.run_id = config_cfg.get("id", run_id)
            self.dir = os.path.join("runs", self.model_name, self.run_id)

            if not self.resume:
                if os.path.exists(self.dir):
                    raise ValueError("Error: given id '%s' already exists." % self.run_id)
                os.makedirs(self.dir, exist_ok=True)
                print(f"Writing config file to file directory: {self.dir}.")
                yaml.dump({"config": self.config,
                           "train_data": train_options,
                           "val_data": val_options},
                          open(os.path.join(self.dir, 'train_config.yml'), 'w'))
                # copy the script for the training model
                model_file = MODELS_PATH[self.model_name]
                os.system("cp " + model_file + " " + self.dir)
            else:
                print(f"Resuming the history in file directory: {self.dir}.")

            print(f"Logging directory: {self.dir}.")

            # redirect the std out stream
            sys.stdout = Logger(os.path.join(self.dir, 'records.txt'))
            center_print('Train configurations begins.')
            pprint(self.config)
            pprint(train_options)
            pprint(val_options)
            center_print('Train configurations ends.')

        # load model
        self.num_classes = model_cfg["num_classes"]
        
        # Create model
        self.model = load_model(self.model_name)(**model_cfg)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Force all model parameters to be on the same device
        for param in self.model.parameters():
            param.data = param.data.to(self.device)
            
        # Move all buffers to device as well
        for buffer in self.model.buffers():
            buffer.data = buffer.data.to(self.device)
            
        # Ensure all submodules are on the correct device
        for module in self.model.modules():
            if hasattr(module, 'to'):
                module.to(self.device)
                
        # Ensure the model is in train mode
        self.model.train()
        
        # Verify device placement
        print("\nDevice verification:")
        print(f"Model device: {next(self.model.parameters()).device}")
        if hasattr(self.model, 'encoder'):
            print(f"Encoder device: {next(self.model.encoder.parameters()).device}")
        print(f"Target device: {self.device}")

        # load optimizer
        optim_cfg = config_cfg.get("optimizer", None)
        optim_name = optim_cfg.pop("name")
        self.optimizer = get_optimizer(optim_name)(self.model.parameters(), **optim_cfg)
        # load scheduler
        self.scheduler = get_scheduler(self.optimizer, config_cfg.get("scheduler", None))
        # load loss
        self.loss_criterion = get_loss(config_cfg.get("loss", None), device=self.device)

        # total number of steps (or epoch) to train
        self.num_steps = train_options["num_steps"]
        self.num_epoch = math.ceil(self.num_steps / len(self.train_loader))

        # the number of steps to write down a log
        self.log_steps = train_options["log_steps"]
        # the number of steps to validate on val dataset once
        self.val_steps = train_options["val_steps"]

        # balance coefficients
        self.lambda_1 = config_cfg["lambda_1"]
        self.lambda_2 = config_cfg["lambda_2"]
        self.warmup_step = config_cfg.get('warmup_step', 0)

        self.contra_loss = MLLoss()
        self.acc_meter = AccMeter()
        self.loss_meter = AverageMeter()
        self.recons_loss_meter = AverageMeter()
        self.contra_loss_meter = AverageMeter()

        if self.resume:
            self._load_ckpt(best=config_cfg.get("resume_best", False), train=True)

    def _test_settings(self, model_cfg, data_cfg, config_cfg):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")

    def _load_ckpt(self, best=False, train=False):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")

    def _save_ckpt(self, step, best=False):
        save_dir = os.path.join(self.dir, f"best_model_{step}.bin" if best else "latest_model.bin")
        torch.save({
            "step": step,
            "best_step": self.best_step,
            "best_metric": self.best_metric,
            "eval_metric": self.eval_metric,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, save_dir)

    def train(self):
        try:
            timer = Timer()
            # Initialize GradScaler only for CUDA
            grad_scalar = GradScaler() if self.device.type == 'cuda' else None
            writer = None if self.debug else SummaryWriter(log_dir=self.dir)
            center_print("Training begins......")
            start_epoch = self.start_step // len(self.train_loader) + 1
            for epoch_idx in range(start_epoch, self.num_epoch + 1):
                # reset meter
                self.acc_meter.reset()
                self.loss_meter.reset()
                self.recons_loss_meter.reset()
                self.contra_loss_meter.reset()
                self.optimizer.step()

                train_generator = tqdm(enumerate(self.train_loader, 1), position=0, leave=True)

                for batch_idx, train_data in train_generator:
                    global_step = (epoch_idx - 1) * len(self.train_loader) + batch_idx
                    self.model.train()
                    I, Y = train_data
                    I = self.train_loader.dataset.load_item(I)
                    in_I, Y = self.to_device((I, Y))

                    # warm-up lr
                    if self.warmup_step != 0 and global_step <= self.warmup_step:
                        lr = self.config['config']['optimizer']['lr'] * float(global_step) / self.warmup_step
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr

                    self.optimizer.zero_grad()
                    
                    # Forward pass without autocast for MPS
                    Y_pre = self.model(in_I)

                    # for BCE Setting:
                    if self.num_classes == 1:
                        Y_pre = Y_pre.squeeze()
                        loss = self.loss_criterion(Y_pre, Y.float())
                        Y_pre = torch.sigmoid(Y_pre)
                    else:
                        loss = self.loss_criterion(Y_pre, Y)

                    # flood
                    loss = (loss - 0.04).abs() + 0.04
                    recons_loss = exp_recons_loss(self.model.loss_inputs['recons'], (in_I, Y))
                    contra_loss = self.contra_loss(self.model.loss_inputs['contra'], Y)
                    loss += self.lambda_1 * recons_loss + self.lambda_2 * contra_loss

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    self.acc_meter.update(Y_pre, Y, self.num_classes == 1)
                    self.loss_meter.update(loss.item())
                    self.recons_loss_meter.update(recons_loss.item())
                    self.contra_loss_meter.update(contra_loss.item())
                    iter_acc = self.acc_meter.mean_acc()

                    if global_step % self.log_steps == 0 and writer is not None:
                        writer.add_scalar("train/Acc", iter_acc, global_step)
                        writer.add_scalar("train/Loss", self.loss_meter.avg, global_step)
                        writer.add_scalar("train/Recons_Loss",
                                          self.recons_loss_meter.avg if self.lambda_1 != 0 else 0.,
                                          global_step)
                        writer.add_scalar("train/Contra_Loss",
                                          self.contra_loss_meter.avg if self.lambda_2 != 0 else 0.,
                                          global_step)
                        writer.add_scalar("train/LR",
                                          self.optimizer.param_groups[0]['lr'],
                                          global_step)

                    if global_step % self.val_steps == 0:
                        self.validate(epoch_idx, global_step, timer, writer)

                    train_generator.set_description(
                        "Epoch %d/%d, Step %d/%d, Loss %.4f, Acc %.4f" %
                        (epoch_idx, self.num_epoch, batch_idx, len(self.train_loader),
                         self.loss_meter.avg, iter_acc))

                # close the tqdm bar when one epoch ends
                train_generator.close()
                print()

            # training ends with integer epochs
            if writer is not None:
                writer.close()
            center_print("Training process ends.")
        except Exception as e:
            raise e

    def validate(self, epoch, step, timer, writer):
        v_idx = random.randint(1, len(self.val_loader) + 1)
        categories = self.val_loader.dataset.categories
        self.model.eval()
        with torch.no_grad():
            acc = AccMeter()
            auc = AUCMeter()
            loss_meter = AverageMeter()
            cur_acc = 0.0  # Higher is better
            cur_auc = 0.0  # Higher is better
            cur_loss = 1e8  # Lower is better
            val_generator = tqdm(enumerate(self.val_loader, 1), position=0, leave=True)

            for idx, val_data in val_generator:
                I, Y = val_data
                I = self.val_loader.dataset.load_item(I)
                in_I, Y = self.to_device((I, Y))
                Y_pre = self.model(in_I)

                # for BCE Setting:
                if self.num_classes == 1:
                    Y_pre = Y_pre.squeeze()
                    loss = self.loss_criterion(Y_pre, Y.float())
                    Y_pre = torch.sigmoid(Y_pre)
                else:
                    loss = self.loss_criterion(Y_pre, Y)

                acc.update(Y_pre, Y, self.num_classes == 1)
                auc.update(Y_pre, Y, self.num_classes == 1)
                loss_meter.update(loss.item())

                val_generator.set_description("Val %d/%d" % (idx, len(self.val_loader)))
                if idx == v_idx:
                    # show images
                    images = I[:4]
                    pred = Y_pre[:4]
                    gt = Y[:4]
                    self.plot_figure(images, pred, gt, 2, categories)

            cur_acc = acc.mean_acc()
            cur_auc = auc.mean_auc()
            cur_loss = loss_meter.avg

            if writer is not None:
                writer.add_scalar("val/Acc", cur_acc, step)
                writer.add_scalar("val/AUC", cur_auc, step)
                writer.add_scalar("val/Loss", cur_loss, step)

            print("Val, FINAL LOSS %.4f, FINAL ACC %.4f, FINAL AUC %.4f" %
                  (cur_loss, cur_acc, cur_auc))

            # save the best model
            if self.eval_metric == "Acc" and cur_acc > self.best_metric:
                self.best_metric = cur_acc
                self.best_step = step
                self._save_ckpt(step, best=True)
            elif self.eval_metric == "AUC" and cur_auc > self.best_metric:
                self.best_metric = cur_auc
                self.best_step = step
                self._save_ckpt(step, best=True)
            elif self.eval_metric == "LogLoss" and cur_loss < self.best_metric:
                self.best_metric = cur_loss
                self.best_step = step
                self._save_ckpt(step, best=True)

            # save the latest model
            self._save_ckpt(step, best=False)

            # close the tqdm bar
            val_generator.close()
            print() 