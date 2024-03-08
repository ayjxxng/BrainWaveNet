import os
import numpy as np
import torch
from omegaconf import DictConfig
import bisect
import math


class LRScheduler:
    def __init__(self, cfg: DictConfig,  optimizer_cfg: DictConfig):
        self.lr_config = optimizer_cfg.lr_scheduler
        self.training_config = cfg
        self.lr = optimizer_cfg.lr

        assert self.lr_config.mode in [
            'step', 'poly', 'cos', 'linear', 'decay']

    def update(self, optimizer: torch.optim.Optimizer, step: int):
        lr_config = self.lr_config
        lr_mode = lr_config.mode
        base_lr = lr_config.base_lr
        target_lr = lr_config.target_lr

        warm_up_from = lr_config.warm_up_from
        warm_up_steps = lr_config.warm_up_steps
        total_steps = self.training_config.total_steps

        assert 0 <= step <= total_steps, f"step:{step}, total_step:{total_steps}"
        if step < warm_up_steps:
            current_ratio = step / warm_up_steps
            self.lr = warm_up_from + (base_lr - warm_up_from) * current_ratio
        else:
            current_ratio = (step - warm_up_steps) / \
                (total_steps - warm_up_steps)
            if lr_mode == 'step':
                count = bisect.bisect_left(lr_config.milestones, current_ratio)
                self.lr = base_lr * pow(lr_config.decay_factor, count)
            elif lr_mode == 'poly':
                poly = pow(1 - current_ratio, lr_config.poly_power)
                self.lr = target_lr + (base_lr - target_lr) * poly
            elif lr_mode == 'cos':
                cosine = math.cos(math.pi * current_ratio)
                self.lr = target_lr + (base_lr - target_lr) * (1 + cosine) / 2
            elif lr_mode == 'linear':
                self.lr = target_lr + \
                    (base_lr - target_lr) * (1 - current_ratio)
            elif lr_mode == 'decay':
                epoch = step // self.training_config.steps_per_epoch
                self.lr = base_lr * lr_config.lr_decay ** epoch

        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr


class Checkpoint:
    def __init__(self, verbose=False, delta=0):
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = -np.Inf
        self.delta = delta

    def __call__(self, val_auc, epoch, optimizer, lr_schedulers, train_loss, model, path):
        score = val_auc
        path_auc = os.path.join(path, 'final_auc_model.pt')
        self.epoch = epoch
        self.optimizer = optimizer
        self.lr = lr_schedulers.lr
        self.train_loss = train_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model, path_auc)
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_auc, model, path_auc)
        else:
            pass
    def save_checkpoint(self, val_auc, model, path):
        if self.verbose:
            print(f'Validation AUC increased ({self.val_auc_max:.5f} --> {val_auc:.5f}).  Saving model ...')
        torch.save({
                    'epoch': self.epoch,
                    'model': model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr': self.lr,
                    'loss': self.train_loss,
                    }, path)
        self.val_auc_max = val_auc
