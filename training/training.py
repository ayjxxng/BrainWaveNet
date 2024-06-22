import os
import warnings
import pickle
from typing import List

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from omegaconf import DictConfig

from models.bwn import BrainWaveNet
from utils.dataloader import continuous_mixup_data
from utils.lr_scheduler import Checkpoint, LRScheduler
from utils.metric import confusion
from utils.utils import Logger

warnings.filterwarnings('ignore')
device = torch.device("cuda")


class Train:
    def __init__(
            self,
            cfg: DictConfig,
            model: nn.Module,
            dataloaders: List[DataLoader],
            fold: int,
            logger: Logger,
    ) -> None:
        self.current_step = 0
        self.cfg = cfg
        self.model = model
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.train_epochs
        self.fold = fold
        self.logger = logger
        self.path = cfg.path
        self.save_path = os.path.join(self.path, f'Fold_{self.fold + 1}')
        os.makedirs(self.save_path, exist_ok=True)

    def train_per_epoch(self, optimizer, criterion, lr_scheduler):
        total_train, train_loss = 0., 0.
        train_pred, train_true = [], []

        self.model.train()
        for batch_x, batch_y in tqdm(self.train_dataloader, leave=True):
            total_train += 1
            self.current_step += 1
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            if self.cfg.training.mixup_data:
                batch_x, batch_y = continuous_mixup_data(batch_x, y=batch_y)

            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pred.append(outputs.detach().cpu())
            train_true.append(batch_y.detach().cpu())

            if self.cfg.wandb:
                wandb.log({"LR": lr_scheduler.lr, "Iter loss": loss.item()})

        train_loss /= total_train
        train_acc, _, _, _, _, _, _ = confusion(train_true, train_pred)
        return train_loss, train_acc

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total, loss = 0., 0.
        pred, true = [], []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                total += 1
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = self.model(batch_x)
                loss += criterion(outputs, batch_y.float()).item()
                pred.append(outputs.detach().cpu())
                true.append(batch_y.detach().cpu())

        loss /= total
        final_result = confusion(true, pred)
        return loss, final_result

    def test(self, dataloader, criterion):
        print('Loading best model...')
        checkpoint = torch.load(os.path.join(self.save_path, 'final_auc_model.pt'))
        self.model.load_state_dict(checkpoint['model'])
        _, final_result = self.evaluate(dataloader, criterion)
        self.logger.log_results(fold=self.fold + 1, results=final_result)

        return final_result

    def train(self):
        self.current_step = 0
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.lr,
                               weight_decay=self.cfg.optimizer.weight_decay)
        criterion = nn.CrossEntropyLoss(reduction="sum").to(device)
        lr_schedulers = LRScheduler(cfg=self.cfg, optimizer_cfg=self.cfg.optimizer)
        check_epoch = Checkpoint(patience=self.cfg.training.patience)

        training_process = []
        for epoch in range(self.epochs):
            print(f"Epoch[{epoch + 1}/{self.cfg.training.train_epochs}] ========================")
            train_loss, train_acc = self.train_per_epoch(optimizer, criterion, lr_schedulers)
            vali_loss, vali_result = self.evaluate(self.val_dataloader, criterion)
            test_loss, test_result = self.evaluate(self.test_dataloader, criterion)

            print(f"| Train Loss: {train_loss:.5f} Train ACC: {train_acc:.5f} "
                  f"Val Loss: {vali_loss:.5f} Val ACC: {vali_result[0]:.5f} Val AUC: {vali_result[3]:.5f} "
                  f"Test Loss: {test_loss:.5f} Test ACC: {test_result[0]:.5f} Test AUC: {test_result[3]:.5f}")

            check_epoch(vali_result[3], epoch, optimizer, lr_schedulers, train_loss, self.model, self.save_path)
            if self.cfg.wandb:
                wandb.log({
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Val Loss": vali_loss,
                    "Val Accuracy": vali_result[0],
                    "Test Loss": test_loss,
                    "Test Accuracy": test_result[0],
                    "Val AUC": vali_result[3],
                    "Test AUC": test_result[3],
                })
            training_process.append({
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Train Accuracy": train_acc,
                "Val Loss": vali_loss,
                "Val Accuracy": vali_result[0],
                "Test Loss": test_loss,
                "Test Accuracy": test_result[0],
                "Val AUC": vali_result[3],
                "Test AUC": test_result[3],
            })
            if check_epoch.early_stop:
                print("Early stopping")
                break
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr_schedulers.lr,
        }, os.path.join(self.save_path, 'train_model.pt'))
        pickle.dump(training_process, open(os.path.join(self.save_path, 'training_process.pkl'), 'wb'))

        # Test Model
        final_result = self.test(self.test_dataloader, criterion)

        return final_result


def build_model(cfg: DictConfig) -> nn.Module:
    model = BrainWaveNet(cfg)
    model = model.to(device)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    if cfg.wandb:
        wandb.config.update({"total_params": total_params})
    return model
  
