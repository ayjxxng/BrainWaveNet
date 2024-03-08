import wandb
import os
import numpy as np
import warnings
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import optim
from typing import List
import torch.utils as utils
from tqdm.auto import tqdm
from utils.dataloader import continus_mixup_data
from utils.lr_scheduler import Checkpoint, LRScheduler
from utils.metric import confusion

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Train:
    def __init__(self, args: DictConfig,
                 model: torch.nn.Module,
                 dataloaders: List[utils.data.DataLoader],
                 kk,
                 path,
                 ) -> None:
        self.args = args
        self.model = model 
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = args.train_epochs
        self.kk = kk
        self.path = path
        self.save_path = os.path.join(path, f'Fold_{self.kk+1}') 
        self.test_log_filepath = os.path.join(path, 'test_result.csv')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def train_per_epoch(self, optimizer, criterion, lr_scheduler):
        total_train, train_loss = 0., 0.
        train_pred, train_true = [], []

        self.model.train()
        for batch_x, batch_y in tqdm(self.train_dataloader):
            total_train += 1
            self.current_step += 1
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if self.args.mixup_data: # mixup
                batch_x, batch_y = continus_mixup_data(batch_x, y=batch_y)

            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y.float()) # CELoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pred.append(outputs.detach().cpu())
            train_true.append(batch_y.detach().cpu())
            if self.args.wandb:
                wandb.log({"LR": lr_scheduler.lr,
                        "Iter loss": loss.item()})
        train_loss /= total_train
        train_acc, _, _, _, _, _, _ = confusion(train_true, train_pred)
        return train_loss, train_acc


    def vali_per_epoch(self, dataloader, criterion):
        total_vali, vali_loss = 0., 0.
        vali_pred, vali_true = [], []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                total_vali += 1
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y.float())

                vali_loss += loss.item()
                vali_pred.append(outputs.detach().cpu())
                vali_true.append(batch_y.detach().cpu())
        vali_loss /= total_vali
        vali_acc, vali_sens, vali_spef, vali_auc, vali_f1, vali_recall, vali_precision = confusion(vali_true, vali_pred)
        return vali_loss, vali_acc, vali_sens, vali_spef, vali_auc, vali_f1, vali_recall, vali_precision


    def test(self, dataloader, criterion):
        print('Loading best model...')
        save_model_path = self.save_path + '/final_auc_model.pt'
        checkpoint = torch.load(save_model_path)
        self.model.load_state_dict(checkpoint['model'])
        total_test, test_loss = 0., 0.
        test_pred, test_true = [], []

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                total_test += 1
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y.float())
                test_loss += loss.item()

                test_pred.append(outputs.detach().cpu())
                test_true.append(batch_y.detach().cpu())
        test_loss /= total_test
        test_acc, test_sens, test_spef, test_auc, test_f1, test_recall, test_precision = confusion(test_true, test_pred)
        final_result = test_loss, test_acc, test_sens, test_spef, test_auc, test_f1, test_recall, test_precision
        print("Fold_{} | Final Loss: {:.5f}, Final Accuracy: {:.5f}, Final Sens: {:.5f}, Final Spef: {:.5f}, Final AUC: {:.5f}, Final F1: {:.5f}, ".format(
            self.kk+1, final_result[0], final_result[1], final_result[2], final_result[3], final_result[4], final_result[5]))
        with open(self.test_log_filepath, 'a') as f:
            f.write('{},{},{},{},{},{},{},{},{}\n'.format(
                    self.kk+1, final_result[0], final_result[1], final_result[2], final_result[3], final_result[4],
                    final_result[5], final_result[6], final_result[7]))
        if self.args.wandb:
            wandb.log({
                "Final Loss": final_result[0],
                "Final Accuracy": final_result[1],
                "Final Sens": final_result[2],
                "Final Spef": final_result[3],
                "Final AUC": final_result[4],
                "Final F1": final_result[5],
                "Final recall": final_result[6],
                "Final presicion": final_result[7],
            })


    def train(self):
        training_process = []
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.optimizer.lr, weight_decay=self.args.weight_decay)
        criterion = nn.CrossEntropyLoss(reduction="sum") # CELoss
        lr_schedulers = LRScheduler(cfg=self.args, optimizer_cfg=self.args.optimizer)
        check_epoch = Checkpoint(patience=self.args.patience, verbose=True)

        self.current_step = 0

        for epoch in range(self.epochs):
            print(f"Epoch[{epoch+1}/{self.args.train_epochs}] ========================")
            train_loss, train_acc = self.train_per_epoch(optimizer, criterion, lr_schedulers)

            val_result = self.vali_per_epoch(self.val_dataloader, criterion)
            test_result = self.vali_per_epoch(self.test_dataloader, criterion)

            print("| Train Loss: {:.5f} Train Acc: {:.5f} Val Loss: {:.5f} Val Acc: {:.5f} Val AUC: {:.5f} Test Loss: {:.5f} Test Acc: {:.5f} Test AUC: {:.5f}".format(
                train_loss, train_acc, val_result[0], val_result[1], val_result[4], test_result[0], test_result[1], test_result[4]))
            check_epoch(val_result[4], val_result[1], epoch, optimizer, lr_schedulers, train_loss, self.model, self.save_path)
            if self.args.wandb:
                wandb.log({
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Val Loss": val_result[0],
                    "Val Accuracy": val_result[1],
                    "Test Loss": test_result[0],
                    "Test Accuracy": test_result[1],
                    "Val AUC": val_result[4],
                    "Test AUC": test_result[4],
                    'Test Sensitivity': test_result[2],
                    'Test Specificity': test_result[3],
                    'Test F1': test_result[5],
                })
            training_process.append({
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Train Accuracy": train_acc,
                "Val Loss": val_result[0],
                "Val Accuracy": val_result[1],
                "Test Loss": test_result[0],
                "Test Accuracy": test_result[1],
                "Val AUC": val_result[4],
                "Test AUC": test_result[4],
                'Test Sensitivity': test_result[2],
                'Test Specificity': test_result[3],
                'Test F1': test_result[5],
                'Test recall': test_result[6],
                'Test precision': test_result[7],
            })
        train_model_path = os.path.join(self.save_path, 'train_model.pt')
        torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr': lr_schedulers.lr,
                    'loss': train_loss,
                    }, train_model_path)
        if self.args.test:
            self.test(self.test_dataloader, criterion)

        np.save(os.path.join(self.save_path, 'training_process.npy'), training_process, allow_pickle=True)
