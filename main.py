import os
import random
from datetime import datetime
import torch
import numpy as np
from omegaconf import open_dict
from sklearn.model_selection import StratifiedShuffleSplit
import wandb
from training import Train

from utils.dataloader import get_dataloader
from models.brainwavenet import BrainWaveNet
from config import get_args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


fix_seed = 42
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args = get_args()

with open_dict(args):
    args.unique_id = datetime.now().strftime("%d-%H%M")

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}-S{}_{}-E{}-T{}_{}'.format(args.unique_id,
                                            args.temporal_dmodel,
                                            args.temporal_nheads,
                                            args.embed_dim,
                                            args.spatial_dmodel,
                                            args.spatial_nheads)
        path = os.path.join(args.checkpoints, setting) 
        if not os.path.exists(path):
            os.makedirs(path)
        test_log_filepath = os.path.join(path, 'test_result.csv')
        with open(test_log_filepath, 'a') as f:
            f.write('Fold,Loss,Accuracy,Sensitivity,Specificity,AUC,F1,Recall,Precision\n')

        split2 = StratifiedShuffleSplit(n_splits=args.n_fold, test_size=args.val_length)
        for kk, (train_index, val_index) in enumerate(split2.split(train_val_data, train_val_keys)):
            if args.wandb:
                wandb.init(project=args.project,
                        group=args.unique_id,
                            name=f"F{kk+1}",
                            config={
                                "unique_id": args.unique_id,
                                "epochs": args.train_epochs,
                                "batch_size": args.batch_size,
                                "n_fold": args.n_fold,

                                "front_end": args.front_end,
                                "n_channels": args.n_channels,
                                "n_frequencies": args.n_frequencies,
                                "n_times": args.n_times,
                                "pooling": args.pooling,
                                "stride": args.stride,

                                "embed_dim": args.embed_dim,
                                "spectral_dmodel": args.spectral_dmodel,
                                "spectral_nheads": args.spectral_nheads,
                                "spectral_dimff": args.spectral_dimff,

                                "temporal_dmodel": args.temporal_dmodel,
                                "temporal_nheads": args.temporal_nheads,
                                "temporal_dimff": args.temporal_dimff,

                                "dropout": args.dropout,
                                "n_blocks": args.n_blocks,

                                "weight_decay": args.weight_decay,
                                "learning_rate": args.optimizer.lr,
                                "target_lr": args.optimizer.lr_scheduler.target_lr,
                                "weight_decay": args.weight_decay,
                                },
                        reinit=True,
                        tags=[args.unique_id,
                            f"emb_d{args.embed_dim}",
                            f"spe_d{args.spectral_dmodel}",
                            f"tem_d{args.temporal_dmodel}"])
            print(f"<<<<<<<<<<<<< Fold[{kk+1}/{args.n_fold}] >>>>>>>>>>>>>")
            train_loader= get_dataloader(args, train_val_data[train_index], train_val_labs[train_index], shuffle=True)
            val_loader= get_dataloader(args, train_val_data[val_index], train_val_labs[val_index], shuffle=False)
            test_loader = get_dataloader(args, test_data, test_labs, shuffle=False)
            with open_dict(args):
                args.steps_per_epoch = len(train_loader)
                args.total_steps = args.steps_per_epoch * args.train_epochs
            dataloaders = [train_loader, val_loader, test_loader]

            model = BrainWaveNet(args)
            model = model.to(device)
            total_params = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad: continue
                param = parameter.numel()
                total_params += param
            print(f"Total Trainable Params: {total_params}")

            if args.wandb:
                wandb.config.update({"total_params": total_params})

            train_model = Train(args, model, dataloaders, kk, path)
            train_model.train()
            if args.wandb:
                wandb.finish()
                
torch.cuda.empty_cache()
