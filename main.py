import os
import pickle
import warnings
from datetime import datetime
import torch
import wandb
import hydra
from omegaconf import DictConfig, open_dict, omegaconf
from sklearn.model_selection import StratifiedShuffleSplit
import gc

from utils.dataloader import init_dataloader
from utils.utils import init_setting, Logger
from training.training import build_model, Train

warnings.filterwarnings('ignore')
wandb.require("core")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Initialize settings
    unique_id = datetime.now().strftime("%m-%d-%H-%M")
    init_setting(cfg)

    # Load datasets
    train_valid_data, train_valid_labs, train_valid_keys = pickle.load(open(cfg.dataset.train_valid_data, 'rb'))
    test_data, test_labs, _ = pickle.load(open(cfg.dataset.test_data, 'rb'))

    length = len(train_valid_data) + len(test_data)
    val_length = int(length * cfg.dataset.valid_set)
    with open_dict(cfg):
        cfg.val_length = val_length

    # setting record of experiments
    setting = 'B{}T{}S{}_E{}_T{}H{}F{}_S{}H{}F{}_E{}B{}_D{}_LR{}_D{}'.format(cfg.model.n_blocks,
                                                                             cfg.model.n_temporal_blocks,
                                                                             cfg.model.n_spatial_blocks,
                                                                             cfg.model.embed_dim,
                                                                             cfg.model.temporal_dmodel,
                                                                             cfg.model.temporal_nheads,
                                                                             cfg.model.temporal_dim_factor,
                                                                             cfg.model.spatial_dmodel,
                                                                             cfg.model.spatial_nheads,
                                                                             cfg.model.spatial_dim_factor,
                                                                             cfg.training.train_epochs,
                                                                             cfg.training.batch_size,
                                                                             cfg.model.dropout,
                                                                             cfg.optimizer.lr,
                                                                             cfg.optimizer.weight_decay
                                                                             )
    unique_id = '{}_{}'.format(unique_id, setting)
    path = os.path.join(cfg.checkpoints, unique_id)
    with open_dict(cfg):
        cfg.unique_id = unique_id
        cfg.path = path
    os.makedirs(path, exist_ok=True)

    logger = Logger(cfg)
    logger.init_logging()

    all_results = []
    split = StratifiedShuffleSplit(n_splits=cfg.n_fold, test_size=cfg.val_length, random_state=cfg.seed)
    for fold, (train_index, valid_index) in enumerate(split.split(train_valid_data, train_valid_keys)):
        init_setting(cfg)
        if cfg.wandb:
            wandb.init(project=cfg.project,
                       group=unique_id,
                       name=f"{unique_id}:F{fold + 1}",
                       config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                       reinit=True,
                       )
        print(f"<<<<<<<<<<<<< Fold[{fold + 1}/{cfg.n_fold}] >>>>>>>>>>>>>")
        dataloaders = init_dataloader(
            cfg=cfg,
            train_idx=train_index,
            valid_idx=valid_index,
            train_valid_samples=train_valid_data,
            train_valid_labels=train_valid_labs,
            test_samples=test_data,
            test_labels=test_labs
        )
        model = build_model(cfg)
        train_model = Train(
            cfg=cfg,
            model=model,
            dataloaders=dataloaders,
            fold=fold,
            logger=logger
        )
        final_result = train_model.train()
        all_results.append(final_result)
        
        if cfg.wandb:
            wandb.finish()

        gc.collect()
        torch.cuda.empty_cache()

    logger.log_results(all_results=all_results)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
    
