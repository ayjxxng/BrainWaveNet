import os
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from typing import Optional, List, Tuple


def init_setting(cfg: DictConfig) -> None:
    """Initialize the environment settings."""
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Logger:
    def __init__(self, cfg: DictConfig, verbose: bool = True) -> None:
        self.cfg = cfg
        self.unique_id = cfg.unique_id
        self.path = cfg.path
        self.log_filepath = os.path.join(self.path, 'test_result.csv')
        self.verbose = verbose
        self.wandb = self.cfg.wandb

    def init_logging(self) -> None:
        header = 'Fold,Accuracy,Sensitivity,Specificity,AUC,F1,Recall,Precision\n'
        with open(self.log_filepath, 'a') as f:
            f.write(header)

    def log_results(
            self,
            fold: Optional[int] = None,
            results: Optional[Tuple[float, ...]] = None,
            all_results: Optional[List[float]] = None
    ) -> None:
        if all_results is not None:
            mean_result = np.mean(all_results, axis=0)
            std_result = np.std(all_results, axis=0)
            with open(self.log_filepath, 'a') as f:
                if mean_result is not None:
                    f.write(f'Avg.,{mean_result[0]},{mean_result[1]},{mean_result[2]},{mean_result[3]},'
                            f'{mean_result[4]},{mean_result[5]},{mean_result[6]}\n')
                    f.write(f'STD,{std_result[0]},{std_result[1]},{std_result[2]},{std_result[3]},'
                            f'{std_result[4]},{std_result[5]},{std_result[6]}\n')
            if self.wandb:
                wandb.init(
                    project=self.cfg.project,
                    group=self.unique_id,
                    name=f"{self.unique_id}:Avg_Results",
                    config=OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True),
                    reinit=True,
                )
                wandb.log({
                    "Avg. ACC": mean_result[0],
                    "Avg. SEN": mean_result[1],
                    "Avg. SPC": mean_result[2],
                    "Avg. AUC": mean_result[3],
                    "Avg. F1": mean_result[4],
                    "Avg. recall": mean_result[5],
                    "Avg. precision": mean_result[6],
                })
                wandb.finish()
            if self.verbose:
                print(f"Avg. ACC: {mean_result[0]:.5f}, Avg. AUC: {mean_result[3]:.5f} "
                      f"Avg. SEN: {mean_result[1]:.5f}, Avg. SPC: {mean_result[2]:.5f} ")

        else:
            with open(self.log_filepath, 'a') as f:
                f.write(f'{fold},{results[0]},{results[1]},{results[2]},{results[3]},'
                        f'{results[4]},{results[5]},{results[6]}\n')
            if self.wandb:
                wandb.log({
                    "Final ACC": results[0],
                    "Final SEN": results[1],
                    "Final SPC": results[2],
                    "Final AUC": results[3],
                    "Final F1": results[4],
                    "Final recall": results[5],
                    "Final precision": results[6],
                })
            if self.verbose:
                print(f"Fold_{fold} | Final ACC: {results[0]:.5f}, Final AUC: {results[3]:.5f} "
                      f"Final SEN: {results[1]:.5f}, Final SPC: {results[2]:.5f} ")
              
