import os
import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import EEGDataset
from src.data.datamodule import EEGDataModule
from src.models.cnn import EEG_CNN
from src.training.trainer import Trainer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    os.makedirs("checkpoints", exist_ok=True)

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            name=f"{cfg.task}_{cfg.model.name}"
        )
    else:
        wandb.init(mode="disabled")

    dataset = EEGDataset(data_dir=cfg.data.data_dir, task=cfg.task)

    datamodule = EEGDataModule(
        dataset=dataset,
        batch_size=cfg.data.batch_size,
        val_split=cfg.data.val_split,
        seed=cfg.seed
    )

    n_channels = dataset.trials.shape[0]
    n_timepoints = dataset.trials.shape[2]

    model = EEG_CNN(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=2,
        dropout=cfg.model.dropout
    )

    device = cfg.device if torch.cuda.is_available() else 'cpu'

    class TrainingConfig:
        def __init__(self, cfg):
            self.task = cfg.task
            self.epochs = cfg.training.epochs
            self.patience = cfg.training.patience
            self.learning_rate = cfg.training.learning_rate
            self.weight_decay = cfg.training.weight_decay

    training_config = TrainingConfig(cfg)

    trainer = Trainer(
        model=model,
        datamodule=datamodule,
        config=training_config,
        device=device
    )

    trainer.run()

    wandb.finish()


if __name__ == "__main__":
    main()
