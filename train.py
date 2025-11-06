import os
import numpy as np
import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import EEGDataset
from src.data.datamodule import EEGDataModule
from src.models.cnn import EEG_CNN
from src.models.EEGPT import EEGPTClassifier
from src.training.trainer import Trainer


device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            name=f"{cfg.task}_{cfg.model.type}"
        )
    else:
        wandb.init(mode="disabled")

    preprocessing_config = OmegaConf.to_container(cfg.preprocessing, resolve=True)
    dataset = EEGDataset(
        data_dir=cfg.data.data_dir,
        task=cfg.task,
        preprocessing_config=preprocessing_config
    )

    datamodule = EEGDataModule(
        dataset=dataset,
        batch_size=cfg.data.batch_size,
        val_split=cfg.data.val_split,
        seed=cfg.seed
    )

    n_channels = dataset.trials.shape[0]
    n_timepoints = dataset.trials.shape[2]

    model_type = cfg.model.get('type', 'cnn').lower()

    if model_type == 'cnn':
        model = EEG_CNN(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=2,
            dropout=cfg.model.cnn.dropout
        )
    elif model_type == 'eegpt':
        electrode_names = dataset.electrode_names.tolist()

        eegpt_config = OmegaConf.to_container(cfg.model.eegpt, resolve=True)
        eegpt_config['img_size'] = [n_channels, n_timepoints]
        eegpt_config['in_channels'] = n_channels

        model = EEGPTClassifier(
            num_classes=2,
            use_channels_names=electrode_names,
            **eegpt_config
        )

        if eegpt_config.get('load_pretrained', False):
            checkpoint_path = eegpt_config.get('pretrained_checkpoint')
            if checkpoint_path and os.path.exists(checkpoint_path):
                pretrain_ckpt = torch.load(checkpoint_path)
                target_encoder_state = {}
                for k, v in pretrain_ckpt['state_dict'].items():
                    if k.startswith("target_encoder."):
                        target_encoder_state[k[15:]] = v 
                model.target_encoder.load_state_dict(target_encoder_state, strict=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'cnn' or 'eegpt'.")

    training_config = cfg.training
    OmegaConf.set_struct(training_config, False)
    training_config.task = cfg.task

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
