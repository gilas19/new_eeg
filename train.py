import os
import numpy as np
import torch
import wandb
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import EEGDataset
from src.data.datamodule import EEGDataModule
from src.models.cnn import EEG_CNN
from src.models.EEGPT import EEGPTClassifier
from src.training.trainer import Trainer


device = 'cuda' if torch.cuda.is_available() else 'cpu'
log = logging.getLogger(__name__)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(cfg, n_channels, n_timepoints, electrode_names):
    """Create model based on configuration."""
    model_type = cfg.model.get('type', 'cnn').lower()

    if model_type == 'cnn':
        model = EEG_CNN(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=2,
            dropout=cfg.model.cnn.dropout
        )
    elif model_type == 'eegpt':
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

    return model


def run_single_fold(cfg, dataset, fold_idx=None, n_folds=None):
    """Run training for a single fold or single split."""
    datamodule = EEGDataModule(
        dataset=dataset,
        batch_size=cfg.data.batch_size,
        val_split=cfg.data.get('val_split', 0.1),
        test_split=cfg.data.get('test_split', 0.1),
        seed=cfg.seed,
        n_folds=n_folds,
        current_fold=fold_idx
    )

    split_info = datamodule.get_split_info()
    wandb.log({
        'split/n_train': split_info['n_train'],
        'split/n_val': split_info['n_val'],
        'split/n_test': split_info['n_test'],
        'split/train_class_0': split_info['train_class_0'],
        'split/train_class_1': split_info['train_class_1'],
        'split/val_class_0': split_info['val_class_0'],
        'split/val_class_1': split_info['val_class_1'],
        'split/test_class_0': split_info['test_class_0'],
        'split/test_class_1': split_info['test_class_1'],
    })

    n_channels = dataset.trials.shape[0]
    n_timepoints = dataset.trials.shape[2]
    electrode_names = dataset.electrode_names.tolist()
    model = create_model(cfg, n_channels, n_timepoints, electrode_names)

    training_config = cfg.training
    OmegaConf.set_struct(training_config, False)
    training_config.task = cfg.task

    trainer = Trainer(
        model=model,
        datamodule=datamodule,
        config=training_config,
        device=device
    )

    best_val_acc = trainer.run()

    test_loss, test_metrics = trainer.test()

    log_dict = {
        'test/loss': test_loss,
        'test/accuracy': test_metrics['accuracy'],
        'test/precision': test_metrics['precision'],
        'test/recall': test_metrics['recall'],
        'test/f1': test_metrics['f1_score'],
        'final/best_val_accuracy': best_val_acc,
    }
    if fold_idx is not None:
        log_dict['fold'] = fold_idx
    wandb.log(log_dict)

    return {
        'val_accuracy': best_val_acc,
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1_score'],
        'test_loss': test_loss
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    os.makedirs("checkpoints", exist_ok=True)

    log.info(f"Loading EEG dataset from: {cfg.data.data_dir}")
    log.info(f"Task: {cfg.task}")
    log.info(f"Device: {device}")

    preprocessing_config = OmegaConf.to_container(cfg.preprocessing, resolve=True)
    dataset = EEGDataset(
        data_dir=cfg.data.data_dir,
        task=cfg.task,
        preprocessing_config=preprocessing_config
    )

    log.info(f"Dataset loaded - Shape: {dataset.trials.shape} (channels, trials, timepoints)")
    log.info(f"Number of channels: {dataset.trials.shape[0]}")
    log.info(f"Number of trials: {dataset.trials.shape[1]}")
    log.info(f"Number of timepoints: {dataset.trials.shape[2]}")
    log.info(f"Class distribution - Class 0: {(dataset.labels == 0).sum()}, Class 1: {(dataset.labels == 1).sum()}")

    n_folds = cfg.data.get('n_folds', None)
    use_cv = n_folds is not None and n_folds > 1

    if use_cv:
        all_results = []

        for fold_idx in range(n_folds):
            if cfg.wandb.enabled:
                wandb.init(
                    project=cfg.wandb.project,
                    entity=cfg.wandb.entity,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    mode=cfg.wandb.mode,
                    name=f"{cfg.task}_{cfg.model.type}_fold{fold_idx+1}",
                    group=f"{cfg.task}_{cfg.model.type}_cv",
                    reinit=True
                )
            else:
                wandb.init(mode="disabled", reinit=True)

            results = run_single_fold(cfg, dataset, fold_idx=fold_idx, n_folds=n_folds)
            all_results.append(results)

            wandb.finish()

        avg_val_acc = np.mean([r['val_accuracy'] for r in all_results])
        std_val_acc = np.std([r['val_accuracy'] for r in all_results])
        avg_test_acc = np.mean([r['test_accuracy'] for r in all_results])
        std_test_acc = np.std([r['test_accuracy'] for r in all_results])
        avg_test_precision = np.mean([r['test_precision'] for r in all_results])
        std_test_precision = np.std([r['test_precision'] for r in all_results])
        avg_test_recall = np.mean([r['test_recall'] for r in all_results])
        std_test_recall = np.std([r['test_recall'] for r in all_results])
        avg_test_f1 = np.mean([r['test_f1'] for r in all_results])
        std_test_f1 = np.std([r['test_f1'] for r in all_results])
        avg_test_loss = np.mean([r['test_loss'] for r in all_results])
        std_test_loss = np.std([r['test_loss'] for r in all_results])

        log.info(f"\n{'='*60}")
        log.info(f"Cross-Validation Summary ({n_folds} folds)")
        log.info(f"{'='*60}")
        log.info(f"Validation Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}")
        log.info(f"Test Accuracy:       {avg_test_acc:.4f} ± {std_test_acc:.4f}")
        log.info(f"Test Precision:      {avg_test_precision:.4f} ± {std_test_precision:.4f}")
        log.info(f"Test Recall:         {avg_test_recall:.4f} ± {std_test_recall:.4f}")
        log.info(f"Test F1 Score:       {avg_test_f1:.4f} ± {std_test_f1:.4f}")
        log.info(f"Test Loss:           {avg_test_loss:.4f} ± {std_test_loss:.4f}")
        log.info(f"{'='*60}\n")

        if cfg.wandb.enabled:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=cfg.wandb.mode,
                name=f"{cfg.task}_{cfg.model.type}_cv_summary",
                reinit=True
            )
            wandb.log({
                'cv_summary/n_folds': n_folds,
                'cv_summary/val_accuracy_mean': avg_val_acc,
                'cv_summary/val_accuracy_std': std_val_acc,
                'cv_summary/test_accuracy_mean': avg_test_acc,
                'cv_summary/test_accuracy_std': std_test_acc,
                'cv_summary/test_precision_mean': avg_test_precision,
                'cv_summary/test_precision_std': std_test_precision,
                'cv_summary/test_recall_mean': avg_test_recall,
                'cv_summary/test_recall_std': std_test_recall,
                'cv_summary/test_f1_mean': avg_test_f1,
                'cv_summary/test_f1_std': std_test_f1,
                'cv_summary/test_loss_mean': avg_test_loss,
                'cv_summary/test_loss_std': std_test_loss,
            })
            wandb.finish()

    else:
        log.info("Starting single train/val/test split")

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

        results = run_single_fold(cfg, dataset)

        log.info(f"\n{'='*60}")
        log.info(f"Training Summary")
        log.info(f"{'='*60}")
        log.info(f"Best Validation Accuracy: {results['val_accuracy']:.4f}")
        log.info(f"Test Accuracy:            {results['test_accuracy']:.4f}")
        log.info(f"Test Precision:           {results['test_precision']:.4f}")
        log.info(f"Test Recall:              {results['test_recall']:.4f}")
        log.info(f"Test F1 Score:            {results['test_f1']:.4f}")
        log.info(f"Test Loss:                {results['test_loss']:.4f}")
        log.info(f"{'='*60}\n")

        wandb.finish()


if __name__ == "__main__":
    main()
