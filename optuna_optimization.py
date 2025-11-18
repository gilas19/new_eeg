import os
import argparse
import yaml
import numpy as np
import torch
import wandb
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import EEGDataset
from src.data.datamodule import EEGDataModule
from src.models.cnn import EEG_CNN
from src.models.EEGPT import EEGPTClassifier
from src.training.trainer import Trainer


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(cfg, n_channels, n_timepoints, electrode_names, trial_params):
    """Create model with trial hyperparameters."""
    model_type = cfg['model']['type'].lower()

    if model_type == 'cnn':
        model = EEG_CNN(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=2,
            dropout=trial_params['dropout']
        )
    elif model_type == 'eegpt':
        eegpt_config = cfg['model']['eegpt'].copy()
        eegpt_config['img_size'] = [n_channels, n_timepoints]
        eegpt_config['in_channels'] = n_channels

        # Override with trial parameters
        eegpt_config['enc_drop_rate'] = trial_params.get('enc_drop_rate', eegpt_config['enc_drop_rate'])
        eegpt_config['enc_attn_drop_rate'] = trial_params.get('enc_attn_drop_rate', eegpt_config['enc_attn_drop_rate'])
        eegpt_config['enc_drop_path_rate'] = trial_params.get('enc_drop_path_rate', eegpt_config['enc_drop_path_rate'])
        eegpt_config['rec_drop_rate'] = trial_params.get('rec_drop_rate', eegpt_config['rec_drop_rate'])
        eegpt_config['rec_attn_drop_rate'] = trial_params.get('rec_attn_drop_rate', eegpt_config['rec_attn_drop_rate'])
        eegpt_config['rec_drop_path_rate'] = trial_params.get('rec_drop_path_rate', eegpt_config['rec_drop_path_rate'])

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


def objective(trial, cfg, dataset, optimization_config):
    """
    Optuna objective function to optimize.

    Args:
        trial: Optuna trial object
        cfg: Configuration dictionary
        dataset: EEG dataset
        optimization_config: Optimization-specific configuration

    Returns:
        float: Validation accuracy (or other metric to optimize)
    """
    # Suggest hyperparameters based on optimization config
    trial_params = {}

    if 'learning_rate' in optimization_config['search_space']:
        lr_config = optimization_config['search_space']['learning_rate']
        trial_params['learning_rate'] = trial.suggest_float(
            'learning_rate',
            lr_config['min'],
            lr_config['max'],
            log=lr_config.get('log', True)
        )
    else:
        trial_params['learning_rate'] = cfg['training']['learning_rate']

    if 'weight_decay' in optimization_config['search_space']:
        wd_config = optimization_config['search_space']['weight_decay']
        trial_params['weight_decay'] = trial.suggest_float(
            'weight_decay',
            wd_config['min'],
            wd_config['max'],
            log=wd_config.get('log', True)
        )
    else:
        trial_params['weight_decay'] = cfg['training'].get('weight_decay', 0.0)

    if 'batch_size' in optimization_config['search_space']:
        bs_config = optimization_config['search_space']['batch_size']
        trial_params['batch_size'] = trial.suggest_categorical(
            'batch_size',
            bs_config['choices']
        )
    else:
        trial_params['batch_size'] = cfg['data']['batch_size']

    if 'dropout' in optimization_config['search_space']:
        dropout_config = optimization_config['search_space']['dropout']
        trial_params['dropout'] = trial.suggest_float(
            'dropout',
            dropout_config['min'],
            dropout_config['max']
        )
    else:
        trial_params['dropout'] = cfg['model'].get('cnn', {}).get('dropout', 0.3)

    if cfg['model']['type'].lower() == 'eegpt':
        for drop_param in ['enc_drop_rate', 'enc_attn_drop_rate', 'enc_drop_path_rate',
                          'rec_drop_rate', 'rec_attn_drop_rate', 'rec_drop_path_rate']:
            if drop_param in optimization_config['search_space']:
                drop_config = optimization_config['search_space'][drop_param]
                trial_params[drop_param] = trial.suggest_float(
                    drop_param,
                    drop_config['min'],
                    drop_config['max']
                )

    if cfg['training']['scheduler']['enabled']:
        scheduler_type = cfg['training']['scheduler']['type']

        if scheduler_type == 'CosineAnnealingLR' and 'eta_min' in optimization_config['search_space']:
            eta_min_config = optimization_config['search_space']['eta_min']
            trial_params['eta_min'] = trial.suggest_float(
                'eta_min',
                eta_min_config['min'],
                eta_min_config['max'],
                log=eta_min_config.get('log', True)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            if 'scheduler_factor' in optimization_config['search_space']:
                factor_config = optimization_config['search_space']['scheduler_factor']
                trial_params['scheduler_factor'] = trial.suggest_float(
                    'scheduler_factor',
                    factor_config['min'],
                    factor_config['max']
                )
            if 'scheduler_patience' in optimization_config['search_space']:
                patience_config = optimization_config['search_space']['scheduler_patience']
                trial_params['scheduler_patience'] = trial.suggest_int(
                    'scheduler_patience',
                    patience_config['min'],
                    patience_config['max']
                )

    set_seed(cfg['seed'])

    datamodule = EEGDataModule(
        dataset=dataset,
        batch_size=trial_params['batch_size'],
        val_split=cfg['data'].get('val_split', 0.1),
        test_split=cfg['data'].get('test_split', 0.1),
        seed=cfg['seed'],
        n_folds=None,
        current_fold=None
    )

    n_channels = dataset.trials.shape[0]
    n_timepoints = dataset.trials.shape[2]
    electrode_names = dataset.electrode_names.tolist()
    model = create_model(cfg, n_channels, n_timepoints, electrode_names, trial_params)

    training_config = OmegaConf.create(cfg['training'])
    OmegaConf.set_struct(training_config, False)
    training_config.task = cfg['task']
    training_config.learning_rate = trial_params['learning_rate']
    training_config.weight_decay = trial_params['weight_decay']

    if training_config.scheduler.enabled:
        if 'eta_min' in trial_params:
            training_config.scheduler.eta_min = trial_params['eta_min']
        if 'scheduler_factor' in trial_params:
            training_config.scheduler.factor = trial_params['scheduler_factor']
        if 'scheduler_patience' in trial_params:
            training_config.scheduler.patience = trial_params['scheduler_patience']

    if cfg['wandb']['enabled']:
        run_name = f"trial_{trial.number}"
        wandb.init(
            project=cfg['wandb']['project'],
            entity=cfg['wandb']['entity'],
            config={**OmegaConf.to_container(OmegaConf.create(cfg), resolve=True), **trial_params},
            name=run_name,
            group=optimization_config.get('study_name', 'optuna_study'),
            reinit=True,
            tags=['optuna', cfg['model']['type'], cfg['task']]
        )
    else:
        wandb.init(mode="disabled")

    trainer = Trainer(
        model=model,
        datamodule=datamodule,
        config=training_config,
        device=device
    )

    best_val_acc = trainer.run()

    wandb.log({
        'trial/number': trial.number,
        'trial/best_val_accuracy': best_val_acc,
    })
    wandb.finish()

    trial.report(best_val_acc, step=1)

    # Handle pruning based on the intermediate value
    if trial.should_prune():
        raise optuna.TrialPruned()

    return best_val_acc


def run_optimization(optimization_config_path):
    """
    Run hyperparameter optimization study.

    Args:
        optimization_config_path: Path to optimization configuration file
    """
    with open(optimization_config_path, 'r') as f:
        optimization_config = yaml.safe_load(f)

    # Load base config using Hydra's compose API to properly resolve defaults
    from hydra import compose, initialize_config_dir
    import os

    config_dir = os.path.abspath('configs')
    base_config_name = os.path.splitext(os.path.basename(optimization_config['base_config']))[0]

    overrides = []
    if 'config_overrides' in optimization_config:
        for key, value in optimization_config['config_overrides'].items():
            if isinstance(value, bool):
                overrides.append(f"{key}={str(value).lower()}")
            else:
                overrides.append(f"{key}={value}")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=base_config_name, overrides=overrides)

    set_seed(cfg.seed)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    preprocessing_config = cfg_dict['preprocessing']

    dataset = EEGDataset(
        data_dir=cfg_dict['data']['data_dir'],
        task=cfg_dict['task'],
        preprocessing_config=preprocessing_config
    )

    study_name = optimization_config.get('study_name', 'eeg_optimization')
    storage_path = optimization_config.get('storage', f'sqlite:///{study_name}.db')

    pruner_config = optimization_config.get('pruner', {})
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=pruner_config.get('n_startup_trials', 5),
        n_warmup_steps=pruner_config.get('n_warmup_steps', 0)
    )

    sampler_config = optimization_config.get('sampler', {})
    sampler = optuna.samplers.TPESampler(
        seed=cfg['seed'],
        n_startup_trials=sampler_config.get('n_startup_trials', 10)
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        direction='maximize',
        pruner=pruner,
        sampler=sampler
    )

    wandb_kwargs = {}
    if cfg_dict['wandb']['enabled']:
        wandb_kwargs = {
            'wandb_kwargs': {
                'project': cfg_dict['wandb']['project'],
                'entity': cfg_dict['wandb']['entity'],
                'group': study_name,
            }
        }

    n_trials = optimization_config.get('n_trials', 50)
    timeout = optimization_config.get('timeout', None)

    print(f"Number of trials: {n_trials}")
    print(f"Model type: {cfg_dict['model']['type']}")
    print(f"Task: {cfg_dict['task']}")

    study.optimize(
        lambda trial: objective(trial, cfg_dict, dataset, optimization_config),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    print("\n" + "="*80)
    print("Optimization completed!")
    print("="*80)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.4f}")

    output_dir = optimization_config.get('output_dir', 'checkpoints/optimization_results')
    os.makedirs(output_dir, exist_ok=True)

    best_params_path = os.path.join(output_dir, f'{study_name}_best_params.yaml')
    with open(best_params_path, 'w') as f:
        yaml.dump({
            'best_trial': study.best_trial.number,
            'best_value': float(study.best_value),
            'best_params': study.best_params,
        }, f, default_flow_style=False)

    # try:
    #     import optuna.visualization as vis
    #     import plotly

    #     # Parameter importance
    #     fig_importance = vis.plot_param_importances(study)
    #     fig_importance.write_html(os.path.join(output_dir, f'{study_name}_param_importance.html'))

    #     # Optimization history
    #     fig_history = vis.plot_optimization_history(study)
    #     fig_history.write_html(os.path.join(output_dir, f'{study_name}_history.html'))

    #     # Parallel coordinate plot
    #     fig_parallel = vis.plot_parallel_coordinate(study)
    #     fig_parallel.write_html(os.path.join(output_dir, f'{study_name}_parallel_coordinate.html'))

    #     print(f"Visualization plots saved to: {output_dir}")
    # except Exception as e:
    #     print(f"Could not generate visualization plots: {e}")

    return study

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization with Optuna')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to optimization configuration file'
    )
    args = parser.parse_args()
    run_optimization(args.config)

if __name__ == "__main__":
    main()
