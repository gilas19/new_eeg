import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from .metrics import compute_metrics


class Trainer:
    def __init__(self, model, datamodule, config, device='cuda'):
        self.model = model.to(device)
        self.datamodule = datamodule
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.get('weight_decay', 0.0)
        )

        scheduler_config = getattr(config, 'scheduler', None)
        if scheduler_config is not None and scheduler_config.get('enabled', False):
            scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau')
            if scheduler_type == 'ReduceLROnPlateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=scheduler_config.get('factor', 0.5),
                    patience=scheduler_config.get('patience', 5),
                    verbose=True
                )
            elif scheduler_type == 'StepLR':
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get('step_size', 30),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            elif scheduler_type == 'CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_config.get('T_max', config.epochs) * datamodule.num_batches_per_epoch(),
                    eta_min=scheduler_config.get('eta_min', 1e-6)
                )
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        else:
            self.scheduler = None

        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        train_loader = self.datamodule.train_dataloader()
        pbar = tqdm(train_loader, desc=f'Training')

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                self.scheduler.step()

            total_loss += loss.item()
            preds = output.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())

            pbar.set_postfix({'loss': loss.item()})

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_preds, all_targets)

        return total_loss / len(train_loader), metrics

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        val_loader = self.datamodule.val_dataloader()

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                preds = output.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(target.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_preds, all_targets)

        return total_loss / len(val_loader), metrics

    def test(self):
        """Evaluate model on test set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        test_loader = self.datamodule.test_dataloader()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                preds = output.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(target.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_preds, all_targets)

        return total_loss / len(test_loader), metrics

    def train(self):
        for epoch in range(self.config.epochs):
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()

            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"\nEpoch {epoch+1}/{self.config.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")

            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/accuracy': train_metrics['accuracy'],
                'train/precision': train_metrics['precision'],
                'train/recall': train_metrics['recall'],
                'train/f1': train_metrics['f1_score'],
                'val/loss': val_loss,
                'val/accuracy': val_metrics['accuracy'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/f1': val_metrics['f1_score'],
                'learning_rate': current_lr,
                'patience_counter': self.patience_counter,
                'best_val_accuracy': self.best_val_acc,
            }

            wandb.log(log_dict)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR):
                    self.scheduler.step()

            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                # torch.save(self.model.state_dict(), f"checkpoints/best_model_{self.config.task}.pt")
                wandb.log({'best_epoch': epoch, 'checkpoint_saved': True})
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.patience:
                wandb.log({'early_stopped_at_epoch': epoch, 'early_stopping_triggered': True})
                break

    def run(self):
        wandb.log({
            'config/task': self.config.task,
            'config/epochs': self.config.epochs,
            'config/learning_rate': self.config.learning_rate,
            'config/patience': self.config.patience,
            'config/weight_decay': self.config.get('weight_decay', 0.0),
            'config/device': str(self.device),
            'config/optimizer': 'Adam',
        })

        self.train()

        wandb.log({
            'final/best_val_accuracy': self.best_val_acc,
        })

        return self.best_val_acc
