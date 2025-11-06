import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from .metrics import compute_metrics, get_classification_report


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
            weight_decay=config.weight_decay
        )

        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        train_loader = self.datamodule.train_dataloader()
        pbar = tqdm(train_loader, desc='Training', leave=False)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"Warning: NaN or Inf in input data at batch {batch_idx}")
                continue

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss at batch {batch_idx}")
                print(f"Output stats - min: {output.min()}, max: {output.max()}, mean: {output.mean()}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = output.argmax(dim=1)
            all_preds.append(preds)
            all_targets.append(target)

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
                all_preds.append(preds)
                all_targets.append(target)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_preds, all_targets)

        return total_loss / len(val_loader), metrics

    def train(self):
        for epoch in range(self.config.epochs):
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()

            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_metrics['accuracy'],
                'train_f1': train_metrics['f1_score'],
                'val_loss': val_loss,
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['f1_score'],
            })

            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                torch.save(self.model.state_dict(), f"checkpoints/best_model_{self.config.task}.pt")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    def run(self):
        print(f"Training on task: {self.config.task}")
        print(f"Device: {self.device}")

        self.train()

        print(f"\nBest validation accuracy: {self.best_val_acc:.4f}")

        return self.best_val_acc
