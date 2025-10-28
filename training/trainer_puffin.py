# training/trainer_puffin.py
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import time
from training.losses import CombinedLoss
from utils.path_utils import ensure_absolute_path, create_directory


class PuffinTrainer:
    """Puffin-specific trainer - single-stage training"""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.training_config = config['training']

        # Fix all paths
        self.config['paths']['model_dir'] = ensure_absolute_path(config['paths']['model_dir'])
        self.config['paths']['log_dir'] = ensure_absolute_path(config['paths']['log_dir'])
        self.config['paths']['output_dir'] = ensure_absolute_path(config['paths']['output_dir'])

        # Create directories
        create_directory(self.config['paths']['model_dir'])
        create_directory(self.config['paths']['log_dir'])
        create_directory(self.config['paths']['output_dir'])

        # Puffin uses CombinedLoss
        self.criterion = CombinedLoss(
            auxiliary_weight=self.training_config.get('auxiliary_weight', 1e-3)
        )

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.training_config['learning_rate']
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() and
                                             config['model']['use_cuda'] else 'cpu')
        self.model.to(self.device)

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_corrs = []
        self.val_corrs = []

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, (sequences, targets) in enumerate(tqdm(self.train_loader, desc="Training")):
            sequences, targets = sequences.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for sequences, targets in self.val_loader:
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def calculate_correlation(self, loader):
        """Calculate correlation between predictions and targets"""
        self.model.eval()
        correlations = []

        with torch.no_grad():
            for sequences, targets in loader:
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                outputs = self.model(sequences)

                for i in range(outputs.shape[0]):
                    pred_vec = outputs[i, 0].flatten().cpu().numpy()  # Single channel
                    target_vec = targets[i].flatten().cpu().numpy()

                    mask = ~(np.isnan(pred_vec) | np.isnan(target_vec))
                    pred_vec = pred_vec[mask]
                    target_vec = target_vec[mask]

                    if len(pred_vec) > 1:
                        correlation = np.corrcoef(pred_vec, target_vec)[0, 1]
                        if not np.isnan(correlation):
                            correlations.append(correlation)

        return np.mean(correlations) if correlations else 0.0

    def train(self):
        """Puffin single-stage training"""
        print("Starting Puffin training...")
        print("Note: This is a simplified training process for demonstration.")
        print("Real motif discovery requires multi-stage training with motif validation.")
        print("=" * 60)

        total_epochs = self.training_config['num_epochs']

        for epoch in range(total_epochs):
            start_time = time.time()

            train_loss = self.train_epoch()
            val_loss = self.validate()
            train_corr = self.calculate_correlation(self.train_loader)
            val_corr = self.calculate_correlation(self.val_loader)

            epoch_time = time.time() - start_time

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_corrs.append(train_corr)
            self.val_corrs.append(val_corr)

            print(f'Epoch {epoch + 1}/{total_epochs} | Time: {epoch_time:.1f}s')
            print(f'  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'  Train Corr: {train_corr:.4f} | Val Corr: {val_corr:.4f}')

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, True)
                self.patience_counter = 0
                print(f'  New best model! Val Loss: {val_loss:.4f}')
            else:
                self.patience_counter += 1
                print(f'  Patience counter: {self.patience_counter}/{self.training_config["early_stopping_patience"]}')

            # Early stopping
            if self.patience_counter >= self.training_config['early_stopping_patience']:
                print("Early stopping triggered")
                break

            # Periodic saving
            if (epoch + 1) % self.training_config['save_freq'] == 0:
                self.save_checkpoint(epoch + 1)

        print("Puffin training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation correlation: {max(self.val_corrs) if self.val_corrs else 0:.4f}")

    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint - cross-platform compatible"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_corrs': self.train_corrs,
            'val_corrs': self.val_corrs
        }

        model_dir = self.config['paths']['model_dir']

        filename = f"puffin_checkpoint_epoch_{epoch}.pth"
        if is_best:
            filename = "puffin_best_model.pth"

        # Use os.path.join for cross-platform compatibility
        save_path = os.path.join(model_dir, filename)

        torch.save(checkpoint, save_path)

        # Verify file was actually saved
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path) / 1024 / 1024  # MB
            print(f"Checkpoint saved: {save_path} ({file_size:.2f} MB)")
        else:
            print(f"Error: File not saved: {save_path}")