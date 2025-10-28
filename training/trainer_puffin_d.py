# training/trainer_puffin_d.py
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import time
from training.losses import FixedPseudoPoissonLoss


class PuffinDTrainer:
    """Puffin_D-specific trainer - uses pseudo-Poisson loss"""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.training_config = config['training']

        # Puffin_D uses FixedPseudoPoissonLoss
        self.criterion = FixedPseudoPoissonLoss()

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config.get('weight_decay', 1e-5)
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

            # Gradient clipping (needed for long sequences)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
        """Puffin_D training"""
        print("Starting Puffin_D training...")

        for epoch in range(self.training_config['num_epochs']):
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

            print(f'Epoch {epoch + 1}/{self.training_config["num_epochs"]} | Time: {epoch_time:.1f}s')
            print(f'  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'  Train Corr: {train_corr:.4f} | Val Corr: {val_corr:.4f}')

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.training_config['early_stopping_patience']:
                print("Early stopping triggered")
                break

            # Periodic saving
            if (epoch + 1) % self.training_config['save_freq'] == 0:
                self.save_checkpoint(epoch + 1)

        print("Puffin_D training completed!")

    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
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
        model_dir = os.path.normpath(model_dir)
        os.makedirs(model_dir, exist_ok=True)

        filename = f"puffin_d_checkpoint_epoch_{epoch}.pth"
        if is_best:
            filename = "puffin_d_best_model.pth"

        save_path = os.path.join(model_dir, filename)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")