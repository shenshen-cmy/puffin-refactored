# training/losses.py
import torch
import torch.nn as nn
import numpy as np


class CombinedLoss(nn.Module):
    """
    Combined loss function for Puffin
    KL divergence (shape) + pseudo-Poisson loss (magnitude)
    """

    def __init__(self, auxiliary_weight=1e-3):
        super(CombinedLoss, self).__init__()
        self.auxiliary_weight = auxiliary_weight

    def forward(self, pred, target):
        # Ensure valid inputs
        pred = torch.clamp(pred, min=1e-10)
        target = torch.clamp(target, min=1e-10)

        # KL divergence loss - focuses on signal shape
        pred_norm = pred / (pred.sum(dim=-1, keepdim=True) + 1e-10)
        target_norm = target / (target.sum(dim=-1, keepdim=True) + 1e-10)

        # Compute KL divergence
        kl_terms = target_norm * (torch.log(target_norm + 1e-10) - torch.log(pred_norm + 1e-10))
        kl_loss = kl_terms.sum(dim=-1).mean()

        # Pseudo-Poisson loss - focuses on signal magnitude
        poisson_loss = (pred / np.log(10) - target * torch.log(pred / np.log(10) + 1e-10)).mean()

        # Combined loss
        total_loss = kl_loss + self.auxiliary_weight * poisson_loss

        return total_loss


class FixedPseudoPoissonLoss(nn.Module):
    """
    Fixed pseudo-Poisson loss function for Puffin_D
    """

    def __init__(self, epsilon=1e-8):
        super(FixedPseudoPoissonLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        # Convert to count scale
        pred_count = torch.pow(10.0, pred) - 1.0
        target_count = torch.pow(10.0, target) - 1.0

        # Ensure numerical stability
        pred_count = torch.clamp(pred_count, min=self.epsilon)
        target_count = torch.clamp(target_count, min=0)

        poisson_loss = pred_count - target_count * torch.log(pred_count + self.epsilon)

        return poisson_loss.mean()