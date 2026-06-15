"""
Loss functions for X2BR-inspired implicit occupancy model.

X2BR uses binary cross-entropy classification loss on sampled point occupancies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OccupancyBCELoss(nn.Module):
    """Binary cross-entropy on sampled point occupancies (X2BR training loss).

    Operates on raw logits for numerical stability.
    Following X2BR: plain BCE without pos_weight or focal weighting.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, N, 1) raw occupancy logits from decoder
            targets: (B, N, 1) binary ground-truth occupancy {0, 1}
        """
        return F.binary_cross_entropy_with_logits(logits, targets)


class DiceLoss(nn.Module):
    """Dice loss on voxel grids (used for validation metrics)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
