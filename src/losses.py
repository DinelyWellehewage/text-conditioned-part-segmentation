import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, targets, eps=1e-6):
    """
    Dice loss for binary segmentation.

    Args:
        logits:
            Tensor of shape [B, 1, H, W]

        targets:
            Tensor of shape [B, H, W]
            or [B, 1, H, W]

    Returns:
        Scalar Dice loss.
    """

    if targets.ndim == 3:
        targets = targets.unsqueeze(1)

    targets = targets.float()

    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * targets).sum(
        dim=(1, 2, 3)
    )

    denominator = probabilities.sum(
        dim=(1, 2, 3)
    ) + targets.sum(
        dim=(1, 2, 3)
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
        )

        dice = dice_loss(
            logits,
            targets,
        )

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
        )

        return {
            "loss": total,
            "bce_loss": bce,
            "dice_loss": dice,
        }