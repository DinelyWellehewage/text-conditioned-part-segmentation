import torch


def _prepare_masks(logits, targets, threshold=0.5):
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)

    probabilities = torch.sigmoid(logits)

    predictions = probabilities >= threshold
    targets = targets >= 0.5

    return predictions, targets


def binary_iou(logits, targets, threshold=0.5, eps=1e-6):
    """
    Compute mean IoU for binary segmentation.
    """

    predictions, targets = _prepare_masks(
        logits,
        targets,
        threshold,
    )

    intersection = (
        predictions & targets
    ).sum(dim=(1, 2, 3)).float()

    union = (
        predictions | targets
    ).sum(dim=(1, 2, 3)).float()

    iou = (intersection + eps) / (union + eps)

    return iou.mean()


def binary_dice_score(logits, targets, threshold=0.5, eps=1e-6):
    """
    Compute mean Dice score for binary segmentation.
    """

    predictions, targets = _prepare_masks(
        logits,
        targets,
        threshold,
    )

    intersection = (
        predictions & targets
    ).sum(dim=(1, 2, 3)).float()

    prediction_sum = predictions.sum(
        dim=(1, 2, 3)
    ).float()

    target_sum = targets.sum(
        dim=(1, 2, 3)
    ).float()

    dice = (
        2.0 * intersection + eps
    ) / (
        prediction_sum + target_sum + eps
    )

    return dice.mean()