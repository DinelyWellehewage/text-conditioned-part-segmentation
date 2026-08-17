import torch


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalize_for_dinov2(image):
    """
    Normalize an RGB tensor for DINOv2.

    Expected input:
        image: Tensor [3, H, W] or [B, 3, H, W]
        values in [0, 1]
    """
    if image.ndim == 3:
        mean = IMAGENET_MEAN.to(image.device)
        std = IMAGENET_STD.to(image.device)

    elif image.ndim == 4:
        mean = IMAGENET_MEAN.unsqueeze(0).to(image.device)
        std = IMAGENET_STD.unsqueeze(0).to(image.device)

    else:
        raise ValueError(
            f"Expected image with 3 or 4 dimensions, got {image.ndim}"
        )

    return (image - mean) / std