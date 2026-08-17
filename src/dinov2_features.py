import math

import torch


def patch_tokens_to_feature_map(patch_tokens):
    """
    Convert DINOv2 patch tokens from:

        [B, N, C]

    to:

        [B, C, H, W]

    Assumes the patch grid is square.
    """
    if patch_tokens.ndim != 3:
        raise ValueError(
            f"Expected patch tokens with shape [B, N, C], "
            f"got {patch_tokens.shape}"
        )

    batch_size, num_patches, feature_dim = patch_tokens.shape

    grid_size = int(math.sqrt(num_patches))

    if grid_size * grid_size != num_patches:
        raise ValueError(
            f"Number of patches ({num_patches}) does not form a square grid"
        )

    feature_map = patch_tokens.reshape(
        batch_size,
        grid_size,
        grid_size,
        feature_dim,
    )

    feature_map = feature_map.permute(0, 3, 1, 2).contiguous()

    return feature_map