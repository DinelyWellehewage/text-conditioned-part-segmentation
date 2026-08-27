import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectMaskConditionedSegmentation(nn.Module):
    def __init__(
        self,
        visual_dim=384,
        text_dim=512,
        hidden_dim=256,
    ):
        super().__init__()

        self.visual_projection = nn.Conv2d(
            visual_dim,
            hidden_dim,
            kernel_size=1,
        )

        self.text_projection = nn.Linear(
            text_dim,
            hidden_dim,
        )

        self.mask_projection = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(
        self,
        visual_features,
        text_features,
        parent_mask,
        output_size,
    ):
        visual = self.visual_projection(visual_features)

        text = self.text_projection(text_features)
        text = text[:, :, None, None]

        if parent_mask.ndim == 3:
            parent_mask = parent_mask.unsqueeze(1)

        parent_mask = F.interpolate(
            parent_mask.float(),
            size=visual.shape[-2:],
            mode="nearest",
        )

        mask_features = self.mask_projection(parent_mask)

        fused = visual * text
        fused = fused + mask_features

        logits = self.decoder(fused)

        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits