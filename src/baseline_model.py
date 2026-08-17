import torch
import torch.nn as nn
import torch.nn.functional as F


class TextConditionedSegmentationBaseline(nn.Module):
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
        output_size,
    ):
        """
        visual_features:
            [B, 384, 32, 32]

        text_features:
            [B, 512]

        output:
            [B, 1, H, W]
        """

        visual = self.visual_projection(visual_features)

        text = self.text_projection(text_features)
        text = text[:, :, None, None]

        fused = visual * text

        logits = self.decoder(fused)

        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits