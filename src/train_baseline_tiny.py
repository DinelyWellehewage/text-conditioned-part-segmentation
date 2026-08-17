import open_clip
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .baseline_model import TextConditionedSegmentationBaseline
from .dinov2_features import patch_tokens_to_feature_map
from .dinov2_preprocess import normalize_for_dinov2
from .losses import BCEDiceLoss
from .pascal_part_dataset import PascalPartDataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    dataset = PascalPartDataset(
        root="data/PascalPart116",
        split="train",
        image_size=(448, 448),
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    # -------------------------
    # Frozen DINOv2
    # -------------------------

    print("Loading DINOv2...")

    dinov2 = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
    ).to(device)

    dinov2.eval()

    for param in dinov2.parameters():
        param.requires_grad = False

    # -------------------------
    # Frozen CLIP
    # -------------------------

    print("Loading CLIP...")

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )

    clip_model = clip_model.to(device)
    clip_model.eval()

    for param in clip_model.parameters():
        param.requires_grad = False

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # -------------------------
    # Trainable decoder
    # -------------------------

    decoder = TextConditionedSegmentationBaseline().to(device)

    criterion = BCEDiceLoss()

    optimizer = AdamW(
        decoder.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # -------------------------
    # Tiny training run
    # -------------------------

    max_steps = 10

    decoder.train()

    for step, batch in enumerate(loader):
        if step >= max_steps:
            break

        images = batch["image"].to(device)
        targets = batch["part_mask"].to(device)
        queries = list(batch["query"])

        normalized_images = normalize_for_dinov2(images)

        with torch.no_grad():
            dino_outputs = dinov2.forward_features(
                normalized_images
            )

            patch_tokens = dino_outputs[
                "x_norm_patchtokens"
            ]

            visual_features = patch_tokens_to_feature_map(
                patch_tokens
            )

            tokens = tokenizer(queries).to(device)

            text_features = clip_model.encode_text(tokens)

            text_features = (
                text_features
                / text_features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

        optimizer.zero_grad()

        logits = decoder(
            visual_features=visual_features,
            text_features=text_features,
            output_size=targets.shape[-2:],
        )

        losses = criterion(
            logits,
            targets,
        )

        loss = losses["loss"]

        loss.backward()
        optimizer.step()

        print(
            f"Step {step + 1:02d}/{max_steps} | "
            f"Total: {loss.item():.4f} | "
            f"BCE: {losses['bce_loss'].item():.4f} | "
            f"Dice: {losses['dice_loss'].item():.4f}"
        )


if __name__ == "__main__":
    main()