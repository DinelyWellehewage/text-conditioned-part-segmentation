import open_clip
import torch
from torch.utils.data import DataLoader

from .baseline_model import TextConditionedSegmentationBaseline
from .dinov2_features import patch_tokens_to_feature_map
from .dinov2_preprocess import normalize_for_dinov2
from .metrics import binary_dice_score, binary_iou
from .pascal_part_dataset import PascalPartDataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    dataset = PascalPartDataset(
        root="data/PascalPart116",
        split="val",
        image_size=(448, 448),
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print("Validation samples:", len(dataset))

    # Frozen DINOv2
    print("Loading DINOv2...")

    dinov2 = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
    ).to(device)

    dinov2.eval()

    for param in dinov2.parameters():
        param.requires_grad = False

    # Frozen CLIP
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

    # Randomly initialized baseline decoder
    decoder = TextConditionedSegmentationBaseline().to(device)
    decoder.eval()

    max_samples = 5

    total_iou = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= max_samples:
                break

            images = batch["image"].to(device)
            targets = batch["part_mask"].to(device)
            queries = list(batch["query"])

            normalized_images = normalize_for_dinov2(images)

            dino_outputs = dinov2.forward_features(
                normalized_images
            )

            patch_tokens = dino_outputs["x_norm_patchtokens"]

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

            logits = decoder(
                visual_features=visual_features,
                text_features=text_features,
                output_size=targets.shape[-2:],
            )

            iou = binary_iou(
                logits,
                targets,
            )

            dice = binary_dice_score(
                logits,
                targets,
            )

            total_iou += iou.item()
            total_dice += dice.item()

            print(
                f"Sample {step + 1:02d} | "
                f"Query: {queries[0]:12s} | "
                f"IoU: {iou.item():.4f} | "
                f"Dice: {dice.item():.4f}"
            )

    evaluated = min(max_samples, len(dataset))

    print("\nValidation summary")
    print("Samples:", evaluated)
    print("Mean IoU:", total_iou / evaluated)
    print("Mean Dice:", total_dice / evaluated)


if __name__ == "__main__":
    main()