from pathlib import Path

import open_clip
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .baseline_model import TextConditionedSegmentationBaseline
from .dinov2_features import patch_tokens_to_feature_map
from .dinov2_preprocess import normalize_for_dinov2
from .losses import BCEDiceLoss
from .metrics import binary_dice_score, binary_iou
from .pascal_part_dataset import PascalPartDataset


DATA_ROOT = "data/PascalPart116"

IMAGE_SIZE = (448, 448)

BATCH_SIZE = 4
NUM_WORKERS = 0

NUM_EPOCHS = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

CHECKPOINT_DIR = Path("checkpoints/baseline")


def load_encoders(device):
    print("Loading DINOv2...")

    dinov2 = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
    ).to(device)

    dinov2.eval()

    for param in dinov2.parameters():
        param.requires_grad = False

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

    return dinov2, clip_model, tokenizer


def extract_features(
    images,
    queries,
    dinov2,
    clip_model,
    tokenizer,
    device,
):
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

    return visual_features, text_features


def train_one_epoch(
    decoder,
    loader,
    dinov2,
    clip_model,
    tokenizer,
    criterion,
    optimizer,
    device,
):
    decoder.train()

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0

    for batch in tqdm(
        loader,
        desc="Training",
        leave=False,
    ):
        images = batch["image"].to(device)
        targets = batch["part_mask"].to(device)
        queries = list(batch["query"])

        visual_features, text_features = extract_features(
            images,
            queries,
            dinov2,
            clip_model,
            tokenizer,
            device,
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

        losses["loss"].backward()
        optimizer.step()

        total_loss += losses["loss"].item()
        total_bce += losses["bce_loss"].item()
        total_dice_loss += losses["dice_loss"].item()

    num_batches = len(loader)

    return {
        "loss": total_loss / num_batches,
        "bce": total_bce / num_batches,
        "dice_loss": total_dice_loss / num_batches,
    }


def validate(
    decoder,
    loader,
    dinov2,
    clip_model,
    tokenizer,
    criterion,
    device,
):
    decoder.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="Validation",
            leave=False,
        ):
            images = batch["image"].to(device)
            targets = batch["part_mask"].to(device)
            queries = list(batch["query"])

            visual_features, text_features = extract_features(
                images,
                queries,
                dinov2,
                clip_model,
                tokenizer,
                device,
            )

            logits = decoder(
                visual_features=visual_features,
                text_features=text_features,
                output_size=targets.shape[-2:],
            )

            losses = criterion(
                logits,
                targets,
            )

            iou = binary_iou(
                logits,
                targets,
            )

            dice = binary_dice_score(
                logits,
                targets,
            )

            total_loss += losses["loss"].item()
            total_iou += iou.item()
            total_dice += dice.item()

    num_batches = len(loader)

    return {
        "loss": total_loss / num_batches,
        "iou": total_iou / num_batches,
        "dice": total_dice / num_batches,
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataset = PascalPartDataset(
        root=DATA_ROOT,
        split="train",
        image_size=IMAGE_SIZE,
    )

    val_dataset = PascalPartDataset(
        root=DATA_ROOT,
        split="val",
        image_size=IMAGE_SIZE,
    )

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    dinov2, clip_model, tokenizer = load_encoders(
        device
    )

    decoder = TextConditionedSegmentationBaseline().to(
        device
    )

    criterion = BCEDiceLoss()

    optimizer = AdamW(
        decoder.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_iou = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print()
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )
        print("-" * 40)

        train_metrics = train_one_epoch(
            decoder,
            train_loader,
            dinov2,
            clip_model,
            tokenizer,
            criterion,
            optimizer,
            device,
        )

        val_metrics = validate(
            decoder,
            val_loader,
            dinov2,
            clip_model,
            tokenizer,
            criterion,
            device,
        )

        print(
            f"Train | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"BCE: {train_metrics['bce']:.4f} | "
            f"Dice loss: {train_metrics['dice_loss']:.4f}"
        )

        print(
            f"Val   | "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"Dice: {val_metrics['dice']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
        }

        torch.save(
            checkpoint,
            CHECKPOINT_DIR / "last.pt",
        )

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]

            torch.save(
                checkpoint,
                CHECKPOINT_DIR / "best.pt",
            )

            print(
                f"Saved new best checkpoint "
                f"(IoU={best_iou:.4f})"
            )


if __name__ == "__main__":
    main()