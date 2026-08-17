import open_clip
import torch
from torch.utils.data import DataLoader

from .baseline_model import TextConditionedSegmentationBaseline
from .dinov2_features import patch_tokens_to_feature_map
from .dinov2_preprocess import normalize_for_dinov2
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

    batch = next(iter(loader))

    images = batch["image"].to(device)
    queries = list(batch["query"])

    print("Query:", queries)

    # --------------------------------------------------
    # DINOv2
    # --------------------------------------------------

    print("Loading DINOv2...")

    dinov2 = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
    ).to(device)

    dinov2.eval()

    for param in dinov2.parameters():
        param.requires_grad = False

    # --------------------------------------------------
    # CLIP
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Baseline decoder
    # --------------------------------------------------

    decoder = TextConditionedSegmentationBaseline().to(device)

    # --------------------------------------------------
    # Visual features
    # --------------------------------------------------

    normalized_images = normalize_for_dinov2(images)

    with torch.no_grad():
        dino_outputs = dinov2.forward_features(normalized_images)

    patch_tokens = dino_outputs["x_norm_patchtokens"]

    visual_features = patch_tokens_to_feature_map(
        patch_tokens
    )

    # --------------------------------------------------
    # Text features
    # --------------------------------------------------

    tokens = tokenizer(queries).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)

    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True,
    )

    # --------------------------------------------------
    # Segmentation
    # --------------------------------------------------

    logits = decoder(
        visual_features=visual_features,
        text_features=text_features,
        output_size=(448, 448),
    )

    probabilities = torch.sigmoid(logits)

    # --------------------------------------------------
    # Checks
    # --------------------------------------------------

    print("\nShapes")

    print("Images:", images.shape)
    print("Visual features:", visual_features.shape)
    print("Text features:", text_features.shape)
    print("Logits:", logits.shape)
    print("Target mask:", batch["part_mask"].shape)

    print("\nPrediction")

    print("Probability min:", probabilities.min().item())
    print("Probability max:", probabilities.max().item())
    print("Finite:", probabilities.isfinite().all().item())

    print("\nTrainable decoder parameters:")
    print(
        sum(
            p.numel()
            for p in decoder.parameters()
            if p.requires_grad
        )
    )


if __name__ == "__main__":
    main()