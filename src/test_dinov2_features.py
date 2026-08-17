import torch
from torch.utils.data import DataLoader

from .dinov2_preprocess import normalize_for_dinov2
from .pascal_part_dataset import PascalPartDataset


def main():
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

    images = normalize_for_dinov2(batch["image"])

    print("Loading DINOv2...")

    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
    )

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    print("DINOv2 loaded")

    with torch.no_grad():
        features = model.forward_features(images)

    print("\nAvailable feature outputs:")
    for key, value in features.items():
        if torch.is_tensor(value):
            print(f"{key}: {tuple(value.shape)}")

    patch_tokens = features["x_norm_patchtokens"]

    print("\nDense patch features:")
    print("Shape:", patch_tokens.shape)
    print("Finite:", patch_tokens.isfinite().all().item())


if __name__ == "__main__":
    main()