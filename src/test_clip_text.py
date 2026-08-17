import torch
import open_clip


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Loading CLIP...")

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    queries = [
        "head",
        "wheel",
        "wing",
        "roof",
    ]

    tokens = tokenizer(queries).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)

    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True,
    )

    print("\nQueries:")
    for query in queries:
        print(" ", query)

    print("\nText features:")
    print("Shape:", text_features.shape)
    print("Finite:", text_features.isfinite().all().item())

    print("\nEmbedding norms:")
    print(text_features.norm(dim=-1))


if __name__ == "__main__":
    main()