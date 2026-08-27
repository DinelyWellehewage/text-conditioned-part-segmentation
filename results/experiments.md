# Experiment Results

## E1 — DINOv2 + CLIP Text Baseline

Architecture:
- Frozen DINOv2 ViT-S/14 visual encoder
- Frozen CLIP ViT-B/32 text encoder
- Lightweight text-conditioned segmentation decoder
- No parent-object mask
- No explicit geometry

Training:
- Dataset: Pascal-Part-116
- Train text-conditioned samples: 53,067
- Validation samples: 4,957
- Input resolution: 448 x 448
- Batch size: 4
- Epochs: 5
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4
- Loss: BCE + Dice

Best validation result:

| Metric | Value |
|---|---:|
| Mean IoU | 0.4531 |
| Mean Dice | 0.5672 |
| Validation Loss | 0.4933 |
| Best Epoch | 5 |

Training progression:

| Epoch | Train Loss | Val Loss | Val IoU | Val Dice |
|---|---:|---:|---:|---:|
| 1 | 0.6021 | 0.5268 | 0.4225 | 0.5383 |
| 2 | 0.5254 | 0.5066 | 0.4408 | 0.5559 |
| 3 | 0.5049 | 0.5016 | 0.4463 | 0.5607 |
| 4 | 0.4906 | 0.5045 | 0.4437 | 0.5591 |
| 5 | 0.4794 | 0.4933 | 0.4531 | 0.5672 |
