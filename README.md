# Text-Conditioned Object-Relative Geometry for Open-Vocabulary Part Segmentation

This project studies whether explicit spatial geometry can improve and explain text-conditioned part segmentation beyond what is already encoded by strong pretrained vision-language models.

Given:

* an RGB image,
* a natural-language part query such as `head`, `wheel`, or `wing`,
* and, in later experiments, a parent-object mask,

the model predicts a binary pixel-level mask for the requested object part.

The project uses:

* **DINOv2 ViT-S/14** as a frozen visual encoder,
* **CLIP ViT-B/32** as a frozen text encoder,
* a lightweight trainable segmentation decoder,
* Pascal-Part-116 for object and part annotations.

The final experimental pipeline compares:

```text
E1: DINOv2 + text
E2: DINOv2 + text + parent-object mask
E3: + absolute image coordinates (X, Y)
E4: + object-relative coordinates (U, V)
E5: + object-relative geometry (U, V, D)
E6: + query-gated object-relative geometry
```

Additional robustness experiments will study:

```text
rotation robustness
noisy parent-object masks
```

---

## Current Baseline Result

The first experiment, **E1: DINOv2 + CLIP text conditioning**, has been trained on Pascal-Part-116.

```text
Training samples:   53,067
Validation samples: 4,957
Input resolution:   448 x 448
Epochs:             5
```

Best validation result:

| Metric          |  Value |
| --------------- | -----: |
| Mean IoU        | 0.4531 |
| Mean Dice       | 0.5672 |
| Validation Loss | 0.4933 |

The E1 baseline does **not** use the parent-object mask or explicit geometry. It therefore provides the reference point for all later experiments.

---

# Repository Structure

```text
text-conditioned-part-segmentation/
│
├── src/
│   ├── __init__.py
│   ├── baseline_model.py
│   ├── object_mask_model.py
│   ├── dinov2_features.py
│   ├── dinov2_preprocess.py
│   ├── losses.py
│   ├── metrics.py
│   ├── pascal_part_dataset.py
│   ├── pascal_part_metadata.py
│   ├── train_baseline.py
│   ├── train_object_mask.py
│   └── visualize_sample.py
│
├── results/
│   └── experiments.md
│
├── run_train_baseline.slurm
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Source Files

## `src/pascal_part_metadata.py`

Contains the Pascal-Part-116 object and object-part class definitions.

It maps numeric annotation IDs to semantic names, for example:

```text
18  -> train
109 -> train's head
```

It also extracts the text query:

```text
train's head -> head
```

The file additionally stores the unseen-object categories used by Pascal-Part-116.

---

## `src/pascal_part_dataset.py`

Implements the PyTorch dataset.

Each Pascal image can contain several objects and several annotated parts. The loader converts them into individual text-conditioned samples.

For example:

```text
image + train + "head" -> train-head mask
image + train + "roof" -> train-roof mask
image + person + "leg" -> person-leg mask
```

Each sample returns:

```python
{
    "image_id": ...,
    "image": ...,
    "parent_mask": ...,
    "part_mask": ...,
    "query": ...,
    "object_id": ...,
    "part_id": ...
}
```

Images and masks are resized to `448 x 448`.

---

## `src/dinov2_preprocess.py`

Applies ImageNet normalization before images are passed to DINOv2.

Input:

```text
[B, 3, 448, 448]
```

Output has the same shape but normalized pixel values.

---

## `src/dinov2_features.py`

Converts DINOv2 patch tokens into a spatial feature map.

DINOv2 ViT-S/14 produces:

```text
[B, 1024, 384]
```

for a `448 x 448` image.

Since:

```text
448 / 14 = 32
```

the 1024 tokens correspond to a `32 x 32` grid.

The file reshapes them into:

```text
[B, 384, 32, 32]
```

for segmentation.

---

## `src/baseline_model.py`

Implements Experiment E1.

Pipeline:

```text
DINOv2 visual features
        +
CLIP text embedding
        ↓
feature projection
        ↓
text-conditioned feature fusion
        ↓
convolutional decoder
        ↓
binary part mask
```

The model uses approximately 1.1M trainable parameters.

DINOv2 and CLIP remain frozen.

---

## `src/object_mask_model.py`

Implements Experiment E2.

It extends the baseline by adding the binary parent-object mask.

Pipeline:

```text
DINOv2 features ───────┐
                       │
CLIP text embedding ───┼── fusion → decoder → part mask
                       │
parent-object mask ────┘
```

This experiment tests whether explicitly identifying the parent object improves part segmentation.

---

## `src/losses.py`

Contains the segmentation training loss.

The current objective is:

```text
BCE loss + Dice loss
```

Binary cross-entropy provides pixel-level supervision, while Dice loss helps handle the strong foreground/background imbalance caused by small object parts.

---

## `src/metrics.py`

Contains the main evaluation metrics:

```text
Mean Intersection over Union (IoU)
Dice score
```

These are computed from thresholded binary segmentation predictions.

---

## `src/train_baseline.py`

Full training pipeline for Experiment E1.

It:

1. loads Pascal-Part-116,
2. loads frozen DINOv2,
3. loads frozen CLIP,
4. extracts image and text features,
5. trains the segmentation decoder,
6. evaluates on the validation set,
7. reports IoU and Dice,
8. saves `last.pt` and `best.pt`.

---

## `src/train_object_mask.py`

Training pipeline for Experiment E2.

It uses the same:

```text
dataset
DINOv2
CLIP
optimizer
loss
resolution
train/validation split
```

as E1, but additionally provides the parent-object mask to the model.

Keeping the rest of the setup unchanged allows a controlled comparison between E1 and E2.

---

## `src/visualize_sample.py`

Visualizes Pascal-Part samples including:

```text
RGB image
parent-object mask
target part mask
```

This is useful for dataset verification and later qualitative model comparisons.

---

# Model Pipeline

## E1 Baseline

```text
RGB image
   │
   ▼
resize to 448 x 448
   │
   ▼
ImageNet normalization
   │
   ▼
Frozen DINOv2 ViT-S/14
   │
   ▼
patch tokens
[B, 1024, 384]
   │
   ▼
spatial reshape
[B, 384, 32, 32]
   │
   │
   │              text query
   │                  │
   │                  ▼
   │          Frozen CLIP ViT-B/32
   │                  │
   │                  ▼
   │              [B, 512]
   │                  │
   └──────────┬───────┘
              ▼
       feature projection
              │
              ▼
       text-conditioned fusion
              │
              ▼
       segmentation decoder
              │
              ▼
       [B, 1, 448, 448]
              │
              ▼
       predicted part mask
```

---

## E2 Parent-Mask Model

```text
RGB image
   │
   ▼
DINOv2
   │
   ▼
visual features ─────────────┐
                             │
text query → CLIP ───────────┼── fusion
                             │
parent-object mask ──────────┘
                             │
                             ▼
                         decoder
                             │
                             ▼
                         part mask
```

---

# Dataset Setup

The project uses **Pascal-Part-116**.

Create a data directory:

```bash
mkdir -p data
cd data
```

Download the dataset archive:

```bash
gdown "https://drive.google.com/uc?id=1QF0BglrcC0teKqx15vP8qJNakGgCWaEH"
```

Extract it:

```bash
tar -xzf PascalPart116.tar.gz
```

The archive may contain macOS metadata files. They can be removed with:

```bash
find PascalPart116 -name '._*' -type f -delete
find PascalPart116 -name '.DS_Store' -type f -delete
```

The resulting structure should be:

```text
data/PascalPart116/
├── images/
│   ├── train/
│   └── val/
├── annotations_detectron2_obj/
│   ├── train/
│   └── val/
├── annotations_detectron2_part/
│   ├── train/
│   └── val/
└── train_16shot.json
```

Expected image counts:

```text
Train: 8431
Validation: 850
```

Verify with:

```bash
find data/PascalPart116/images/train -type f | wc -l
find data/PascalPart116/images/val -type f | wc -l
```

The dataset itself is intentionally excluded from Git.

---

# Environment Setup

Clone the repository:

```bash
git clone https://github.com/DinelyWellehewage/text-conditioned-part-segmentation.git
cd text-conditioned-part-segmentation
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Locally

The full training pipeline can be started with:

```bash
python -m src.train_baseline
```

By default, the code expects Pascal-Part-116 at:

```text
data/PascalPart116
```

A different dataset location can be provided using:

```bash
PASCAL_PART_ROOT=/path/to/PascalPart116 \
python -m src.train_baseline
```

For E2:

```bash
PASCAL_PART_ROOT=/path/to/PascalPart116 \
python -m src.train_object_mask
```

Training with DINOv2 and CLIP is significantly faster on a CUDA GPU.

---

# FAU Alex Cluster Setup

On the cluster, Python 3.12 is required for the current DINOv2 source.

Load:

```bash
module load python/3.12-base
```

Create and activate the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset can be stored outside the home directory, for example:

```text
/home/vault/<group>/<user>/text-conditioned-part-segmentation/data/PascalPart116
```

Then set:

```bash
export PASCAL_PART_ROOT=/home/vault/<group>/<user>/text-conditioned-part-segmentation/data/PascalPart116
```

The compute nodes may not have outbound internet access. DINOv2 and CLIP weights should therefore be cached before launching a Slurm job.

For Hugging Face models, an offline cache can be configured with:

```bash
export HF_HOME=/path/to/huggingface-cache
export HF_HUB_OFFLINE=1
```

---

# Baseline Training on Slurm

The baseline can be launched with:

```bash
sbatch run_train_baseline.slurm
```

Check the job:

```bash
squeue -u $USER
```

Inspect logs:

```bash
tail -f logs/baseline_<JOBID>.out
```

The best and final model checkpoints are stored under:

```text
checkpoints/baseline/
├── best.pt
└── last.pt
```

Checkpoints are excluded from Git.

---

# Experimental Plan

The project follows a controlled ablation strategy.

### E1 — Text-only conditioning

```text
DINOv2 + CLIP text
```

This establishes how much spatial information is already available from pretrained DINOv2 features.

### E2 — Parent-object conditioning

```text
DINOv2 + CLIP text + parent-object mask
```

Tests whether identifying the selected parent object improves segmentation.

### E3 — Absolute coordinates

```text
+ normalized image X,Y coordinates
```

Tests whether generic spatial information improves performance.

### E4 — Object-relative coordinates

```text
+ U,V
```

Tests whether coordinates relative to the selected object are more useful than absolute image coordinates.

### E5 — Boundary-aware geometry

```text
+ U,V,D
```

Adds normalized distance from the parent-object boundary.

### E6 — Query-gated geometry

```text
text query
    ↓
geometry gate
    ↓
adaptive weighting of U,V,D
```

Tests whether different part queries benefit from different geometric cues.

---

# Robustness Experiments

After the main ablations, the project will evaluate:

### Rotation

Rotate:

```text
RGB image
parent-object mask
ground-truth part mask
```

together and recompute geometry.

This tests whether object-relative coordinates depend strongly on canonical object orientation.

### Noisy parent masks

Perturb the parent-object mask using operations such as erosion or dilation.

This tests how sensitive the geometry-based models are to imperfect object masks.

---

# Research Goal

The main question is not simply whether adding geometry improves segmentation.

The project aims to determine:

```text
Does explicit object-relative geometry provide useful and
interpretable spatial information beyond what is already encoded
by strong pretrained visual features?
```

The E1 baseline therefore serves as the reference point, and every later experiment adds only one controlled source of spatial information.
