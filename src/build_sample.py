from pathlib import Path

import numpy as np
from PIL import Image

from pascal_part_metadata import (
    get_object_name,
    get_part_name,
    get_query_name,
)

DATA_ROOT = Path("data/PascalPart116")
IMAGE_ID = "2008_008336"

# Pick one part from the image.
PART_ID = 109  # train's head
OBJECT_ID = 18  # train

image_path = DATA_ROOT / "images/train" / f"{IMAGE_ID}.jpg"
obj_path = DATA_ROOT / "annotations_detectron2_obj/train" / f"{IMAGE_ID}.png"
part_path = DATA_ROOT / "annotations_detectron2_part/train" / f"{IMAGE_ID}.png"

image = np.array(Image.open(image_path).convert("RGB"))
obj_labels = np.array(Image.open(obj_path))
part_labels = np.array(Image.open(part_path))

# Binary parent-object mask.
parent_mask = (obj_labels == OBJECT_ID).astype(np.uint8)

# Binary target-part mask.
part_mask = (part_labels == PART_ID).astype(np.uint8)

query = get_query_name(PART_ID)

print("Image ID:", IMAGE_ID)

print("\nObject")
print("  ID:", OBJECT_ID)
print("  Name:", get_object_name(OBJECT_ID))

print("\nPart")
print("  ID:", PART_ID)
print("  Full name:", get_part_name(PART_ID))
print("  Query:", query)

print("\nShapes")
print("  Image:", image.shape)
print("  Parent mask:", parent_mask.shape)
print("  Part mask:", part_mask.shape)

print("\nPixel counts")
print("  Parent-object pixels:", parent_mask.sum())
print("  Target-part pixels:", part_mask.sum())

# Sanity check:
# every target-part pixel should belong to the selected parent object.
outside_pixels = np.logical_and(part_mask == 1, parent_mask == 0).sum()

print("  Part pixels outside parent object:", outside_pixels)