from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pascal_part_metadata import get_object_name, get_part_name

DATA_ROOT = Path("data/PascalPart116")
IMAGE_ID = "2008_008336"

OBJECT_ID = 18
PART_ID = 109

image_path = DATA_ROOT / "images/train" / f"{IMAGE_ID}.jpg"
obj_path = DATA_ROOT / "annotations_detectron2_obj/train" / f"{IMAGE_ID}.png"
part_path = DATA_ROOT / "annotations_detectron2_part/train" / f"{IMAGE_ID}.png"

image = np.array(Image.open(image_path).convert("RGB"))
obj_labels = np.array(Image.open(obj_path))
part_labels = np.array(Image.open(part_path))

parent_mask = obj_labels == OBJECT_ID
part_mask = part_labels == PART_ID

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("RGB image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(parent_mask, cmap="gray")
plt.title(f"Parent: {get_object_name(OBJECT_ID)}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(part_mask, cmap="gray")
plt.title(f"Part: {get_part_name(PART_ID)}")
plt.axis("off")

output_path = Path("sample_visualization.png")

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved visualization to: {output_path}")