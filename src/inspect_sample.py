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

image_path = DATA_ROOT / "images/train" / f"{IMAGE_ID}.jpg"
obj_path = DATA_ROOT / "annotations_detectron2_obj/train" / f"{IMAGE_ID}.png"
part_path = DATA_ROOT / "annotations_detectron2_part/train" / f"{IMAGE_ID}.png"

image = np.array(Image.open(image_path).convert("RGB"))
obj_mask = np.array(Image.open(obj_path))
part_mask = np.array(Image.open(part_path))

print("Image shape:", image.shape)
print("Object mask shape:", obj_mask.shape)
print("Part mask shape:", part_mask.shape)

object_ids = [int(x) for x in np.unique(obj_mask) if x != 255]
part_ids = [int(x) for x in np.unique(part_mask) if x != 255]

print("\nObjects in image:")
for object_id in object_ids:
    print(f"  {object_id}: {get_object_name(object_id)}")

print("\nParts in image:")
for part_id in part_ids:
    print(
        f"  {part_id}: "
        f"{get_part_name(part_id)} "
        f"(query='{get_query_name(part_id)}')"
    )