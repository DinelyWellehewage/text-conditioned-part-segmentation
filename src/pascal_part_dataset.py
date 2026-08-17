from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .pascal_part_metadata import (
    CLASS_NAMES,
    OBJ_CLASS_NAMES,
    get_query_name,
)

from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class PascalPartDataset(Dataset):
    def __init__(self, root, split="train", image_size=(448, 448)):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        self.image_dir = self.root / "images" / split
        self.obj_dir = self.root / "annotations_detectron2_obj" / split
        self.part_dir = self.root / "annotations_detectron2_part" / split

        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []

        image_paths = sorted(self.image_dir.glob("*.jpg"))

        for image_path in image_paths:
            image_id = image_path.stem

            obj_path = self.obj_dir / f"{image_id}.png"
            part_path = self.part_dir / f"{image_id}.png"

            if not obj_path.exists() or not part_path.exists():
                continue

            obj_labels = np.array(Image.open(obj_path))
            part_labels = np.array(Image.open(part_path))

            object_ids = [
                int(x)
                for x in np.unique(obj_labels)
                if x != 255 and x < len(OBJ_CLASS_NAMES)
            ]

            part_ids = [
                int(x)
                for x in np.unique(part_labels)
                if x != 255 and x < len(CLASS_NAMES)
            ]

            for part_id in part_ids:
                full_part_name = CLASS_NAMES[part_id]
                object_name = full_part_name.split("'s ", 1)[0]

                object_id = OBJ_CLASS_NAMES.index(object_name)

                if object_id not in object_ids:
                    continue

                samples.append(
                    {
                        "image_id": image_id,
                        "image_path": image_path,
                        "obj_path": obj_path,
                        "part_path": part_path,
                        "object_id": object_id,
                        "part_id": part_id,
                    }
                )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = np.array(
            Image.open(sample["image_path"]).convert("RGB")
        )

        obj_labels = np.array(Image.open(sample["obj_path"]))
        part_labels = np.array(Image.open(sample["part_path"]))

        object_id = sample["object_id"]
        part_id = sample["part_id"]

        parent_mask = (obj_labels == object_id).astype(np.float32)
        part_mask = (part_labels == part_id).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        parent_mask = torch.from_numpy(parent_mask)
        part_mask = torch.from_numpy(part_mask)


        image = TF.resize(
            image,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        parent_mask = TF.resize(
            parent_mask.unsqueeze(0),
            self.image_size,
            interpolation=InterpolationMode.NEAREST,
        ).squeeze(0)

        part_mask = TF.resize(
            part_mask.unsqueeze(0),
            self.image_size,
            interpolation=InterpolationMode.NEAREST,
        ).squeeze(0)



        query = get_query_name(part_id)

        return {
            "image_id": sample["image_id"],
            "image": image,
            "parent_mask": parent_mask,
            "part_mask": part_mask,
            "query": query,
            "object_id": object_id,
            "part_id": part_id,
        }