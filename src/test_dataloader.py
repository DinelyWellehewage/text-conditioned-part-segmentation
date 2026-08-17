from torch.utils.data import DataLoader

from pascal_part_dataset import PascalPartDataset


dataset = PascalPartDataset(
    root="data/PascalPart116",
    split="train",
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
)

batch = next(iter(loader))

print("Batch keys:")
for key in batch.keys():
    print(" ", key)

print("\nShapes")
print("Image:", batch["image"].shape)
print("Parent mask:", batch["parent_mask"].shape)
print("Part mask:", batch["part_mask"].shape)

print("\nMetadata")
print("Image ID:", batch["image_id"])
print("Query:", batch["query"])
print("Object ID:", batch["object_id"])
print("Part ID:", batch["part_id"])