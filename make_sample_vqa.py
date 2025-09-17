import os
import json
import shutil

subset_json_path = "/Users/amitpadegal/Desktop/Capstone/vqa_20k_subset.json"  # your new small JSON
vqa_root = "/Users/amitpadegal/Downloads/"

with open(subset_json_path, 'r') as f:
    annotations = json.load(f)

# Collect full image paths
image_paths = set()  # use a set to avoid duplicates

for ann in annotations:
    image_paths.add(os.path.join(vqa_root, ann['image']))

print("Total unique images:", len(image_paths))

destination_root = "/Users/amitpadegal/Desktop/Capstone/vqa_sample"
os.makedirs(destination_root, exist_ok=True)

for path in image_paths:
    if os.path.exists(path):
        shutil.copy(path, destination_root)
    else:
        print(f"Warning: Image not found: {path}")
