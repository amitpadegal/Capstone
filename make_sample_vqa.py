import os
import json
import shutil

subset_json_path = "/Users/amitpadegal/Desktop/Capstone/vqa_20k_subset.json"  # your new small JSON
foil_dataset_path = "/Users/amitpadegal/Desktop/Capstone/foildataset_10k.json"  # original FOIL dataset JSON
vqa_root = "/Users/amitpadegal/Downloads/"

with open(subset_json_path, 'r') as f:
    annotations = json.load(f)
with open(foil_dataset_path, 'r') as f:
    foil_data = json.load(f)

# Collect full image paths
image_paths = set()  # use a set to avoid duplicates

for ann in annotations:
    image_paths.add(os.path.join(vqa_root, ann['image']))

tmp = foil_data['images']
for ann in tmp.values():
    image_paths.add(os.path.join(vqa_root, 'val2014/' + ann))

print("Total unique images:", len(image_paths))

destination_root = "/Users/amitpadegal/Desktop/Capstone/vqa_sample"
os.makedirs(destination_root, exist_ok=True)

for path in image_paths:
    if os.path.exists(path):
        shutil.copy(path, destination_root)
    else:
        print(f"Warning: Image not found: {path}")
