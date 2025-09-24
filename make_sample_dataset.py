import os
import json
import random

# Set your paths
ann_root = "/Users/amitpadegal/Desktop/Capstone/"  # e.g., ./data/annotations
train_files = ['vg_qa']  # whichever ones you want
urls = {
    # 'vqa_val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_val.json',
    'vg_qa': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/vg_qa.json'
}

# Combine full annotation list
full_annotation = []
for f in train_files:
    filepath = os.path.join(ann_root, f"{f}.json")
    if not os.path.exists(filepath):
        from torchvision.datasets.utils import download_url
        download_url(urls[f], ann_root)
    with open(filepath, 'r') as fp:
        full_annotation += json.load(fp)

print("Total samples:", len(full_annotation))
sample_size = 20000
small_annotation = random.sample(full_annotation, sample_size)

output_path = os.path.join(ann_root, 'gqa_20k_subset.json')
with open(output_path, 'w') as out:
    json.dump(small_annotation, out)

print(f"Saved 20k subset to {output_path}")

