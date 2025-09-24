import json
from collections import defaultdict
import random

with open("foildataset_10k.json", "r") as f:
    data = json.load(f)

ann = data["annotations"]

for a in ann:
    a["image_path"] = f'COCO_val2014_{a["image_id"]:012d}.jpg'
    a["random_caption"] = random.choice(ann)["original_caption"]

wrapped = {
    "annotations": ann}

with open("foildataset_10k_modified.json", "w") as f:
    json.dump(wrapped, f, indent=2)
# Load FOIL annotations
# with open("foildataset.json", "r") as f:
#     data = json.load(f)

# print(len(data["annotations"]))
# im = data["images"]

# samples = random.sample(data["annotations"], 10000)
# print(samples[0])
# images = dict()
# for sample in samples:
#     images[sample["image_id"]] = im[f'{sample["image_id"]}']

# wrapped = {
#     "annotations": samples,
#     "images": images
# }
# with open("foildataset_10k.json", "w") as f:
#     json.dump(wrapped, f, indent=2)

# annotations = data["annotations"]
# images = data["images"]
# # print(annotations[0])
# # Group annotations by foil_id
# pairs = defaultdict(dict)
# l = dict()
# for ann in annotations:
#     id = ann["id"]
#     label = ann["foil"]
#     if label:
#         pairs[id]["foil"] = ann
#     else:
#         pairs[id]["original"] = ann

# for img in images:
#     l[img["id"]] = img["file_name"]

# # print(l[0])
# # Create a paired dataset: (image_id, original_caption, foil_caption, target_word, foil_word)
# paired_dataset = []
# for foil_id, pair in pairs.items():

#     if "original" in pair and "foil" in pair:  # only keep complete pairs
#         # print(foil_id)
#         orig = pair["original"]
#         foil = pair["foil"]
#         paired_dataset.append({
#             "foil_id": foil_id,
#             "image_id": orig["image_id"],  # same for both
#             "original_caption": orig["caption"],
#             "foil_caption": foil["caption"],
#             "target_word": foil["target_word"],
#             "foil_word": foil["foil_word"],
#         })
# wrapped = {
#     "annotations": paired_dataset,
#     "images": l
# }

# # save to file
# with open("output.json", "w") as f:
#     json.dump(wrapped, f, indent=2)
# Example: see first 2 pairs


# 533128
# 1296684