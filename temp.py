from lavis.datasets.builders import load_dataset
import os
from load_data import val_qa_pairs
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from lavis.models import load_model_and_preprocess
from lavis.datasets.builders import load_dataset
DATA_URL = {
    "train": "http://images.cocodataset.org/zips/train2014.zip",  # md5: 0da8c0bd3d6becc4dcb32757491aca88
    "val": "http://images.cocodataset.org/zips/val2014.zip",  # md5: a3d79f5ed8d289b7a7554ce06a5782b3
    "test": "http://images.cocodataset.org/zips/test2014.zip",  # md5: 04127eef689ceac55e3a572c2c92f264
    "test2015": "http://images.cocodataset.org/zips/test2015.zip",  # md5: 04127eef689ceac55e3a572c2c92f264
}

vqa_dataset = load_dataset("coco_vqa")
    
# print(vqa_dataset['val'][0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip_vqa", model_type="vqav2", is_eval=True, device=device
)

vqa_dataset = load_dataset("coco_vqa")

# print(f"Sample question: {vqa_dataset['val'][0]['image']}") #PIL Image
# print(f"Sample image path: {vqa_dataset['val'][0]['text_input']}")
# print(f"Sample answer: {vqa_dataset['val'][0]['question_id']}")
# print(f"Sample answer: {vqa_dataset['val'][0]['instance_id']}")