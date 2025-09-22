import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

class GQADataset(Dataset):
    def __init__(self, gqa_json_path, image_root, transform=None, max_samples=None, seed=0):
        self.image_root = image_root
        self.transform = transform
        
        # Load JSON
        with open(gqa_json_path, 'r') as f:
            gqa_data = json.load(f)
        
        # Optionally subsample
        if max_samples is not None:
            import random
            random.seed(seed)
            gqa_data = random.sample(gqa_data, max_samples)

        
        # Store
        self.samples = []
        for idx, sample in enumerate(gqa_data):
            self.samples.append({
                "image_file": f'{sample["image"]}',
                "question": sample["question"],
                "question_id": sample["question_id"],
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img_path = sample['image_file'].split('/')[1]
        image_path = os.path.join(self.image_root, img_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        
        # Question & Answer
        question = sample["question"]
        question_id = sample["question_id"]
        
        return image, question, question_id

# if __name__ == "__main__":
#     # Example usage
#     normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
#     transform_test = transforms.Compose([
#         transforms.Resize((480,480),interpolation=InterpolationMode.BICUBIC),
#         transforms.ToTensor(),
#         normalize,
#         ])  
#     dataset = GQADataset(
#         gqa_json_path='/Users/amitpadegal/Desktop/Capstone/BLIP-Analysis/annotation/vg_qa.json',
#         image_root='/Users/amitpadegal/Downloads/',
#         transform=None,
#         max_samples=100,
#         seed=42
#     )
#     print(f'Dataset size: {len(dataset)}')
#     img, q, a, w = dataset[0]
#     print(f'Image shape: {img.size}, Question: {q}, Answer: {a}, Weights: {w}')