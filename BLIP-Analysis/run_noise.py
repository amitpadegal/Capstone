from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from vqa import blip_vqa
import os
import json
import random
import torch

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
transform = transforms.Compose([
        transforms.Resize((480, 480),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
# import io
# print(os.path.exists("image/dog_image.jpeg"))
# with open("image/dog_image.jpeg", "rb") as f:
#     image = Image.open(io.BytesIO(f.read())).convert("RGB")


with open("/Users/amitpadegal/Desktop/Capstone/vqa_20k_subset.json", "r") as f:
    data = json.load(f)
# print(data[:1000])  # Print the first 1000 characters to verify content

# noise_img = img + torch.randn_like(img) * noise_strength
# noise_img = torch.clamp(noise_img, 0.0, 1.0)

l = []
dummy_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."

# jumbled_sentences = [jumble_sentence(data[i]['question']) for i in range(15)]
model = blip_vqa(pretrained='model_vqa.pth')
for i in range(2):
        image_path = os.path.join('/Users/amitpadegal/Desktop/Capstone/vqa_sample', data[i]['image'].split('/')[1])
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)
        noise_half = torch.randn_like(image) * 0.5
        noise_full = torch.randn_like(image) * 1.0
        original = data[i]['question']
        
        _, res_o = model(image, original, "orig_ablation.json",inference= 'generate')
        _, res_h = model(noise_half, original, "half_noise_ablation.json",inference= 'generate')
        _, res_f = model(noise_full, original, "full_noise_ablation.json",inference= 'generate')
        _, res_d = model(image, dummy_text, "dummy_ablation.json",inference= 'generate')
        d = {}
        d['original'] = res_o
        d['half_noise'] = res_h
        d['full_noise'] = res_f
        d['dummy_text'] = res_d
        l.append(d)
        # print(original, long, jumble)

print(l)
with open("blip_vqa_output_5.json", "w") as f:  
        json.dump(l, f, indent=2)
        
        # print(f"Q: {question}")
        # print(f"A: {logits}\n")

# image = Image.open("/Users/amitpadegal/Desktop/Capstone/BLIP-Analysis/image/dog_image.jpeg").convert('RGB')
# image = transform(image)
# image = image.unsqueeze(0)
# # print(image.shape)``
# question = "Given the distinct physical characteristics presented in this photograph, including the dog's coat texture, facial structure, and overall build, could you provide a precise identification of the canine breed depicted?"

# model = blip_vqa(pretrained='model_vqa.pth')

# logits = model(image, question, inference= 'generate')
# print(logits)
