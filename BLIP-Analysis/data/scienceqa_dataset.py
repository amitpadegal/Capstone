import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from data.utils import pre_question

from torchvision.datasets.utils import download_url
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset


class ScienceQADataset(Dataset):
    def __init__(self, transform=None, split='train'):
        self.ds = load_dataset("derek-thomas/ScienceQA")[split]
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]  # this is a dict

        # 'image' can be None, a URL, or a local path depending on the dataset version
        image = item['image']

        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif image is not None:
            image = image.convert('RGB')  # already a PIL Image
        else:
            # handle missing images if needed
            image = Image.new('RGB', (224, 224), color='white')

        if self.transform:
            image = self.transform(image)

        question = item['question']
        choices = item['choices']
        choices_sent = f"Following are the options:"
        for choice in choices:
            choices_sent += f" {choice},"
        question = pre_question(question + ' ' + choices_sent[:-1])  # Remove trailing comma
        # answer = item['answer']
        print(question)

        return image, question


class scienceqa_dataset(Dataset):
    def __init__(self, transform, split="train"):
        self.split = split  
        self.ds = load_dataset("derek-thomas/ScienceQA")      

        self.transform = transform
        self.vqa_root = '/Users/amitpadegal/Desktop/Capstone/vqa_sample'
        
        self.annotation = json.load(open('/Users/amitpadegal/Desktop/Capstone/foildataset_10k_modified.json','r'))
        self.annotation = self.annotation['annotations']
        
        # if split=='train':
        #     urls = {'vqa_train':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_train.json',
        #             'vg_qa':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vg_qa.json'}
        
        #     self.annotation = []
        #     for f in train_files:
        #         download_url(urls[f],ann_root)
        #         self.annotation += json.load(open(os.path.join(ann_root,'%s.json'%f),'r'))
        # else:
        #     download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_val.json',ann_root)
        #     self.annotation = json.load(open(os.path.join(ann_root,'vqa_val.json'),'r'))    
            
            # download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/answer_list.json',ann_root)
            # self.answer_list = json.load(open(os.path.join(ann_root,'answer_list.json'),'r'))    
                
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        image_path = self.ds[index]['image']
        print(type(image_path))
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)  
        # add = 'Answer if this caption matches the image: '      
        question = self.ds[index]['question']  
        options = f"Following are the options: {self.ds[index]["choices"][0]},{self.ds[index]["choices"][1]},{self.ds[index]["choices"][2]},{self.ds[index]["choices"][3]}"
        
        if self.split == 'test':
            # orig = pre_question(add + ann['original_caption'])
            print(question, options)
            question = pre_question(question + ' ' + options)        
            return image, question


        # elif self.split=='train':                       
            
        #     question = pre_question(ann['question'])        
            
        #     if ann['dataset']=='vqa':               
        #         answer_weight = {}
        #         for answer in ann['answer']:
        #             if answer in answer_weight.keys():
        #                 answer_weight[answer] += 1/len(ann['answer'])
        #             else:
        #                 answer_weight[answer] = 1/len(ann['answer'])

        #         answers = list(answer_weight.keys())
        #         weights = list(answer_weight.values())

        #     elif ann['dataset']=='vg':
        #         answers = [ann['answer']]
        #         weights = [0.2]  

        #     return image, question, answers, weights
        
            