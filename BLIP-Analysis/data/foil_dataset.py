import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from data.utils import pre_question

from torchvision.datasets.utils import download_url

class foil_dataset(Dataset):
    def __init__(self, transform, split="train"):
        self.split = split        

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
        
        ann = self.annotation[index]
        foil_id = ann['foil_id']
        # temp = ann['imagePath'].split('/')[1]
        # if ann['dataset']=='vqa':
        #     image_path = os.path.join(self.vqa_root,ann['image'])    
        # elif ann['dataset']=='vg':
        #     image_path = os.path.join(self.vg_root,ann['image'])  
        image_path = os.path.join(self.vqa_root,ann['image_path'])
            
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)  
        add = 'Answer if this caption matches the image: '        
        
        if self.split == 'test':
            orig = pre_question(add + ann['original_caption'])   
            foil = pre_question(add + ann['foil_caption'])   
            random = pre_question(add + ann['random_caption'])               
            return image, orig, foil, random, foil_id


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
        
            