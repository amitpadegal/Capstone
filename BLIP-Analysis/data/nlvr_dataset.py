import os
import json
import random

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption
from datasets import load_dataset
import dask.dataframe as dd
import io

class nlvr_dataset(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images 
        ann_root (string): directory to store the annotation file
        split (string): train, val or test
        '''
        # urls = {'train':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nlvr_train.json',
        #         'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nlvr_dev.json',
        #         'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nlvr_test.json'}
        # filenames = {'train':'nlvr_train.json','val':'nlvr_dev.json','test':'nlvr_test.json'}
        
        # download_url(urls[split],ann_root)
        # self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        
        # self.transform = transform
        # self.image_root = image_root
        files = [f"hf://datasets/TIGER-Lab/NLVR2/nlvr2/test-{i:05d}-of-00084.parquet" for i in range(13)] 
        self.df = dd.read_parquet(files).compute()
        # self.df = dd.read_parquet("hf://datasets/TIGER-Lab/NLVR2/nlvr2/test-*.parquet")
        # self.ds = load_dataset("TIGER-Lab/NLVR2")[split]
        self.transform = transform

        
    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, index):    
        row = self.df.iloc[index]
        images = row['images']
        # print(type(images))
        # print(images[0].keys())
        img_bytes = images[0]['bytes']
        image0 = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_bytes = images[1]['bytes']
        image1 = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # image0 = images[0].convert('RGB')
        image0 = self.transform(image0)
        # image1 = images[1].convert('RGB')
        image1 = self.transform(image1)
        
        # image0_path = os.path.join(self.image_root,ann['images'][0])        
        # image0 = Image.open(image0_path).convert('RGB')   
        # image0 = self.transform(image0)   
        
        # image1_path = os.path.join(self.image_root,ann['images'][1])              
        # image1 = Image.open(image1_path).convert('RGB')     
        # image1 = self.transform(image1)          
        # print(row['question'])
        sentence = pre_caption(row['question'], 40)
        # print(sentence)
        
        if row['answer']=='A':
            label = 1
        else:
            label = 0
            
        words = sentence.split(' ')
        
        if 'left' not in words and 'right' not in words:
            if random.random()<0.5:
                return image0, image1, sentence, label
            else:
                return image1, image0, sentence, label
        else:
            if random.random()<0.5:
                return image0, image1, sentence, label
            else:
                new_words = []
                for word in words:
                    if word=='left':
                        new_words.append('right')
                    elif word=='right':
                        new_words.append('left')        
                    else:
                        new_words.append(word)                    
                        
                sentence = ' '.join(new_words)
                return image1, image0, sentence, label
            
            
        