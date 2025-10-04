from med import BertConfig
from nlvr_encoder import BertModel
from vit import interpolate_pos_embed
from blip import create_vit, init_tokenizer, is_url

from timm.models.hub import download_cached_file

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from rus import *


class BLIP_NLVR(nn.Module):
    def __init__(self,                 
                 med_config = 'med_config.json',  
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                   
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False) 
                    
        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)
                )  

    def forward(self, image, text, targets,file, train=True):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        image0_embeds, image1_embeds = torch.split(image_embeds,targets.size(0))  
        # print("image0",image0_embeds.size(), "image1",image1_embeds.size())  

        text = self.tokenizer(text, padding='longest', return_tensors="pt").to(image.device) 
        text.input_ids[:,0] = self.tokenizer.enc_token_id        

        output, question_emb = self.text_encoder(text.input_ids, 
                                   attention_mask = text.attention_mask, 
                                   encoder_hidden_states = [image0_embeds,image1_embeds],
                                   encoder_attention_mask = [image_atts[:image0_embeds.size(0)],
                                                             image_atts[image0_embeds.size(0):]],        
                                   return_dict = True,
                                  )  
        # print("question", question.size())
        # hidden_state = output.last_hidden_state[:,0,:]  

        hidden_state = output.last_hidden_state
        image0_tmp,image1_tmp, question_tmp, question_states_tmp = image0_embeds.squeeze(0), image1_embeds.squeeze(0), question_emb.squeeze(0), hidden_state.squeeze(0)
        image0_tmp = cluster_embeddings(image0_tmp, question_tmp.shape[0])
        image1_tmp = cluster_embeddings(image1_tmp, question_tmp.shape[0])
        self.n_components = (question_tmp.shape[0]//2) + 1
        print(image0_tmp.shape, image1_tmp.shape, question_tmp.shape, question_states_tmp.shape)
        # self.n_components = 3
        print(self.n_components)
        kmeans_im0, data_im0 = clustering(image0_tmp, pca=True, n_components=self.n_components, n_clusters=self.n_components)
        kmeans_im1, data_im1 = clustering(image1_tmp, pca=True, n_components=self.n_components, n_clusters=self.n_components)
        kmeans_txt, data_txt = clustering(question_tmp, pca=True, n_components=self.n_components, n_clusters=self.n_components)
        kmeans_out, data_out = clustering(question_states_tmp, pca=True, n_components=self.n_components, n_clusters=self.n_components)
        print(kmeans_im0.size, kmeans_txt.size, kmeans_out.size)
        kmeans_im0,kmeans_im1, kmeans_txt, kmeans_out = kmeans_im0.reshape(-1, 1), kmeans_im1.reshape(-1, 1), kmeans_txt.reshape(-1, 1), kmeans_out.reshape(-1, 1)
        P0, maps = convert_data_to_distribution(kmeans_im0, kmeans_txt, kmeans_out)
        res0 = get_measure(P0, file + 'image0.json')
        P1, maps = convert_data_to_distribution(kmeans_im1, kmeans_txt, kmeans_out)
        res1 = get_measure(P1, file + 'image1.json')
        # print("hidden_state", hidden_state.size())      
        prediction = self.cls_head(hidden_state)

        if train:            
            loss = F.cross_entropy(prediction, targets)   
            return loss
        else:
            return prediction, [res0, res1]
    
def blip_nlvr(pretrained='',**kwargs):
    model = BLIP_NLVR(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model  

        
def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    
    for key in list(state_dict.keys()):
        if 'crossattention.self.' in key:
            new_key0 = key.replace('self','self0')
            new_key1 = key.replace('self','self1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]
        elif 'crossattention.output.dense.' in key:
            new_key0 = key.replace('dense','dense0')
            new_key1 = key.replace('dense','dense1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]  
                
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg

def clustering(X, pca=False, n_clusters=10, n_components=5):
    # print(X.shape)
    if not isinstance(X, np.ndarray):
        X = X.detach().numpy()
    X = np.nan_to_num(X)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0],-1)
    if pca:
        # print(np.any(np.isnan(X)), np.all(np.isfinite(X)))
        X = normalize(X)
        X = PCA(n_components=n_components).fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    return kmeans.labels_, X

def cluster_embeddings(embeddings, target_samples):
    """
    Reduce the number of samples in embeddings using K-Means clustering.
    
    Parameters:
        embeddings (np.ndarray): Input embeddings of shape (num_samples, num_features).
        target_samples (int): Desired number of samples after clustering.

    Returns:
        np.ndarray: Cluster centroids representing reduced embeddings.
    """
    
    num_samples = embeddings.shape[0]
    if num_samples <= target_samples:
        return embeddings  # No clustering needed
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=target_samples, random_state=42, n_init=10)
    embeddings = embeddings.detach().numpy()
    kmeans.fit(embeddings)
    
    return kmeans.cluster_centers_  # Use centroids as reduced samples