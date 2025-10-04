import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
from med import BertConfig, BertModel, BertLMHeadModel
from blip import create_vit, init_tokenizer, load_checkpoint
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from rus import *

class BLIP_VQA(nn.Module):
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
        self.n_components = 10
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()  
        
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False) 
        
        decoder_config = BertConfig.from_json_file(med_config)        
        self.text_decoder = BertLMHeadModel(config=decoder_config)          


    def forward(self, image, question, file, answer=None, n=None, weights=None, train=False, inference='generate', k_test=128):
        # with open('temp_data.txt', 'a') as f:  # 'a' mode to append
        #     f.write("Input Image Shape: {}\n".format(image.shape))
        #     f.write("Input Question: {}\n".format(question))
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        question = self.tokenizer(question, padding='longest', truncation=True, max_length=35, 
                                  return_tensors="pt").to(image.device) 
        question.input_ids[:,0] = self.tokenizer.enc_token_id
        
        if train:               
            '''
            n: number of answers for each question
            weights: weight for each answer
            '''                     
            answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device) 
            answer.input_ids[:,0] = self.tokenizer.bos_token_id
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      

            question_output, question_emb = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = True)    


            question_states = []                
            question_atts = []  
            for b, n in enumerate(n):
                question_states += [question_output.last_hidden_state[b]]*n
                question_atts += [question.attention_mask[b]]*n                
            question_states = torch.stack(question_states,0)    
            question_atts = torch.stack(question_atts,0)     

            answer_output = self.text_decoder(answer.input_ids, 
                                              attention_mask = answer.attention_mask, 
                                              encoder_hidden_states = question_states,
                                              encoder_attention_mask = question_atts,                  
                                              labels = answer_targets,
                                              return_dict = True,   
                                              reduction = 'none',
                                             )      
            
            loss = weights * answer_output.loss
            loss = loss.sum()/image.size(0)

            return loss, image_embeds, question_emb
            

        else: 
            question_output, question_emb = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True) 
            if inference=='generate':
                num_beams = 5

                # question_states = question_output.last_hidden_state.repeat_interleave(num_beams,dim=0)
                question_states = question_output.last_hidden_state
                image_tmp, question_tmp, question_states_tmp = image_embeds.squeeze(0), question_emb.squeeze(0), question_states.squeeze(0)
                image_tmp = cluster_embeddings(image_tmp, question_tmp.shape[0])
                self.n_components = (question_tmp.shape[0]//2) + 1
                # self.n_components = 3
                print(self.n_components)
                kmeans_im, data_im = clustering(image_tmp, pca=True, n_components=self.n_components, n_clusters=self.n_components)
                kmeans_txt, data_txt = clustering(question_tmp, pca=True, n_components=self.n_components, n_clusters=self.n_components)
                kmeans_out, data_out = clustering(question_states_tmp, pca=True, n_components=self.n_components, n_clusters=self.n_components)
                print(kmeans_im.size, kmeans_txt.size, kmeans_out.size)
                kmeans_im, kmeans_txt, kmeans_out = kmeans_im.reshape(-1, 1), kmeans_txt.reshape(-1, 1), kmeans_out.reshape(-1, 1)
                P, maps = convert_data_to_distribution(kmeans_im, kmeans_txt, kmeans_out)
                res = get_measure(P, file)
                # print(res)

                question_atts = torch.ones(question_states.size()[:-1],dtype=torch.long).to(question_states.device)
                model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}
                
                bos_ids = torch.full((image.size(0),1),fill_value=self.tokenizer.bos_token_id,device=image.device)

                outputs = self.text_decoder.generate(input_ids=bos_ids,
                                                     max_length=10,
                                                     min_length=1,
                                                     num_beams=num_beams,
                                                     eos_token_id=self.tokenizer.sep_token_id,
                                                     pad_token_id=self.tokenizer.pad_token_id, 
                                                     **model_kwargs)
                
                answers = []    
                for output in outputs:
                    answer = self.tokenizer.decode(output, skip_special_tokens=True)    
                    answers.append(answer)
                return answers, res
            
            elif inference=='rank':
                max_ids = self.rank_answer(question_output.last_hidden_state, question.attention_mask, 
                                           answer.input_ids, answer.attention_mask, k_test) 
                return max_ids
 
                
                
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none')   
        
        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques,k)

        max_topk_ids = log_probs_sum.argmax(dim=1) 
        max_ids = topk_ids[max_topk_ids>=0,max_topk_ids]

        return max_ids
    
    
def blip_vqa(pretrained='',**kwargs):
    model = BLIP_VQA(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
#         assert(len(msg.missing_keys)==0)
    return model  


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    
        
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