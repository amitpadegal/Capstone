from datasets import load_dataset

ds = load_dataset("vikhyatk/gqa-val")
from torch.utils.data import DataLoader
import torch
import clip
from PIL import Image
import utils
import numpy as np
import argparse
from ruamel.yaml import YAML
from pathlib import Path
import os
import random
import torch.backends.cudnn as cudnn
import os
from data import create_dataset, create_loader
from data.vqa_dataset import vqa_collate_fn
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from rus import *

def calculate_modality_dependence(res):
    redundancy, uniqueness1, uniqueness2, synergy = res['redundancy'], res['unique1'], res['unique2'], res['synergy']
    total = redundancy + uniqueness1 + uniqueness2 + synergy

    dependence1 = uniqueness1 / total
    dependence2 = uniqueness2 / total
    tot = dependence1 + dependence2
    dep1 = dependence1/tot
    dep2 = dependence2/tot

    return dep1, dep2

class CustomCollateFn:
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, batch):

        img_tensor = self.preprocess(batch[0]['image'])
        questions = tuple(d['question'] for d in batch[0]['qa'])
        replicated_imgs = img_tensor.repeat(len(questions), 1, 1, 1)
        return {
            'image': replicated_imgs,
            'question_tuples': questions
        }

def evaluation(model, preprocess, data_loader, device):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
    tot_1, tot_2 = 0, 0
    for n, sample in enumerate(data_loader):        
        # image = image.to(device,non_blocking=True)             
        # question = question.to(device,non_blocking=True)
        # print(question)
        
        image = sample['image'].to(device)  # Already preprocessed and stacked
        question = sample['question_tuples']  # List of 32 tuples
        # image = image.to(device)
        # print(question)
        text = clip.tokenize(question).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            print(image_features.shape, text_features.shape)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            output_features = image_features * text_features
            image_tmp, question_tmp, question_states_tmp = image_features, text_features.squeeze(0), output_features.squeeze(0)
            image_tmp = cluster_embeddings(image_tmp, question_tmp.shape[0])
            kmeans_im, data_im = clustering(image_tmp, pca=True, n_components=10, n_clusters=10)
            kmeans_txt, data_txt = clustering(question_tmp, pca=True, n_components=10, n_clusters=10)
            kmeans_out, data_out = clustering(question_states_tmp, pca=True, n_components=10, n_clusters=10)
            print(kmeans_im.size, kmeans_txt.size, kmeans_out.size)
            kmeans_im, kmeans_txt, kmeans_out = kmeans_im.reshape(-1, 1), kmeans_txt.reshape(-1, 1), kmeans_out.reshape(-1, 1)
            P, maps = convert_data_to_distribution(kmeans_im, kmeans_txt, kmeans_out)
            res = get_measure(P)
            dep1, dep2 = calculate_modality_dependence(res)
            tot_1 += dep1
            tot_2 += dep2

        if n == 0:
            print("Bias 1:", tot_1/(n+1))
            print("Bias 2:", tot_2/(n+1))
            break

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

def main(args, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    #### Dataset #### 
    # print("Creating vqa datasets")
    # datasets = create_dataset('vqa', config, preprocess)   

    collate_fn = CustomCollateFn(preprocess)

    samplers = [None, None]

    test_loader = DataLoader(
    ds['test'],
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,  # <- define this
    num_workers=4
)

    
    evaluation(model, preprocess, test_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='vqa.yaml') 
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--result_dir', default='result')
    parser.add_argument('--evaluate', default=True, action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = "cuda:0"
    else:
        args.device = "cpu"

    yaml = YAML()
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)