'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
from ruamel.yaml import YAML
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
from test import calculate_modality_dependence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from vqa import blip_vqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result


def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    image_grad_accumulator, text_grad_accumulator, tot_batches = 0, 0, 0
    
    for i,(image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)      
        model.zero_grad()
        loss, image_emb, question_emb = model(image, question, answer, train=True, n=n, weights=weights)   

        image_emb.requires_grad_(True)
        question_emb.requires_grad_(True)

        image_emb.retain_grad()  # Retain gradient for image_emb
        question_emb.retain_grad()  # Retain gradient for question_emb
        # optimizer.zero_grad()
        loss.backward()
        image_grad = image_emb.grad
        text_grad = question_emb.grad

        image_grad_mean_square = torch.mean(image_grad ** 2).item()
        text_grad_mean_square = torch.mean(text_grad ** 2).item()
        
        
        # Accumulate the mean square gradients
        image_grad_accumulator += image_grad_mean_square
        text_grad_accumulator += text_grad_mean_square
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, (image_grad_accumulator, text_grad_accumulator, tot_batches)


@torch.no_grad()
def evaluation(model, data_loader, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
    bias1, bias2 = 0, 0
    
    if config['inference']=='rank':   
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        answer_candidates.input_ids[:,0] = model.tokenizer.bos_token_id
        
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             

        if config['inference']=='generate':
            answers, res = model(image, question, train=False, inference='generate') 

            dep1, dep2 = calculate_modality_dependence(res)
            bias1 += dep1
            bias2 += dep2
            
            for answer, ques_id in zip(answers, question_id):
                ques_id = int(ques_id.item())       
                result.append({"question_id":ques_id, "answer":answer})             
            
        elif config['inference']=='rank':    
            answer_ids = model(image, question, answer_candidates, train=False, inference='rank', k_test=config['k_test'])      

            for ques_id, answer_id in zip(question_id, answer_ids):
                result.append({"question_id":int(ques_id.item()), "answer":answer_list[answer_id]}) 
        
        if n == 1000:
            print("Bias 1:", bias1/(n+1))
            print("Bias 2:", bias2/(n+1))
            data = {
    "bias1": bias1 / (n + 1),
    "bias2": bias2 / (n + 1)
}

            with open("blip_vqa.jsonl", 'a') as f:
                f.write(json.dumps(data) + "\n")
            break

    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Dataset #### 
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config)   
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]
    
    train_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_test']],
                                              num_workers=[4,4],is_trains=[True, False], 
                                              collate_fns=[vqa_collate_fn,None]) 
    #### Model #### 
    print("Creating model")
    model = blip_vqa(pretrained=config['pretrained'], image_size=config['image_size'], 
                       vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    # Check the requires_grad attribute for all model parameters
    # for name, param in model.named_parameters():
    #     if 'embedding' in name:  # Check if it's part of the embedding layer
    #         print(f"{name}: {param.requires_grad}")

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0 
    epoch_image_grad_accumulator = []
    epoch_text_grad_accumulator = []
    gradient_contributions = []
       
    print("Start training")
    # print(args.evaluate)
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                
            train_stats, (image_grad_accumulator, text_grad_accumulator, total_batches) = train(model, train_loader, optimizer, epoch, device)
            image_grad_magnitude = image_grad_accumulator / total_batches
            text_grad_magnitude = text_grad_accumulator / total_batches

            total_magnitude = image_grad_magnitude + text_grad_magnitude
            a = image_grad_magnitude / total_magnitude
            b = text_grad_magnitude / total_magnitude

            # Log the contributions for visualization
            gradient_contributions.append((a, b))
            print(f"Epoch {epoch+1}/{config['max_epoch']}: a (image contribution) = {a}, b (text contribution) = {b}")
            
            # Store the epoch-level accumulators for later analysis
            epoch_image_grad_accumulator.append(image_grad_magnitude)
            epoch_text_grad_accumulator.append(text_grad_magnitude) 

        else:         
            break        
        
        if utils.is_main_process():     
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")                        
                    
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  

        dist.barrier()     
    # epochs = list(range(1, config['max_epoch'] + 1))
    # a_values, b_values = zip(*gradient_contributions)

    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs, a_values, label="Image Contribution (a)", marker='o')
    # plt.plot(epochs, b_values, label="Text Contribution (b)", marker='o')
    # plt.title("Image and Text Contributions Over Epochs")
    # plt.xlabel("Epochs")
    # plt.ylabel("Contribution")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    vqa_result = evaluation(model_without_ddp, test_loader, device, config)        
    result_file = save_result(vqa_result, args.result_dir, 'vqa_result')  
                      
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
            

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