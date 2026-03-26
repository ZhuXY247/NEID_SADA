import imp
import json 
import re
from pathlib import Path
from tkinter import NONE
from tkinter.messagebox import NO
import unicodedata
import numpy as np
import pandas as pd
import random
import os
from time import time

from datasets import load_dataset,Dataset,DatasetDict
import torch.nn as nn
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD,RMSprop#,AdamW
from transformers import AutoTokenizer
from transformers import get_scheduler
from transformers_interpret import SequenceClassificationExplainer
from sklearn.metrics import classification_report,accuracy_score
import itertools
import heapq
import nltk
import math
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
import umap

import matplotlib.pyplot as plt
from utils.adamW import AdamW


def set_args(new_args):
    global args
    args = new_args
    init()

def init():
    global tokenizer,device,max_length
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, padding=True, truncation=True,model_max_length = args.internal_max_seq_length)
    device = args.device
    max_length = args.internal_max_seq_length


def seed_everything(seed_value):

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def tokenize_function(example):
    return tokenizer(example["text"], truncation=True,max_length = args.internal_max_seq_length)

def collate_function(data):

    text = [d['text'] for d in data]
    labels = [d['labels'] for d in data]
    input_ids = [d['input_ids'] for d in data]
    token_type_ids = [d['token_type_ids'] for d in data]
    attention_mask = [d['attention_mask'] for d in data]

    
    
    max_len = max([len(input_id) for input_id in input_ids])

    input_ids = [input_id if len(input_id) == max_len else np.concatenate([input_id,[102] * (max_len - len(input_id))]) for input_id in input_ids ]
    token_type_ids = [token_type if len(token_type) == max_len else np.concatenate([token_type,[0] * (max_len - len(token_type))]) for token_type in token_type_ids ]
    attention_mask = [mask if len(mask) == max_len else np.concatenate([mask,[0] * (max_len - len(mask))]) for mask in attention_mask ]

    if 'sample_type' in data[0].keys():
        sample_type = [d['sample_type'] for d in data]

        batch = {'input_ids':torch.LongTensor(input_ids),
                'token_type_ids':torch.LongTensor(token_type_ids),
                'attention_mask':torch.LongTensor(attention_mask),
                'sample_type':torch.LongTensor(sample_type),
                'labels':torch.LongTensor(labels),
                'text':text
                }
    else:
        batch = {'input_ids':torch.LongTensor(input_ids),
            'token_type_ids':torch.LongTensor(token_type_ids),
            'attention_mask':torch.LongTensor(attention_mask),
            'labels':torch.LongTensor(labels),
            'text':text
            }

    return batch

def constrative_loss(feature,label,novel_label = None):
    pairwised_consine = torch.cosine_similarity(feature.double().unsqueeze(1),feature.unsqueeze(0),dim = -1)
    pairwised_consine = torch.exp(pairwised_consine)
    pairwised_consine = torch.triu(pairwised_consine,diagonal = 1)
    # print(pairwised_consine)

    denominator = torch.sum(pairwised_consine)

    numerator = pairwised_consine[0,0] 
    max_label = torch.max(label)
    for l in range(max_label + 1):
        l_idx = np.array(list(range(0,len(label))))[label.detach().cpu().numpy() == l]
        if len(l_idx) > 1: 
            pair_idx = list(itertools.combinations(l_idx, 2))
            # print(pair_idx)
            for idx in pair_idx:
                numerator = numerator + pairwised_consine[idx]
    # print(numerator,denominator)
    loss = - torch.log( (numerator + 1e-4) / (denominator + 1e-4) ) / len(label)
    
    return loss

def build_optimizer(model,lr,n_epoch,data_loader = None,with_scheduler = False,betas = (0.9,0.999),with_bc=True):
#     optimizer = RMSprop(model.parameters(), lr=lr,momentum = 0, weight_decay = 0.005)
    optimizer = AdamW(model.parameters(), lr=lr,betas=betas,with_bc = with_bc)
#     optimizer = AdamW(model.parameters(), lr=lr)
#     optimizer = AdamW(model.parameters(), lr=lr)
#     optimizer = SGD(model.parameters(), lr=lr, weight_decay = 0.005)

    if with_scheduler:
        assert data_loader!=None, print('params error!')
        # init lr
        num_training_steps = len(data_loader) * n_epoch 

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps= int(0.1 * num_training_steps),
            num_training_steps=num_training_steps,
        )
        return optimizer,lr_scheduler
    return optimizer,None

def train_one_epoch_without_mask(model,dataloader,optimizer,device,lr_scheduler=None):
    progress_bar = tqdm(range( len(dataloader)))
#     print('model training...')

    model.train()

    for batch in dataloader:
        text = batch.pop("text")
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        if type(lr_scheduler) != type(None):
            lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    return model

def test_without_mask(model,dataloader, device, cls = None, return_samples = False):
    acc = 0
    model.eval()
    y_true = []
    y_pred = []
    text_list = []
    for batch in tqdm(dataloader):
        
        text = batch.pop("text")
        text_list.extend(text)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        if type(cls) != None:
            tmp = logits[:,cls]
            logits = torch.ones_like(logits) * (-1e5)
            logits[:,cls] = tmp
            
        predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()

        if len(y_true) == 0:
            y_true = labels
            y_pred = predictions
        else:
            y_true = np.concatenate([y_true,labels])
            y_pred = np.concatenate([y_pred,predictions])

    if len(y_true) >0:
        # print(classification_report(y_true, y_pred,digits= 4))

        acc = accuracy_score(y_true, y_pred)

    if return_samples:
        text_samples = pd.DataFrame({'text':text_list,'predictions':y_pred,'labels':y_true})
        return text_samples,acc
    
    return acc

def get_whole_word_attributions(word_attributions, with_pos = False):
#     input:
#      ('[CLS]', 0.0),
#      ('trap', 0.027725534317037028),
#      ('##ez', -0.17850601948716388),
#      ('##e', 0.024500284336490763),
#      ('software', 0.23354358430515224),
#      ('[SEP]', 0.23354358430515224)

#       output:
#        ('trapeze', -0.042093400277878695, [0,1]),
#       ('software', 0.23354358430515224, [2,3]),
    sentence = []
    whole_word_attributions = []
    word_idxs = []
    i = -1
    while True:
        word_attribution = word_attributions[i]
        whole_word_attribution = []
        word_idx = []
        i = i + 1
        if i >= len(word_attributions):
            break
        whole_word = word_attributions[i][0]
        whole_word_attribution.append(word_attributions[i][1].astype(np.float))
        word_idx.append(i)

        while i + 1 < len(word_attributions) and word_attributions[i + 1][0].startswith('##'):
            i = i + 1
            whole_word = whole_word + word_attributions[i][0][2:]
            whole_word_attribution.append(word_attributions[i][1].astype(np.float))
            word_idx.append(i)

        sentence.append(whole_word)   
        
        whole_word_attributions.append(np.mean(whole_word_attribution))
        word_idxs.append(word_idx)
    
    assert word_idxs[-1][-1] < len(word_attributions),print('word_attributions error')
    
    result = []
    if with_pos:
        pos_tag = np.array(nltk.pos_tag(sentence[1:-1]))
        for whole_word,att,pos,word_idx in zip(sentence[1:-1],whole_word_attributions[1:-1],pos_tag[:,1],word_idxs[1:-1]):  #exclude [CLS] and [SEP]
            if pos in ['JJ','JJR','JJS','NN','NN','NNS','NNP','NNPS','RB','RBR','RBS','UH','VB','VBD','VBG','VBN','VBP','VBZ','WRB']:
                result.append([whole_word,att,word_idx])
    else:
        for whole_word,att,word_idx in zip(sentence[1:-1],whole_word_attributions[1:-1],word_idxs[1:-1]):  #exclude [CLS] and [SEP]
            result.append([whole_word,att,word_idx])
    result = np.array(result,dtype=object)
    
    return result

    
