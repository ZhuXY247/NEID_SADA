import heapq
import math
from pathlib import Path
import numpy as np
import pandas as pd
import random
import os
import torch.nn.functional as F
from datasets import load_dataset,Dataset,DatasetDict
import torch.nn as nn
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers import get_scheduler
from transformers import  BertForNextSentencePrediction
from transformers import pipeline
from datasets import load_metric
from dataloader import Data
from model import CLBert, NoiseGenerator
from utils.build_ml import seed_everything, get_whole_word_attributions
from utils.tools import set_seed, view_generator, TeacherWrapper
import matplotlib.pyplot as plt
import umap
from time import time
from utils.sequence_classification import TripletSequenceClassificationExplainer, SequenceClassificationExplainer

class NoiseManager:
    def __init__(self, args, data):
        set_seed(args.seed)
        self.args = args
        self.n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = data.num_labels
        self.student_model = NoiseGenerator(hidden_size=768, device=self.device, dropout_prob=args.ratio).to(self.device)
        self.teacher_model = CLBert(args.bert_model, device=self.device, num_labels=self.num_labels)


        if self.n_gpu > 1:
            self.teacher_model = nn.DataParallel(self.teacher_model)
            self.student_model = nn.DataParallel(self.student_model)

        ### 重要！
        self.optimizer = AdamW([
            # {'params': self.teacher_model.parameters()},
            {'params': self.student_model.parameters(), 'lr': args.lr }
        ], lr = args.lr)
        
        num_training_steps = len(data.train_distillation_dataloader) * args.num_distillate_epochs
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.viz_count = 0

    def cash_teacher_masks(self, data, args, ratio, with_abs, with_pos, is_continuous):
        """
        核心加速逻辑：离线生成 Teacher Mask
        """
        cache_name = (
            f"mask_cache_{args.internal_dataset}_"
            f"l{args.labeled_ratio}_k{args.known_cls_ratio}_"
            f"inp{args.input_strategy}_spk{int(args.with_speaker)}_"
            f"r{ratio}_pos{with_pos}_seed{args.seed}.pt"
        )
        cache_path = os.path.join(args.save_results_path, cache_name)

        if os.path.exists(cache_path):
            return torch.load(cache_path)
        os.makedirs(args.save_results_path, exist_ok=True)
        cached_masks = {}
        dataloader = data.train_distillation_dataloader
        self.teacher_model.eval()
        for batch in tqdm(dataloader, desc="Caching Masks"):
            batch_data = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'special_tokens_mask': batch[4],
                'text': self.tokenizer.batch_decode(batch[0], skip_special_tokens=True)
            }
            indices = batch[5]
            with torch.no_grad():
                masks = self.generate_teacher_mask(
                    batch=batch_data,
                    top_n_ratio=ratio,
                    with_abs=with_abs,
                    with_pos=with_pos,
                    is_continuous=is_continuous
                )
            for i, idx in enumerate(indices):
                cached_masks[idx.item()] = masks[i].cpu()
        torch.save(cached_masks, cache_path)
        return cached_masks

    def generate_teacher_mask(self, batch, top_n_ratio, with_abs=False, with_pos=False, is_continuous=False):
        '''
        choose the top_n_ratio relevant tokens
        with_abs : use the absolute value of the contribution
        with_pos : use part-of-speech tagging to keep only nouns, adjectives, adverbs
        is_continuous: Keep consecutive words , if is_continuous == True the with_pos == False
        '''
        self.teacher_model.eval()
        attention_mask = batch["attention_mask"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        special_tokens_mask = batch["special_tokens_mask"].to(self.device)

        if is_continuous:
            with_pos = False  # if is_continuous == True the with_pos == False

        raw_model = self.teacher_model.module if isinstance(self.teacher_model, nn.DataParallel) else self.teacher_model
        wrapped_teacher = TeacherWrapper(raw_model)
        if self.args.input_strategy == "CONTEXT":
            multiclass_explainer = TripletSequenceClassificationExplainer(model=wrapped_teacher,
                                                                          tokenizer=self.tokenizer)
        else:
            multiclass_explainer = SequenceClassificationExplainer(model=wrapped_teacher, tokenizer=self.tokenizer)
        batch_size, seq_len = input_ids.shape
        teacher_masks = torch.zeros((batch_size, seq_len, 1)).to(self.device)
        sep_token_id = self.tokenizer.sep_token_id
        for i in range(batch_size):
            torch.cuda.empty_cache()
            with torch.enable_grad():
                curr_input_ids = input_ids[i]
                if self.args.input_strategy == "CONTEXT":
                    sep_indices = (curr_input_ids == sep_token_id).nonzero(as_tuple=True)[0]
                    prev_ids = curr_input_ids[1: sep_indices[0]]
                    curr_ids = curr_input_ids[sep_indices[0] + 1: sep_indices[1]]
                    next_ids = curr_input_ids[sep_indices[1] + 1: sep_indices[2]]

                    text_prev = self.tokenizer.decode(prev_ids, skip_special_tokens=True)
                    text_curr = self.tokenizer.decode(curr_ids, skip_special_tokens=True)
                    text_next = self.tokenizer.decode(next_ids, skip_special_tokens=True)

                    word_attributions = np.array(multiclass_explainer(
                        text_prev=text_prev,
                        text_curr=text_curr,
                        text_next=text_next
                    ))
                else:
                    sentence = self.tokenizer.decode(curr_input_ids, skip_special_tokens=True)
                    word_attributions = np.array(multiclass_explainer(text=sentence))

                if with_abs:
                    word_attributions[:, 1] = [abs(eval(attribution[1])) for attribution in word_attributions]
            #   else:
                #   word_attributions[:,1] = [eval(attribution[1]) for attribution in word_attributions]

                # get whole word attributions
                word_attributions = get_whole_word_attributions(word_attributions, with_pos)

                if self.args.save_model_path and self.viz_count < 10:
                    viz_dir = os.path.join(self.args.save_model_path, "attribution_viz")
                    if not os.path.exists(viz_dir):
                        os.makedirs(viz_dir)
                    file_name = f"viz_sample_{self.viz_count}.html"
                    save_path = os.path.join(viz_dir, file_name)
                    try:
                        multiclass_explainer.visualize(html_filepath=save_path)
                        # print(f"Saved attribution visualization to {save_path}")
                    except Exception as e:
                        print(f"Visualization failed for sample {self.viz_count}: {e}")

                    self.viz_count += 1
                # print(sentence)
                # print(word_attributions)
                if word_attributions.ndim < 2 or len(word_attributions) == 0:
                    teacher_masks[i, :, 0] = torch.zeros(seq_len).to(self.device)
                    continue

                top_N = math.ceil((len(word_attributions) * (1 - top_n_ratio)))

                if not is_continuous:
                    tmp = zip(range(len(word_attributions)), word_attributions[:, 1].astype(float))
                    top_index = heapq.nlargest(top_N, tmp, key=lambda x: x[1])
                    top_index = [top[0] for top in top_index]
                else:
                    # [whole_word,att,word_idx]
                    max_attr = np.sum(word_attributions[:, 1][0:top_N])
                    max_idx = 0
                    for begin_idx in range(1, len(word_attributions) - top_N, 1):
                        if np.sum(word_attributions[:, 1][begin_idx:begin_idx + top_N]) > max_attr:
                            # print(len(word_attributions[:,1][begin_idx:begin_idx + top_N]))
                            max_attr = np.sum(word_attributions[:, 1][begin_idx:begin_idx + top_N])
                            max_idx = begin_idx

                    top_index = list(range(max_idx, max_idx + top_N))
                current_mask = torch.ones(seq_len).to(self.device)

                # print(multiclass_explainer.predicted_class_name)
                # multiclass_explainer.visualize()
                for idx in top_index:

                    token_pos_list = word_attributions[idx][2]
                    for pos in token_pos_list:
                        if pos < seq_len:
                            current_mask[pos] = 0.0

                if special_tokens_mask is not None:
                    # current_mask = torch.max(current_mask, special_tokens_mask[i].float())
                    current_mask = current_mask * (1 - special_tokens_mask[i].float())
                else:
                    current_mask[0] = 0.0
                    input_id_len = torch.sum(attention_mask[i]).item()
                    current_mask[input_id_len - 1:] = 0.0

                teacher_masks[i, :, 0] = current_mask

        return teacher_masks


    def Mask_BERT_with_ratio(self, args, data):

            num_epochs = args.num_distillate_epochs

            with_pos = args.with_pos
            with_abs = args.with_abs
            is_continuous = args.is_continuous

            ratio = args.ratio
            random_states = getattr(args, 'random_states', [args.seed])

            cached_masks = self.cash_teacher_masks(
                data, args, args.ratio, args.with_abs, args.with_pos, args.is_continuous
            )

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, padding=True, truncation=True, model_max_length = args.internal_max_seq_length)

            for k, random_state in enumerate(random_states):
                seed_everything(random_state)
                min_loss = 99999
                best_student_state = None
                wait_patient = 20
                wait = 0
                student_net = self.student_model.module if hasattr(self.student_model, 'module') else self.student_model
                for epoch in range(num_epochs):
                    self.teacher_model.train()
                    self.student_model.train()

                    tr_loss = 0
                    nb_tr_steps = 0

                    train_dataloader = data.train_distillation_dataloader
                    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

                    for step, batch in enumerate(pbar):
                        if isinstance(batch, dict):
                            batch = {k: v for k, v in batch.items()}
                        else:
                            batch = {
                                'input_ids': batch[0],
                                'attention_mask': batch[1],
                                'token_type_ids': batch[2],
                                'labels': batch[3],
                                'special_tokens_mask': batch[4],
                                'indices': batch[5]
                            }
                        batch['text'] = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        special_tokens_mask = batch["special_tokens_mask"].to(self.device)
                        indices = batch['indices']
                        teacher_mask_list = [cached_masks[idx.item()] for idx in indices]
                        teacher_mask = torch.stack(teacher_mask_list).to(self.device)

                        teacher_obj = self.teacher_model.module if isinstance(self.teacher_model,
                                                                            nn.DataParallel) else self.teacher_model
                        backbone = teacher_obj.backbone
                        base_model = getattr(backbone, backbone.base_model_prefix, backbone)
                        raw_embeddings = base_model.embeddings.word_embeddings(input_ids)

                        student_mask = self.student_model(
                            raw_embeddings,
                            attention_mask=attention_mask,
                            special_tokens_mask=special_tokens_mask,
                            temperature=0.3,
                            mask_threshold=args.mask_threshold
                        )

                        loss_distill = F.binary_cross_entropy(student_mask, teacher_mask)
                        loss = loss_distill
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        tr_loss += loss.item()
                        nb_tr_steps += 1

                        pbar.set_postfix({'loss': tr_loss / nb_tr_steps})
                    epoch_avg_loss = tr_loss / nb_tr_steps
                    print(f"Epoch {epoch + 1} Loss: {epoch_avg_loss:.5f} (Best: {min_loss:.5f})")

                    if epoch_avg_loss < min_loss:
                        print(f" Loss decreased ({min_loss:.5f} -> {epoch_avg_loss:.5f}). Saving model...")
                        min_loss = epoch_avg_loss
                        best_student_state = {k: v.cpu().clone() for k, v in student_net.state_dict().items()}
                        wait = 0
                    else:
                        wait += 1
                        if wait >= wait_patient:
                            break

                if best_student_state is not None:
                    student_net.load_state_dict(best_student_state)
                else:
                    print("[Warning] No best model found (Training might have failed).")
