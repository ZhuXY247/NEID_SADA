import math

from utils.tools import *
from utils.contrastive import SupConLoss
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoConfig, AutoModelForSequenceClassification
import torch.fft

class NoiseGenerator(nn.Module):
    def __init__(self, hidden_size, device=None, dropout_prob=0.1) -> None:
        super(NoiseGenerator, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
                )
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化策略：Warm Start
        让初始的 Mask 偏向于 1 (完全保留)，防止模型初期崩溃。
        """
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def gumbel_sigmoid(self, logits, temperature=0.1):
        """保持梯度传导的采样"""
        if self.training:
            eps = 1e-10
            u1 = torch.rand_like(logits)
            noise = torch.log(u1 + eps) - torch.log(1 - u1 + eps)
            gate_inputs = (noise + logits) / temperature
            return torch.sigmoid(gate_inputs)
        else:
            return torch.sigmoid(logits)

    def forward(self, hidden_states, attention_mask=None, special_tokens_mask=None, temperature=0.1,
                mask_threshold=0.5):
        logits = self.net(hidden_states)
        if self.training:
            probs = self.gumbel_sigmoid(logits, temperature=temperature)
        else:
            probs = torch.sigmoid(logits / temperature)
        probs = probs.squeeze(-1) if probs.dim() == 3 else probs
        probs = probs * attention_mask.to(probs.device)
        if special_tokens_mask is not None:
            probs = probs * (1 - special_tokens_mask.to(probs.device))
        if not self.training:
            if special_tokens_mask is not None:
                valid_mask = attention_mask.to(probs.device) * (1 - special_tokens_mask.to(probs.device))
            else:
                valid_mask = attention_mask.to(probs.device)
            valid_len = valid_mask.sum(dim=1)
            k_sample = (valid_len.float() * mask_threshold).ceil().long()
            k_sample = torch.clamp(k_sample, min=1)
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            batch_indices = torch.arange(probs.size(0), device=probs.device)
            col_indices = (k_sample - 1).clamp(min=0, max=probs.size(1) - 1)
            threshold = sorted_probs[batch_indices, col_indices].unsqueeze(1)
            gate = (probs >= threshold).float() * (probs > 0).float()
            probs = probs * gate
            probs = probs * attention_mask.to(probs.device)
            if special_tokens_mask is not None:
                probs = probs * (1 - special_tokens_mask.to(probs.device))
        probs = probs.unsqueeze(-1)
        return probs

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load_model(self, save_path):
        state_dict = torch.load(save_path, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)

class BertForModel(nn.Module):
    def __init__(self,model_name, num_labels, device=None, loss_weights=None):
        super(BertForModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.device = device
        self.loss_weights = loss_weights
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.classifier = nn.Linear(768, self.num_labels)

        self.dropout = nn.Dropout(0.1)
        self.backbone.to(self.device)
        self.classifier.to(self.device)

    def forward(self, X, output_hidden_states=False, output_attentions=False):
        """logits are not normalized by softmax in forward function"""
        outputs = self.backbone(**X, output_hidden_states=True)
        # extract last layer [CLS]
        CLSEmbedding = outputs.hidden_states[-1][:,0]
        CLSEmbedding = self.dropout(CLSEmbedding)
        logits = self.classifier(CLSEmbedding)
        output_dir = {"logits": logits}
        if output_hidden_states:
            output_dir["hidden_states"] = outputs.hidden_states[-1][:, 0]
        if output_attentions:
            output_dir["attentions"] = outputs.attention
        return output_dir

    def mlmForward(self, X, Y):
        outputs = self.backbone(**X, labels=Y)
        return outputs.loss

    def loss_ce(self, logits, Y):
        if self.loss_weights is not None:
            loss = nn.CrossEntropyLoss(weight=self.loss_weights)
        else:
            loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output

    def save_backbone(self, save_path):
        # torch.save(self.backbone.state_dict(), save_path)
        self.backbone.save_pretrained(save_path)
    
    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load_model(self, save_path):
        state_dict = torch.load(save_path, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)


class CLBert(nn.Module):
    def __init__(self,model_name, device, feat_dim=128, num_labels=2):
        super(CLBert, self).__init__()
        self.model_name = model_name
        self.device = device
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, feat_dim)
        )
        self.num_labels = num_labels
        self.classifier = nn.Linear(hidden_size, num_labels).to(self.device)
        self.backbone.to(self.device)
        self.head.to(device)

    def forward(self, X, output_hidden_states=False, output_attentions=False, output_logits=False):
        """logits are not normalized by softmax in forward function"""

        outputs = self.backbone(**X, output_hidden_states=True, output_attentions=True)
        cls_embed = outputs.hidden_states[-1][:,0]
        features = F.normalize(self.head(cls_embed), dim=1)
        output_dir = {"features": features, "logits": self.classifier(cls_embed)}
        if output_hidden_states:
            output_dir["hidden_states"] = cls_embed
        if output_attentions:
            output_dir["attentions"] = outputs.attentions
        return output_dir

    def loss_cl(self, embds, label=None, mask=None, temperature=0.07, base_temperature=0.07):
        """compute contrastive loss"""
        loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)
        output = loss(embds, labels=label, mask=mask)
        return output

    def save_backbone(self, save_path):
        # torch.save(self.backbone.state_dict(), save_path)
        self.backbone.save_pretrained(save_path)

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load_model(self, save_path):
        state_dict = torch.load(save_path, map_location=self.device)
        self.load_state_dict(state_dict, strict=False)


