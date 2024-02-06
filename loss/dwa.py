import os, sys
import torch
from torch import nn
from torch._tensor import Tensor 
import torch.nn.functional as F
import numpy as np

from .vanilla_clf_stable import VanillaClassifierStableV0

class DWAClassifierV0(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.epoch = 0
        self.train_loss_buffer = np.zeros([self.args.n_classes, self.args.epochs])
        self.sample_count = 0
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        losses = torch.zeros(self.args.n_classes).to(pred.device)
        
        B, C = tuple(pred.size())
        
        for cidx in range(C):
            c_pred = pred[target[:, cidx] == 1]
            c_target = target[target[:, cidx] == 1]

            c_loss = self.loss_fn(c_pred, c_target)

            losses[cidx] = c_loss

        if self.sample_count >= self.args.num_train_sample:
            self.train_loss_buffer[:, self.epoch] = losses.detach().clone().tolist()
            self.epoch += 1
            self.sample_count = 0
            
        else:
            self.sample_count += B
        
        if self.epoch > 1:
            w_i = torch.Tensor(self.train_loss_buffer[:, self.epoch-1] / self.train_loss_buffer[:, self.epoch-2]).to(pred.device)
            batch_weight = F.softmax(w_i/self.args.gumbel_tau, dim=-1)
        else:
            batch_weight = torch.ones_like(losses).to(pred.device)

        loss = torch.mul(losses, batch_weight).sum()

        return loss

        

class DWASegmenterV0(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.epoch = 0
        self.train_loss_buffer = np.zeros([self.args.seg_n_classes, self.args.epochs])
        self.sample_count = 0

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        logits = self.act(pred)

        B, _, H, W = tuple(logits.size())

        entropy = logits * target

        class_entropy = torch.sum(entropy, axis = [0, 2, 3])

        if self.sample_count >= self.args.num_train_sample:
            self.train_loss_buffer[:, self.epoch] = class_entropy.clone().tolist()
            self.epoch += 1
            self.sample_count = 0
        else:
            self.sample_count += B

        if self.epoch > 1:
            w_i = torch.Tensor(self.train_loss_buffer[:, self.epoch-1] / self.train_loss_buffer[:, self.epoch-2]).to(pred.device)
            batch_weight = self.args.seg_n_classes * F.softmax(w_i/self.args.gumbel_tau, dim=-1)
            weight_entropy = class_entropy * batch_weight
            loss = (-1 / (B * H * W)) * torch.sum(weight_entropy)
        else:
            loss = (-1 / (B * H * W)) * torch.sum(class_entropy)

        return loss
    
class DWASegmenterV1(DWASegmenterV0):
    def __init__(self, args) -> None:
        super().__init__(args)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        logits = self.act(pred)
        
        losses = torch.zeros(self.args.seg_n_classes).to(logits.device)

        B, C, _, _ = tuple(logits.size())

        _logits = logits.permute(0, 2, 3, 1).flatten(0, -2)
        _target = target.permute(0, 2, 3, 1).flatten(0, -2)

        for cidx in range(C):
            c_logits = _logits[_target[:, cidx] == 1]
            c_target = _target[_target[:, cidx] == 1]

            c_entropy = torch.sum(c_logits * c_target, dim=[1, 0])

            B_c, _ = tuple(c_logits.size())

            losses[cidx] = (-1 / B_c) * c_entropy

        if self.sample_count >= self.args.num_train_sample:
            self.train_loss_buffer[:, self.epoch] = losses.clone().tolist()
            self.epoch += 1
            self.sample_count = 0
        else:
            self.sample_count += B
        
        if self.epoch > 1:
            w_i = torch.Tensor(self.train_loss_buffer[:, self.epoch-1] / self.train_loss_buffer[:, self.epoch-2]).to(pred.device)
            batch_weight = self.args.seg_n_classes * F.softmax(w_i/self.args.gumbel_tau, dim=-1)
        else:
            batch_weight = torch.ones_like(losses).to(pred.device)

        loss = torch.mul(losses, batch_weight).sum()

        return loss

class DWASegmenterV2(DWASegmenterV0):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:        
        losses = torch.zeros(self.args.seg_n_classes).to(pred.device)

        B, C, _, _ = tuple(pred.size())

        _preds = pred.permute(0, 2, 3, 1).flatten(0, -2)
        _target = target.permute(0, 2, 3, 1).flatten(0, -2)

        for cidx in range(C):
            c_pred = _preds[_target[:, cidx] == 1]
            c_target = _target[_target[:, cidx] == 1]

            c_loss = self.loss_fn(c_pred, c_target)

            losses[cidx] = c_loss

        if self.sample_count >= self.args.num_train_sample:
            self.train_loss_buffer[:, self.epoch] = losses.detach().clone().tolist()
            self.epoch += 1
            self.sample_count = 0
            
        else:
            self.sample_count += B
        
        if self.epoch > 1:
            w_i = torch.Tensor(self.train_loss_buffer[:, self.epoch-1] / self.train_loss_buffer[:, self.epoch-2]).to(pred.device)
            batch_weight = F.softmax(w_i/self.args.gumbel_tau, dim=-1)
        else:
            batch_weight = torch.ones_like(losses).to(pred.device)

        loss = torch.mul(losses, batch_weight).sum()

        return loss