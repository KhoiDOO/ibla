import os, sys
import torch
from torch import nn
from torch._tensor import Tensor 
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np
from functools import partial

from .vanilla_clf_stable import VanillaClassifierStableV0


class EQLClassifierV0(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.args = args
        self.gamma = args.eq_gamma
        self.mu = args.eq_mu
        self.alpha = args.eq_alpha
        
        self.register_buffer('pos_grad', torch.zeros(args.n_classes))
        self.register_buffer('neg_grad', torch.zeros(args.n_classes))
        self.register_buffer('pos_neg', torch.ones(args.n_classes) * 100)
        
        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)
    
    def forward(self, pred, target) -> Tensor:
        self.n_i, self.n_c = pred.size()

        self.gt_classes = target
        self.pred_class_logits = pred

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        _target = expand_label(pred, target)

        pos_w, neg_w = self.get_weight(pred)

        weight = pos_w * _target + neg_w * (1 - _target)

        cls_loss = F.binary_cross_entropy_with_logits(pred, _target, reduction='none')
        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(pred.detach(), _target.detach(), weight.detach())

        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, cls_score):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        if self.test_with_obj:
            cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]

        # dist.all_reduce(pos_grad)
        # dist.all_reduce(neg_grad)

        self.pos_grad += pos_grad.mean()
        self.neg_grad += neg_grad.mean()
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

    def get_weight(self, cls_score):
        neg_w = torch.cat([self.map_func(self.pos_neg.to(cls_score.device)), cls_score.new_ones(1)])
        pos_w = 1 + self.alpha * (1 - neg_w)
        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w