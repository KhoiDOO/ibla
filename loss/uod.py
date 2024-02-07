import os, sys
import torch
from torch import nn
from torch._tensor import Tensor 
import torch.nn.functional as F
import numpy as np

from .vanilla_clf_stable import VanillaClassifierStableV0

class UODSegmenterV0(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.epoch = 0
        self.train_loss_buffer = np.zeros([self.args.seg_n_classes, self.args.epochs])
        self.sample_count = 0
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:        
        losses = torch.zeros(self.args.seg_n_classes).to(pred.device)

        B, C, H, W = tuple(pred.size())

        for cidx in range(C):
            c_pred = pred[:, cidx]
            c_target = target[:, cidx]

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

        loss_reg = torch.mul(losses, batch_weight).sum()

        logits = self.act(pred)

        entropy = logits * target

        loss = (-1 / (B * H * W)) * torch.sum(entropy)

        return loss + loss_reg