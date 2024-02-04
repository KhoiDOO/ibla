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

        class_entropy = torch.sum(entropy, axis = [0, 2, 3]).clamp(0.00001)

        if self.sample_count >= self.args.num_train_sample:
            print(self.sample_count, self.args.num_train_sample)
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