import os, sys
import torch
from torch import nn
from torch._tensor import Tensor 
import torch.nn.functional as F
import numpy as np

from .vanilla_clf_stable import VanillaClassifierStableV0


class DynamicArray:
    def __init__(self, size = 2) -> None:
        self.__storage = []
        self.__size = size
    
    def append(self, x):
        if len(self.__storage) < self.__size:
            self.__storage.append(x)
        else:
            self.__storage = self.__storage[1:]
            self.__storage.append(x)
    
    @property
    def size(self):
        return self.__size
    
    @property
    def storage(self):
        return self.__storage

class HDLRWClassifierV0(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.epoch = 0
        self.train_loss_buffer = DynamicArray(size = args.mem_size)

    def forward(self, pred, target) -> Tensor:

        cls_loss = {}

        logits = self.act(pred)

        B, C = tuple(logits.size())
        
        for b_logits, b_target in zip(logits, target):
            if b_target.item() in cls_loss:
                cls_loss[b_target.item()].append(b_logits[b_target.item()])
            else:
                cls_loss[b_target.item()] = [b_logits[b_target.item()]]
        
        sum_cls_loss = {
            _cls : sum(cls_loss[_cls]) for _cls in cls_loss
        }

        backup_loss_dict = {
            _cls : sum_cls_loss[_cls].item() for _cls in sum_cls_loss
        }

        self.train_loss_buffer.append(backup_loss_dict)

        if self.epoch > 1:
            self.epoch += 1
            weighted_dict = {}

            for _cls in backup_loss_dict:
                if _cls in self.train_loss_buffer.storage[-1]:
                    nom = self.train_loss_buffer.storage[-1][_cls]
                else:
                    nom = 1

                if _cls in self.train_loss_buffer.storage[-2]:
                    dom = self.train_loss_buffer.storage[-2][_cls]
                else:
                    dom = 1
                
                weighted_dict[_cls] = torch.Tensor([nom / dom]).to(pred.device)

            exp_weight_dict = {
                _cls : torch.exp(weighted_dict[_cls]) for _cls in weighted_dict
            }

            exp_sum = sum(list(exp_weight_dict.values()))
            softmax_weight_dict = {
                _cls : C * exp_weight_dict[_cls] / exp_sum for _cls in exp_weight_dict
            }

            weighted_loss_dict = {
                _cls : softmax_weight_dict[_cls] * sum_cls_loss[_cls] for _cls in softmax_weight_dict
            }

            return (-1 / B) * sum(list(weighted_loss_dict.values()))

        else:
            self.epoch += 1
            return (-1 / B) * sum(list(sum_cls_loss.values()))
    

class HDLRWSegmenterV0(VanillaClassifierStableV0):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.epoch = 0
        self.train_loss_buffer = np.zeros([self.args.seg_n_classes, self.args.epochs])

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        logits = self.act(pred)

        B, C, H, W = tuple(logits.size())

        entropy = logits * target

        class_entropy = torch.sum(entropy, axis = [0, 2, 3]).clamp(0.00001)

        self.train_loss_buffer[:, self.epoch] = class_entropy.clone().tolist()

        if self.epoch > 1:
            w_i = torch.Tensor(self.train_loss_buffer[:, self.epoch-1] / self.train_loss_buffer[:, self.epoch-2]).to(pred.device)
            batch_weight = self.args.seg_n_classes * F.softmax(w_i/self.args.T, dim=-1)
            weght_entropy = class_entropy * batch_weight
            loss = (-1 / (B * H * W)) * torch.sum(weght_entropy)
        else:
            loss = (-1 / (B * H * W)) * torch.sum(class_entropy)

        self.epoch += 1
        return loss