import os, sys
import torch
from torch import nn 
import torch.nn.functional as F


class VanillaClassifierV0(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.act = nn.Softmax(dim=1)
        self.log_type = {
            'log2' : torch.log2,
            'log10' : torch.log10,
            'log' : torch.log
        }

    def forward(self, pred, target) -> torch.Tensor:

        pred_soft = self.act(pred)

        logits = self.log_type[self.args.log_type](pred_soft)

        B, C = tuple(logits.size())

        entropy = logits * F.one_hot(target, num_classes=C).float()

        return (-1 / B) * torch.sum(entropy)


class VanillaClassifierV1(VanillaClassifierV0):
    def __init__(self, args) -> None:
        super().__init__(args=args)
    
    def forward(self, pred, target) -> torch.Tensor:

        cls_loss = {}

        pred_soft = self.act(pred)

        logits = self.log_type[self.args.log_type](pred_soft)

        B = list(target.size())[0]
        
        for b_logits, b_target in zip(logits, target):
            if b_target.item() in cls_loss:
                cls_loss[b_target.item()].append(b_logits[b_target.item()])
            else:
                cls_loss[b_target.item()] = [b_logits[b_target.item()]]
        
        sum_cls_loss = {
            _cls : sum(cls_loss[_cls]) for _cls in cls_loss
        }

        return (-1 / B) * sum(list(sum_cls_loss.values()))

class VanillaClassifierV2(VanillaClassifierV0):
    def __init__(self, args) -> None:
        super().__init__(args=args)
    
    def forward(self, pred, target) -> torch.Tensor:

        cls_loss = {}

        pred_soft = self.act(pred)

        logits = self.log_type[self.args.log_type](pred_soft)

        B, C = tuple(logits.size())
        
        for b_logits, b_target in zip(logits, target):
            
            entropy = torch.matmul(b_logits, F.one_hot(b_target, num_classes=C).float())
            
            if b_target.item() in cls_loss:
                cls_loss[b_target.item()].append(entropy)
            else:
                cls_loss[b_target.item()] = [entropy]
        
        sum_cls_loss = {
            _cls : sum(cls_loss[_cls]) for _cls in cls_loss
        }

        return (-1 / B) * sum(list(sum_cls_loss.values()))