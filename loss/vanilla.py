import torch
from torch import nn
from .loss_fn import loss_dict

class Vanilla(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
    
    def forward(self, pd, gt):
        return loss_dict[self.args.task](pd, gt)