import os, sys
import torch
from torch import nn
from torchvision import models


def CLF(args):
    if args.model == 'resnet18':
        return models.resnet18(num_classes = args.n_classes)
    else:
        raise ValueError(f"the backbone {args.model} is not supported in classification experiments")
