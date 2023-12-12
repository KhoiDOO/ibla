import os, sys
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_ds
from metrics import metric_dict, metric_batch
from loss import lossmap
from model import getmodel

from utils import folder_setup, save_cfg, Logging, save_json, invnorm, invnorm255


def train_func(args):

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index = args.idx)

    # folder setup and save setting
    exp_dir = folder_setup(args)
    args.exp_dir = exp_dir
    save_cfg(args, exp_dir)

    # dataset setup
    data, args = get_ds(args)
    train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl = data

    if args.verbose:
        print(f"Number Training Samples: {len(train_ds)}")
        print(f"Number Validating Samples: {len(valid_ds)}")
        print(f"Number Testing Samples: {len(test_ds)}")

        print(f"Number Training Batchs: {len(train_dl)}")
        print(f"Number Validating Batchs: {len(valid_dl)}")
        print(f"Number Testing Batchs: {len(test_dl)}")

    # logging setup
    log_interface = Logging(args)

    