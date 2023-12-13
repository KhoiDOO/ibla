import os, sys
import argparse
from typing import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='MTLA')

    # DATASET
    parser.add_argument('--ds', type=str, required=True, choices = [
        'oxford', 'nyu', 'celeb', 'city'],
        help='dataset used in training')
    parser.add_argument('--bs', type=int, required=True, default=64,
        help='batch size used for data set')
    parser.add_argument('--pinmem', action='store_true',
        help='toggle to pin memory in data loader')
    parser.add_argument('--wk', type=int, default=12,  
        help='number of worker processor contributing to data preprocessing')
    parser.add_argument('--citi_mode', type=str, default='fine',  choices=['fine', 'coarse'],
        help='mode used for cityscape dataset')
    
    # TRAINING GENERAL SETTINGS
    parser.add_argument('--idx', type=int, default=0,
        help='device index used in training')
    parser.add_argument('--loss', type=str, default='vanilla', 
        choices=['vanilla'],
        help='mtl method used in training')
    parser.add_argument('--epochs', type=int, default=100,
        help='number of epochs used in training')
    parser.add_argument('--test', action='store_true',
        help='toggle to say that this experiment is just flow testing')

    # LOGGING
    parser.add_argument('--wandb', action='store_true',
        help='toggle to use wandb for online saving')
    parser.add_argument('--log', action='store_true',
        help='toggle to use tensorboard for offline saving')
    parser.add_argument('--wandb_prj', type=str, default="IBLA",
        help='toggle to use wandb for online saving')
    parser.add_argument('--wandb_entity', type=str, default="scalemind",
        help='toggle to use wandb for online saving')
    parser.add_argument('--verbose', action='store_true',
        help='toggle to use print information during training')

    # MODEL
    parser.add_argument('--init_ch', type=int, default=32,
        help='number of kernel in the first')
    parser.add_argument('--lr', type=float, default=0.001,
        help='learning rate')


    args = parser.parse_args()

    from trainer import train_func

    train_func(args)