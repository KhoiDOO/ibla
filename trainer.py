import os, sys
from rich.progress import track
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_ds
from utils import folder_setup, save_cfg, Logging, save_json, invnorm, invnorm255

from mapping import mapping


def train_func(args):

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index = args.idx)

    # folder setup and save setting
    args.exp_dir = folder_setup(args)
    save_cfg(args, args.exp_dir)

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

    # task mapping
    if args.task not in mapping[args.ds]:
        raise ValueError(f"Currently, task {args.task} is not supported")
    task_dict = mapping[args.ds][args.task]

    # metrics
    metric_dict = task_dict["metrics"]

    # loss
    train_loss_fn = task_dict["loss"][args.loss](args=args)
    eval_loss_fn = task_dict["loss"]['vanilla'](args=args)

    # model
    model = task_dict["model"][args.model](args=args).to(device)

    # optimizer, scheduler
    optimizer = Adam(model.parameters(), lr = 0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max= len(train_dl)*args.epochs)

    if args.wandb:
        log_interface.watch(model)

    # training
    if args.verbose:
        print("Training")
    old_valid_loss = 1e26
    epoch_prog = range(args.epochs) if args.verbose else track(range(args.epochs))

    for epoch in epoch_prog:
        args.epoch = epoch
        if args.verbose:
            print(f"Epoch: {epoch}")
        
        # train data loader
        model.train()
        train_prog = track(enumerate(train_dl)) if args.verbose else enumerate(train_dl)
        for batch, (img, target) in train_prog:
            img = img.to(device)
            target = target.to(device)

            pred = model(img)
            loss = train_loss_fn(pred, target)

            log_interface(key="train/loss", value=loss.item())

            for metric_key in metric_dict:
                metric_value = metric_dict[metric_key](pred, target)
                log_interface(key=f"train/{metric_key}", value=metric_value)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # valid data loader 
        valid_prog = track(enumerate(valid_dl)) if args.verbose else enumerate(valid_dl)
        model.eval()
        with torch.no_grad():
            for batch, (img, target) in valid_prog:
                img = img.to(device)
                target = target.to(device)

                pred = model(img)
                loss = eval_loss_fn(pred, target)

                log_interface(key="valid/loss", value=loss.item())

                for metric_key in metric_dict:
                    metric_value = metric_dict[metric_key](pred, target)
                    log_interface(key=f"valid/{metric_key}", value=metric_value)
        
        # Logging can averaging
        log_interface.step(epoch=epoch)

        # save best and last model
        mean_valid_loss = log_interface.log_avg["valid/loss"]
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': mean_valid_loss
        }
        if  mean_valid_loss <= old_valid_loss:
            old_valid_loss = mean_valid_loss

            save_path = args.exp_dir + f"/best.pt"
            torch.save(save_dict, save_path)
        
        save_path = args.exp_dir + f"/last.pt"
        torch.save(save_dict, save_path)
    
    
    log_interface.reset()
    if args.verbose:
        print("Validating")
    test_prog = track(enumerate(test_dl)) if args.verbose else enumerate(test_dl)
    model.eval()
    with torch.no_grad():
        for batch, (img, target) in test_prog:
            img = img.to(device)
            target = target.to(device)

            pred = model(img)
            loss = eval_loss_fn(pred, target)

            log_interface(key="test/loss", value=loss.item())

            for metric_key in metric_dict:
                metric_value = metric_dict[metric_key](pred, target)
                log_interface(key=f"test/{metric_key}", value=metric_value)

    # finalization
    if args.task == "seg":
        perform_dir = args.exp_dir + f"/{args.task}"
        if not os.path.exists(perform_dir):
            os.mkdir(perform_dir)
        
        img_dir = args.exp_dir + f"/img"
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        
        for idx in range(10):
            img, target = test_ds[idx]
            img = img.unsqueeze(0).to(device)
            
            pred = model(img)
            
            pred_task_np = torch.argmax(pred[0], dim=0).cpu().unsqueeze(0).permute(1, -1, 0).numpy()
            lble_task_np = torch.argmax(target, dim=0).cpu().unsqueeze(0).permute(1, -1, 0).numpy()

            pred_path = perform_dir + f"/pred_{idx}.pdf"
            lble_path = perform_dir + f"/lble_{idx}.pdf"                

            for _img, _path in zip([pred_task_np, lble_task_np], [pred_path, lble_path]):
                plt.figure()

                plt.imshow(_img)
                plt.axis('off')
                plt.savefig(_path, format='pdf', dpi=300)

                plt.close()
            
            if args.ds == 'oxford':
                img_np = invnorm(img[0]).cpu().permute(1, -1, 0).numpy()
                
            path = img_dir + f"/{idx}.pdf"
            plt.figure()

            plt.imshow(img_np)
            plt.axis('off')
            plt.savefig(path, format='pdf', dpi=300, pad_inches=0)

            plt.close()
    
    if args.verbose:
        print("Ending")