# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import random
import numpy as np
import torchvision
import wandb
import lightning as L
import argparse
import cv2

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from mcucoder.model import MCUCoder
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from models import CustomImageDataset
wandb.require("core")

# class CustomImageDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.image_files[idx])
#         image = Image.open(img_name)
#         if self.transform:
#             image = self.transform(image)
#         return image


def set_seeds():
    torch.manual_seed(0)
    random.seed(10)
    np.random.seed(0)

def load_datasets(imagenet_root):
    crop = 224
    imagenet_transform = transforms.Compose([
        transforms.Resize(224), 
        transforms.CenterCrop(224),
        # transforms.RandomCrop(crop),
        transforms.ToTensor(),
    ])
    ImageNet_train = CustomImageDataset(root_dir='/data22/aho/high_res_imagenet/', transform=imagenet_transform)

    resize = 224
    imagenet_transform = transforms.Compose([
        transforms.Resize(224), 
        transforms.CenterCrop(224),
        # transforms.Resize((resize, resize), antialias=True),
        transforms.ToTensor(),
    ])
    ImageNet_val = CustomImageDataset(root_dir='/data22/aho/KODAK/', transform=imagenet_transform)

    return ImageNet_train, ImageNet_val

def train(model, train_loader, val_loader, wandb_logger, loss):
    # lr_monitor = LearningRateMonitor(logging_interval='step')
    # torch.set_float32_matmul_precision('high')
    trainer = L.Trainer(
        accelerator="gpu",
        # max_epochs=40,
        max_steps=1_000_000,
        logger=wandb_logger,
        enable_checkpointing=False,
        # precision="bf16-mixed",
        # accumulate_grad_batches=8,
        # check_val_every_n_epoch=5
    )
    trainer.fit(model, train_loader, val_loader)
    
    if loss =='mse':
        torch.save(model.state_dict(), "MCUCoder1M300k196"+"MSE"+".pth")
    if loss =='msssim':
        torch.save(model.state_dict(), "MCUCoder1M300k196"+"MSSSIM"+".pth")

def main(args):
    set_seeds()
    ImageNet_train, ImageNet_val = load_datasets(args.imagenet_root)

    model = MCUCoder(args.loss).to(device='cuda')

    wandb_logger = WandbLogger(name=args.wandb_name, project=args.wandb_project)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Training
    train_loader = DataLoader(ImageNet_train, batch_size=args.batch_size, num_workers=1)
    val_loader = DataLoader(ImageNet_val, batch_size=2, num_workers=1)
    train(model, train_loader, val_loader, wandb_logger, args.loss)

    print("Training complete and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MCUCoder model")
    parser.add_argument("--batch_size", type=int,default=16, help="batch_size")
    parser.add_argument("--imagenet_root", type=str, default="/data22/datasets/imagenet/val/", help="ImageNet dataset root directory")
    parser.add_argument("--wandb_name", type=str, default="MCUCoderRL", help="WandB run name")
    parser.add_argument("--wandb_project", type=str, default="MCUCoderRL", help="WandB project name")
    parser.add_argument("--loss", type=str, default="msssim")

    args = parser.parse_args()
    main(args)
