# Fine-tune an existing MCUCoder checkpoint with VGG loss.
# Keeps the main training code untouched and saves outputs into a separate run folder.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import argparse
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import lightning as L
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from mcucoder import MCUCoder, CustomImageDataset

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
wandb.require("core")


def set_seeds():
    torch.manual_seed(0)
    random.seed(10)
    np.random.seed(0)


def build_datasets(train_root, val_root, resize=224):
    train_transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
        ]
    )
    train_ds = CustomImageDataset(root_dir=train_root, transform=train_transform)
    val_ds = CustomImageDataset(root_dir=val_root, transform=val_transform)
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MCUCoder on VGG loss from a pretrained checkpoint.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained checkpoint (.pth or .ckpt).")
    parser.add_argument("--train_root", type=str, default="/data22/aho/high_res_imagenet/", help="Training images root.")
    parser.add_argument("--val_root", type=str, default="/data22/aho/KODAK/", help="Validation images root.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=1, help="Dataloader workers.")
    parser.add_argument("--number_of_iterations", type=int, default=100_000, help="Fine-tuning steps.")
    parser.add_argument("--number_of_channels", type=int, default=196, help="Latent channels.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning.")
    parser.add_argument("--wandb_name", type=str, default="MCUCoder_finetune_vgg", help="WandB run name.")
    parser.add_argument("--wandb_project", type=str, default="MCUCoderRL", help="WandB project.")
    parser.add_argument("--pure_vgg", action="store_true", help="Use VGG loss from step 0 (skip MS-SSIM warmup).")
    args = parser.parse_args()

    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, val_ds = build_datasets(args.train_root, args.val_root, resize=224)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2, num_workers=args.num_workers)

    model = MCUCoder(loss="vgg", N=args.number_of_channels, total_steps=args.number_of_iterations, lr=args.lr).to(device)

    # Load pretrained weights (supports raw state_dict or Lightning checkpoint)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    if args.pure_vgg:
        model.vgg_switch_step = 0  # use VGG loss from the first step

    # Unique run dir per fine-tune
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{args.wandb_name}_{ts}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(name=args.wandb_name, project=args.wandb_project, save_dir=str(run_dir))
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch{epoch:04d}",
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
    )

    trainer = L.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        max_steps=args.number_of_iterations,
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_cb, lr_monitor],
    )

    trainer.fit(model, train_loader, val_loader)

    final_name = f"MCUCoder_finetune_vgg_{ts}.pth"
    final_path = run_dir / final_name
    torch.save(model.state_dict(), final_path)

    # Optional copy beside this script for convenience
    root_copy = Path(__file__).resolve().parent / final_name
    torch.save(model.state_dict(), root_copy)

    print(f"Fine-tune complete. Saved model to {final_path} and {root_copy}")


if __name__ == "__main__":
    main()
