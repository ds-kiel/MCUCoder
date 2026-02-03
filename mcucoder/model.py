# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import lightning as L
from copy import copy
import torchvision
import random
from dahuffman import HuffmanCodec
import numpy as np
import cv2
import torch
import json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import pickle
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import wandb
import torchvision.models as models
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from compressai.layers import GDN
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    conv1x1,
    subpel_conv3x3,
)
from compressai.models.utils import conv, deconv
from torch import Tensor

device = 'cuda'

class ResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block.

    Introduced by [He2016], this block sandwiches a 3x3 convolution
    between two 1x1 convolutions which reduce and then restore the
    number of channels. This reduces the number of parameters required.

    [He2016]: `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/abs/1512.03385>`_, by Kaiming He, Xiangyu Zhang,
    Shaoqing Ren, and Jian Sun, CVPR 2016.

    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = min(in_ch, out_ch) // 2
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_ch, mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(mid_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)

        out = x
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out + identity
        
def psnr_batch(img1, img2):
    mse = F.mse_loss(img1, img2, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)
    psnr_values = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return torch.mean(psnr_values.detach().cpu())

def ms_ssim_batch(img1, img2, data_range=1.0):

    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range).to(device)
    
    # Calculate MS-SSIM for each image pair in the batch
    ms_ssim_values = [ms_ssim(img1[i].unsqueeze(0), img2[i].unsqueeze(0)).item() for i in range(img1.size(0))]
    
    # Convert MS-SSIM values to dB
    ms_ssim_db_values = [ms_ssim_to_db(value) for value in ms_ssim_values]
    
    # Return on CPU to avoid keeping validation buffers on GPU
    return torch.tensor(ms_ssim_db_values, device='cpu')

def ms_ssim_to_db(ms_ssim):
    return -10 * np.log10(1 - ms_ssim)


class VGGPerceptualLoss(nn.Module):
    """VGG16-based perceptual loss using ImageNet weights."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        self.blocks = nn.ModuleList(
            [
                vgg[:4].eval(),   # relu1_2
                vgg[4:9].eval(),  # relu2_2
                vgg[9:16].eval(), # relu3_3
                vgg[16:23].eval() # relu4_3
            ]
        )
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        x, y = input, target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss = loss + F.l1_loss(x, y)
        return loss
    
# NOTE: original encoder kept for reference
class Encoder(L.LightningModule):
    def __init__(self):
        super(Encoder, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(3, 16, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=4, padding=1)
        self.conv3 = nn.Conv2d(16, 12, 3, stride=1, padding=1)

        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# class Encoder(L.LightningModule):
#     def __init__(self):
#         super(Encoder, self).__init__()
        
#         # Encoder
#         self.conv1 = nn.Conv2d(3, 16, 7, stride=2, padding=3)
#         self.conv2 = nn.Conv2d(16, 16, 5, stride=4, padding=1)
#         self.conv3 = nn.Conv2d(16, 12, 3, stride=1, padding=1)

#     def forward(self, x, p_selected=None):
#         x = F.relu(self.conv1(x))
#         if p_selected is not None:
#             k = int((1 - p_selected) * x.shape[1])
#             if k > 0:
#                 mask = torch.ones_like(x)
#                 mask[:, -k:, :, :] = 0
#                 x = x * mask

#         x = F.relu(self.conv2(x))

#         if p_selected is not None:
#             k = int((1 - p_selected) * x.shape[1])
#             if k > 0:
#                 mask = torch.ones_like(x)
#                 mask[:, -k:, :, :] = 0
#                 x = x * mask

#         x = self.conv3(x)
#         return x



class Decoder(L.LightningModule):
    def __init__(self,N):
        super(Decoder, self).__init__()
                
        self.N = N
        self.dec = nn.Sequential(
            AttentionBlock(12),
            deconv(12, self.N, kernel_size=5, stride=2),
            
            ResidualBottleneckBlock(self.N, self.N),
            ResidualBottleneckBlock(self.N, self.N),
            ResidualBottleneckBlock(self.N, self.N),
            AttentionBlock(self.N),
            ResidualBottleneckBlock(self.N, self.N),
            ResidualBottleneckBlock(self.N, self.N),
            ResidualBottleneckBlock(self.N, self.N),
            AttentionBlock(self.N),
            ResidualBottleneckBlock(self.N, self.N),
            ResidualBottleneckBlock(self.N, self.N),
            ResidualBottleneckBlock(self.N, self.N),
            deconv(self.N, self.N, kernel_size=5, stride=2),
            AttentionBlock(self.N),
            ResidualBottleneckBlock(self.N, self.N),
            ResidualBottleneckBlock(self.N, self.N),
            ResidualBottleneckBlock(self.N, self.N),
            AttentionBlock(self.N),
            ResidualBottleneckBlock(self.N, self.N),
            ResidualBottleneckBlock(self.N, self.N),
            ResidualBottleneckBlock(self.N, self.N),
            AttentionBlock(self.N),
            ResidualBottleneckBlock(self.N, self.N),
            ResidualBottleneckBlock(self.N, self.N),
            ResidualBottleneckBlock(self.N, self.N),
            deconv(self.N, 3, kernel_size=5, stride=2),
        )
        
    def forward(self, x): 
        x = self.dec(x)
        x = torch.sigmoid(x)
        return x


#Define the Convolutional Autoencoder

class MCUCoder(L.LightningModule):
    def __init__(self, loss=None, N=196, total_steps=None, lr=1e-4):
        super(MCUCoder, self).__init__()
        #
        self.encoder = Encoder()
        self.decoder = Decoder(N=N)
        self.loss_func = loss
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        self.vgg_loss = None  # lazily created when needed to avoid GPU memory use
        self.total_steps = total_steps
        self.vgg_switch_step = None
        self.lr = lr
        self.lambda_rate = 1e-3
        
        self.p = None
        self.replace_value = 0
        self.last_latent = None

        self.training_step_loss = []
        self.validation_step_psnr = []
        self.validation_step_ms_ssim = []
        self.saved_images=[]

    def random_noise(self, x, r1, r2):
        temp_x = x.clone()
        noise = (r1 - r2) * torch.rand(x.shape) + r2
        return torch.clamp(temp_x + noise.cuda(), min=0.0, max=1.0)
    

    def forward(self, x):
        # Encoder
        if self.p:
            selected_p = self.p
        else:
            selected_p = np.random.randint(1, 13)/12
            # selected_p = np.random.choice([1, 4, 8, 12]) /12
        x = self.encoder(x)
        # x = self.encoder(x, selected_p)
        x = self.rate_less(x, selected_p)
        self.last_latent = x

        # noise
        if not self.training:
            noise = torch.rand_like(x, dtype=torch.float) * 0.02 - 0.01
            x = x + noise.clone()
        
        x = self.decoder(x)
        self.rec_image = x.detach().clone()

        return x

    def set_rate_weight(self, lambda_rate: float):
        """Optional setter to enable/adjust rate-distortion weighting without changing init signature."""
        self.lambda_rate = float(lambda_rate)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        first_phase = int(self.trainer.max_steps * 0.95)
        # first_phase = 200_000

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=first_phase, gamma=0.1)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def _get_vgg_loss(self):
        if self.vgg_loss is None:
            self.vgg_loss = VGGPerceptualLoss().to(self.device)
        return self.vgg_loss


    def _select_active_loss(self):
        """Return the loss name to use this step (handles staged vgg training)."""
        if self.loss_func != 'vgg':
            return self.loss_func

        if self.vgg_switch_step is None:
            max_steps = self.total_steps
            if self.trainer is not None and getattr(self.trainer, "max_steps", None):
                max_steps = self.trainer.max_steps
            if max_steps:
                self.vgg_switch_step = int(max_steps * 0.8)

        if self.vgg_switch_step is None:
            return 'msssim'

        return 'vgg' if self.global_step >= self.vgg_switch_step else 'msssim'
        
    def training_step(self, train_batch, batch_idx):
        images = train_batch
        
        outputs = self(images)

        active_loss = self._select_active_loss()
        rate_bpp = None
        if active_loss =='msssim':
            distortion_loss = 1 - self.ms_ssim(outputs, images)
            self.log('during_train_loss_ms_ssim', distortion_loss, on_epoch=True, prog_bar=True, logger=True)
        
        elif active_loss =='mse':
            distortion_loss = nn.MSELoss()(outputs, images)
            self.log('during_train_loss_mse', distortion_loss, on_epoch=True, prog_bar=True, logger=True)
        elif active_loss == 'vgg':
            vgg_loss = self._get_vgg_loss()
            distortion_loss = vgg_loss(outputs, images)
            self.log('during_train_loss_vgg', distortion_loss, on_epoch=True, prog_bar=True, logger=True)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_func}")

        if self.lambda_rate > 0.0 and self.last_latent is not None:
            rate_bpp = self.compute_proxy_bpp(self.last_latent)
            self.log('train_rate_bpp', rate_bpp, on_epoch=True, prog_bar=True, logger=True)
            loss = distortion_loss + self.lambda_rate * rate_bpp
        else:
            loss = distortion_loss

        # Store detached loss to avoid holding computation graphs across steps
        self.training_step_loss.append(loss.detach().cpu())

        return {
            'loss': loss
        }

    def on_train_epoch_end(self):
        
        loss = torch.stack([x for x in self.training_step_loss]).mean()
        self.log('train_loss_epoch', loss, on_epoch=True, prog_bar=True, logger=True)
        
        self.training_step_loss.clear()

        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log('learning_rate', lr, on_step=False, on_epoch=True)

    def validation_step(self, val_batch, batch_idx):
        
        self.saved_images = []
        msssim_temp = []
        psnr_temp = []

        for t in [2, 6, 12]:
            self.p = t/12
            images = val_batch
            outputs = self(images)
        
            psnr = psnr_batch(outputs, images)
            psnr_temp.append(psnr)
            
            ms_ssim = ms_ssim_batch(outputs, images)
            msssim_temp.append(ms_ssim)
            
            # Keep validation visualizations on CPU to reduce GPU memory
            self.saved_images.append(self.rec_image[0].detach().cpu())

        
        self.validation_step_psnr.append(psnr_temp)
        self.validation_step_ms_ssim.append(msssim_temp)

        self.p = None
        return {'loss': psnr, 'ms_ssim': ms_ssim}
            
    def on_validation_epoch_end(self):            
        psnr = torch.stack([x[0] for x in self.validation_step_psnr]).mean()
        self.log('val_psnr_2l_epoch', psnr, on_epoch=True, prog_bar=True, logger=True)

        psnr = torch.stack([x[1] for x in self.validation_step_psnr]).mean()
        self.log('val_psnr_6l_epoch', psnr, on_epoch=True, prog_bar=True, logger=True)

        psnr = torch.stack([x[2] for x in self.validation_step_psnr]).mean()
        self.log('val_psnr_12l_epoch', psnr, on_epoch=True, prog_bar=True, logger=True)

        
        ms_ssim = torch.stack([x[0] for x in self.validation_step_ms_ssim]).mean()
        self.log('val_ms_ssim_2l_epoch', ms_ssim, on_epoch=True, prog_bar=True, logger=True)

        ms_ssim = torch.stack([x[1] for x in self.validation_step_ms_ssim]).mean()
        self.log('val_ms_ssim_6l_epoch', ms_ssim, on_epoch=True, prog_bar=True, logger=True)

        ms_ssim = torch.stack([x[2] for x in self.validation_step_ms_ssim]).mean()
        self.log('val_ms_ssim_12l_epoch', ms_ssim, on_epoch=True, prog_bar=True, logger=True)

        
        self.logger.experiment.log({"rec_image_2l": wandb.Image(self.saved_images[0])})
        self.logger.experiment.log({"rec_image_6l": wandb.Image(self.saved_images[1])})
        self.logger.experiment.log({"rec_image_12l": wandb.Image(self.saved_images[2])})

        self.p = None
        self.validation_step_psnr.clear()
        self.validation_step_ms_ssim.clear()

    def rate_less(self,x,selected_p=None):
        if self.p:
            # p shows the percentage of keeping
            p = self.p
        else:
            p = selected_p
            # p = np.random.randint(1, 13)/12
            # p = np.random.choice([1, 4, 8, 12]) / 12 

        if p != 1.0:
            p_channels = int(p * x.shape[1])
            replace_channels = x.shape[1] - p_channels
            if replace_channels > 0:
                keep = x[:, :p_channels, :, :]
                replace_tensor = torch.full(
                    (x.shape[0], replace_channels, x.shape[2], x.shape[3]),
                    self.replace_value,
                    device=x.device,
                    dtype=x.dtype,
                )
                x = torch.cat((keep, replace_tensor), dim=1)
        return x

    def compute_proxy_bpp(self, latent: Tensor) -> Tensor:
        """Approximate bpp using a per-channel Gaussian model on quantized latent."""
        if latent is None:
            return torch.tensor(0.0, device=self.device)

        noisy_latent = latent + torch.empty_like(latent).uniform_(-0.5, 0.5)
        flat = noisy_latent.view(noisy_latent.size(0), noisy_latent.size(1), -1)
        sigma = flat.std(dim=2, unbiased=False).clamp(min=1e-6).unsqueeze(-1)

        nll = 0.5 * (math.log(2 * math.pi) + 2 * torch.log(sigma)) + (flat ** 2) / (2 * (sigma ** 2))
        total_nll = nll.sum()

        num_pixels = noisy_latent.size(0) * noisy_latent.size(2) * noisy_latent.size(3)
        bpp = total_nll / (math.log(2.0) * num_pixels)
        return bpp
