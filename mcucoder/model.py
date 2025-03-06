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
    
    return torch.tensor(ms_ssim_db_values, device='cuda')

def ms_ssim_to_db(ms_ssim):
    return -10 * np.log10(1 - ms_ssim)
    
#Define the Convolutional Autoencoder

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
    def __init__(self, loss=None, N=196):
        super(MCUCoder, self).__init__()
        #
        self.encoder = Encoder()
        self.decoder = Decoder(N=N)
        self.loss_func = loss
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        
        self.p = None
        self.replace_value = 0

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
        x = self.encoder(x)
        x = self.rate_less(x)

        # noise
        if not self.training:
            noise = torch.rand_like(x, dtype=torch.float) * 0.02 - 0.01
            x = x + noise.clone()
        
        x = self.decoder(x)
        self.rec_image = x.detach().clone()

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        first_phase = int(self.trainer.max_steps * 0.95)
        # first_phase = 200_000

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=first_phase, gamma=0.1)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

        
    def training_step(self, train_batch, batch_idx):
        images = train_batch
        
        outputs = self(images)

        if self.loss_func =='msssim':
            loss = 1 - self.ms_ssim(outputs, images)
            self.log('during_train_loss_ms_ssim', loss, on_epoch=True, prog_bar=True, logger=True)
        
        if self.loss_func =='mse':
            loss = nn.MSELoss()(outputs, images)
            self.log('during_train_loss_mse', loss, on_epoch=True, prog_bar=True, logger=True)

        self.training_step_loss.append(loss)

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
            
            self.saved_images.append(self.rec_image[0])

        
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

    def rate_less(self,x):
        temp_x = x.clone()
        if self.p:
            # p shows the percentage of keeping
            p = self.p
        else:
            p = np.random.randint(1, 13)/12
            
        if p != 1.0:            
            p = int(p * x.shape[1])
            replace_tensor = torch.rand(x.shape[0], x.shape[1]-p, x.shape[2], x.shape[3]).fill_(self.replace_value)
            temp_x[:,-(x.shape[1]-p):,:,:] =  replace_tensor
        return temp_x
