# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import lightning as L
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
        self.image_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith(self.valid_exts):
                    self.image_files.append(os.path.join(dirpath, f))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
