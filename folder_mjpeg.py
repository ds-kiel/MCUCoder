import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
import warnings
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19
from tqdm import tqdm
from mcucoder.model import VGGPerceptualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size for processing frames')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing videos')
    parser.add_argument('--output_dir', type=str, default="./results/JPEG/", help='Directory to save output images and videos')
    parser.add_argument("--vgg_weights_path", type=str, default=None, help="Optional path to local VGG19 weights (otherwise ImageNet weights are used)")
    return parser.parse_args()

class VideoDataset(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.frames = self._load_frames()

    def _load_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, frame_index):
        frame = self.frames[frame_index]
        if self.transform:
            frame = self.transform(frame)
        return frame, frame_index

def get_video_frames(video_path, resize):
    transform = transforms.Compose([
        transforms.Resize((resize, resize), antialias=True),
        transforms.ToTensor(),
    ])
    video_dataset = VideoDataset(video_path=video_path, transform=transform)
    return video_dataset

def psnr_batch(img1, img2):
    mse = F.mse_loss(img1, img2, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)
    psnr_values = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr_values

def ms_ssim_to_db(ms_ssim):
    return -10 * np.log10(1 - ms_ssim)

def ms_ssim_batch(img1, img2, data_range=1.0):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range)
    ms_ssim_values = [ms_ssim(img1[i].unsqueeze(0), img2[i].unsqueeze(0)).item() for i in range(img1.size(0))]
    ms_ssim_db_values = [ms_ssim_to_db(value) for value in ms_ssim_values]
    return ms_ssim_db_values


class VGGLoss(nn.Module):
    def __init__(self, device="cuda", weights_path=None):
        super().__init__()
        try:
            if weights_path:
                vgg = vgg19(weights=None)
                state_dict = torch.load(weights_path, map_location=device)
                vgg.load_state_dict(state_dict)
            else:
                vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        except Exception as exc:
            raise RuntimeError(f"Unable to load VGG19 weights: {exc}") from exc

        self.features = nn.Sequential(*list(vgg.features.children())[:16]).to(device)
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1),
        )

    def forward(self, pred, target):
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        pred_features = self.features(pred_norm)
        target_features = self.features(target_norm)
        return self.criterion(pred_features, target_features)


def build_vgg_loss(device, weights_path):
    try:
        vgg_metric = VGGPerceptualLoss().to(device)
        vgg_metric.eval()
        return vgg_metric
    except Exception as exc:
        warnings.warn(f"VGG perceptual loss disabled: {exc}")

    if not weights_path:
        return None

    try:
        legacy_vgg = VGGLoss(device=device, weights_path=weights_path)
        return legacy_vgg
    except RuntimeError as exc:
        warnings.warn(f"Failed to load provided VGG weights: {exc}")
        return None


def compress_and_evaluate_jpeg(images, quality, output_dir, vgg_loss_fn=None):
    size_list = []
    psnr_list = []
    ms_ssim_list = []
    vgg_loss_list = []
    compressed_images = []

    for i, image in enumerate(images):
        image_pil = transforms.ToPILImage()(image.cpu())
        output_path = os.path.join(output_dir, f"compressed_{i}_{quality}.jpg")
        image_pil.save(output_path, 'JPEG', quality=quality)
        
        compressed_image = Image.open(output_path)
        compressed_image = transforms.ToTensor()(compressed_image).unsqueeze(0)
        compressed_images.append(compressed_image)
        
        # Calculate size
        size_list.append(os.path.getsize(output_path) / 1024)
        
        # Calculate PSNR for the image
        psnr_value = psnr_batch(compressed_image, image.unsqueeze(0))
        psnr_list.extend(psnr_value)
        
        # Calculate MS-SSIM for the image
        ms_ssim_value = ms_ssim_batch(compressed_image, image.unsqueeze(0))
        ms_ssim_list.extend(ms_ssim_value)

        if vgg_loss_fn is not None:
            with torch.no_grad():
                vgg_value = vgg_loss_fn(
                    compressed_image.to(device),
                    image.unsqueeze(0).to(device)
                ).detach().cpu().item()
            vgg_loss_list.append(vgg_value)
    
    avg_vgg = np.mean(vgg_loss_list) if len(vgg_loss_list) > 0 else None
    return np.mean(size_list), np.mean(psnr_list), np.mean(ms_ssim_list), avg_vgg, compressed_images

def save_video_for_quality(compressed_images, quality, output_dir, frame_size, video_name):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(output_dir, f'{video_name}_Q_{quality}_output_video.mp4')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 20, frame_size)

    for img in compressed_images:
        img_np = img.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_bgr = (img_np * 255).astype(np.uint8)  # Convert to BGR for OpenCV
        video_writer.write(img_bgr)

    video_writer.release()
    return output_video_path

def eval_videos_in_folder(args):
    video_files = [f for f in os.listdir(args.input_folder) if os.path.isfile(os.path.join(args.input_folder, f))]
    qualities = range(1, 40, 5)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_video_dir = os.path.join(args.output_dir, "videos")
    os.makedirs(output_video_dir, exist_ok=True)

    vgg_loss_fn = build_vgg_loss(device=device, weights_path=args.vgg_weights_path)
    
    overall_results = {quality: {'sizes': [], 'psnrs': [], 'ms_ssims': [], 'vgg': []} for quality in qualities}

    all_for_print = []
    for quality in qualities:
        print(f'Evaluating quality level {quality}')
        
        
        for video_file in video_files:
            video_path = os.path.join(args.input_folder, video_file)
            video_dataset = get_video_frames(video_path, 224)
            video_loader = DataLoader(video_dataset, batch_size=args.batch_size, num_workers=1)
            video_stem, _ = os.path.splitext(video_file)

            sizes = []
            psnrs = []
            ms_ssims = []
            vgg_losses = []
            all_compressed_images = []

            for idx, (images, _) in enumerate(video_loader):
                avg_size, avg_psnr, avg_ms_ssim, avg_vgg, compressed_images = compress_and_evaluate_jpeg(images, quality, args.output_dir, vgg_loss_fn=vgg_loss_fn)
                sizes.append(avg_size)
                psnrs.append(avg_psnr)
                ms_ssims.append(avg_ms_ssim)
                if avg_vgg is not None:
                    vgg_losses.append(avg_vgg)
                all_compressed_images.extend(compressed_images)

            avg_size = np.mean(sizes)
            avg_psnr = np.mean(psnrs)
            avg_ms_ssim = np.mean(ms_ssims)
            avg_vgg = np.mean(vgg_losses) if len(vgg_losses) > 0 else None

            # Convert image size to bits
            avg_size = avg_size * 8 * 1024
            
            # Calculate bpp (bits per pixel)
            bpp = avg_size / (224 * 224)
            
            overall_results[quality]['sizes'].append(bpp)
            overall_results[quality]['psnrs'].append(avg_psnr)
            overall_results[quality]['ms_ssims'].append(avg_ms_ssim)
            if avg_vgg is not None:
                overall_results[quality]['vgg'].append(avg_vgg)

            compressed_video_path = save_video_for_quality(
                all_compressed_images,
                quality,
                output_video_dir,
                (224, 224),
                video_stem,
            )
            
        avg_bpp = np.mean(overall_results[quality]['sizes'])
        avg_psnr = np.mean(overall_results[quality]['psnrs'])
        avg_ms_ssim = np.mean(overall_results[quality]['ms_ssims'])
        avg_vgg = np.mean(overall_results[quality]['vgg']) if len(overall_results[quality]['vgg']) > 0 else None

        all_for_print.append([float(avg_bpp), float(avg_psnr), float(avg_ms_ssim), None if avg_vgg is None else float(avg_vgg)])
        msg_parts = [
            f'Quality: {quality}',
            f'Avg bpp: {avg_bpp:.2f}',
            f'Avg PSNR: {avg_psnr:.2f}dB',
            f'Avg MS_SSIM: {avg_ms_ssim:.2f}dB',
        ]
        if avg_vgg is not None:
            msg_parts.append(f'VGG Loss: {avg_vgg:.4f}')
        print(", ".join(msg_parts))
    print(all_for_print)

if __name__ == "__main__":
    args = parse_args()
    eval_videos_in_folder(args)
