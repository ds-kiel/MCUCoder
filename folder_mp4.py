import os
import cv2
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import moviepy.editor as mp
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torchvision.models import VGG19_Weights, vgg19
from mcucoder.model import VGGPerceptualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def psnr_batch(img1, img2):
    mse = F.mse_loss(img1, img2, reduction='none')
    mse = mse.reshape(mse.size(0), -1).mean(dim=1)
    psnr_values = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr_values

def ms_ssim_to_db(ms_ssim):
    return -10 * np.log10(1 - ms_ssim)

def ms_ssim_batch(img1, img2, data_range=1.0):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range)
    
    # Calculate MS-SSIM for each image pair in the batch
    ms_ssim_values = [ms_ssim(img1[i].unsqueeze(0), img2[i].unsqueeze(0)).item() for i in range(img1.size(0))]
    
    # Convert MS-SSIM values to dB
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


def build_vgg_loss(device, weights_path=None):
    try:
        vgg_metric = VGGPerceptualLoss().to(device)
        vgg_metric.eval()
        return vgg_metric
    except Exception as exc:
        warnings.warn(f"VGG perceptual loss disabled: {exc}")

    if not weights_path:
        return None
    try:
        return VGGLoss(device=device, weights_path=weights_path)
    except RuntimeError as exc:
        warnings.warn(f"Failed to load provided VGG weights: {exc}")
        return None

def get_video_fps(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None
    
    # Get the frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Release the video capture object
    cap.release()
    
    return fps

def extract_and_resize_frames(video_path, size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    previous_frame = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)
        
        frames.append(frame)
        previous_frame = frame
        frame_idx += 1
    
    cap.release()
    return frames


def compress_video_from_frames(frames, output_path, video_path, compression_ratio=23):
    # Convert frames to a video clip

    fps = get_video_fps(video_path)
    clip = mp.ImageSequenceClip(frames, fps=fps)

    
    # # H264
    # clip.write_videofile(output_path, codec='libx264', preset='veryfast', ffmpeg_params=['-crf', str(compression_ratio)], verbose=False, logger=None)

    
    # H265
    clip.write_videofile(output_path, codec='libx265', preset='veryfast', ffmpeg_params=[
        '-crf', str(compression_ratio),
        '-pix_fmt', 'yuv420p',           # Standard pixel format
        ]    
        , verbose=False, logger=None)
    
def read_frames_from_video(video_path, size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame = cv2.resize(frame, size)
        frames.append(frame)
        
    cap.release()
    
    return frames

def get_file_size(file_path, frame_count):
    avg_size = os.path.getsize(file_path)  # B

    # Convert image size to bits
    avg_size = avg_size * 8 
    
    # Calculate bpp (bits per pixel)
    bpp = avg_size / (224 * 224 * frame_count)
    return bpp


def compute_vgg_loss_over_frames(vgg_loss_fn, pred_frames, target_frames, batch_size=8):
    if vgg_loss_fn is None:
        return None
    losses = []
    for start in range(0, pred_frames.shape[0], batch_size):
        end = start + batch_size
        pred_batch = pred_frames[start:end].to(device)
        target_batch = target_frames[start:end].to(device)
        with torch.no_grad():
            loss_val = vgg_loss_fn(pred_batch, target_batch).detach().cpu().item()
        losses.append(loss_val)
    if len(losses) == 0:
        return None
    return float(np.mean(losses))

def process_video(video_path, compressed_video_path, compression_ratio, vgg_loss_fn=None):
    # Step 1: Extract and Resize Frames
    frames = extract_and_resize_frames(video_path)
    
    # Step 2: Compress the Frames into a Video
    compress_video_from_frames(frames, compressed_video_path, video_path, compression_ratio=compression_ratio)
    
    # Step 3: Read Compressed Video Frames
    compressed_frames = read_frames_from_video(compressed_video_path)
    
    # Ensure frames are in the same order and count
    if len(frames) != len(compressed_frames):
        print("Number of frames in original and compressed videos do not match.")
        compressed_frames = compressed_frames[:len(frames)]
        
    # assert len(frames) == len(compressed_frames), "Number of frames in original and compressed videos do not match."
    
    # Convert to tensors
    original_frames = torch.stack([torch.tensor(frame).permute(2, 0, 1).float() / 255.0 for frame in frames])
    compressed_frames = torch.stack([torch.tensor(frame).permute(2, 0, 1).float() / 255.0 for frame in compressed_frames])
    
    # transforms.ToPILImage()(original_frames[0].cpu()).save("./original_frames.jpg")
    # transforms.ToPILImage()(compressed_frames[0].cpu()).save("./compressed_frames.jpg")
    # das
    
    # Calculate PSNR and MS-SSIM
    psnr_values = psnr_batch(compressed_frames, original_frames)
    ms_ssim_values = ms_ssim_batch(compressed_frames, original_frames)
    
    # Compute VGG perceptual loss
    vgg_loss = compute_vgg_loss_over_frames(vgg_loss_fn, compressed_frames, original_frames)
    
    # Get compressed video size
    compressed_size = get_file_size(compressed_video_path, frame_count=len(compressed_frames))
    
    # Compute means of metrics
    mean_psnr = psnr_values.mean().item()
    mean_ms_ssim = np.mean(ms_ssim_values)
    
    return mean_psnr, mean_ms_ssim, compressed_size, vgg_loss

def main(original_video_path, compressed_video_path, quality_list):
    all_for_print = []
    vgg_loss_fn = build_vgg_loss(device=device, weights_path=vgg_weights_path)

    for quality in quality_list:
        
        psnrs = []
        ms_ssims = []
        sizes = []
        vgg_losses = []
        
        for filename in tqdm(sorted(os.listdir(original_video_path))):
            if filename.endswith('.mp4'):
                video_path = os.path.join(original_video_path, filename)
                
                mean_psnr, mean_ms_ssim, compressed_size, vgg_loss = process_video(
                    video_path,
                    compressed_video_path,
                    compression_ratio=quality,
                    vgg_loss_fn=vgg_loss_fn,
                )
                psnrs.append(mean_psnr)
                ms_ssims.append(mean_ms_ssim)
                sizes.append(compressed_size)
                if vgg_loss is not None:
                    vgg_losses.append(vgg_loss)

                
        msg_parts = [
            f'Quality: {quality}',
            f'Avg bpp: {np.mean(sizes):.2f}',
            f'Avg PSNR: {np.mean(psnrs):.2f}dB',
            f'Avg MS_SSIM: {np.mean(ms_ssims):.2f}dB',
        ]
        if len(vgg_losses) > 0:
            msg_parts.append(f'VGG Loss: {np.mean(vgg_losses):.4f}')
        print(", ".join(msg_parts))
        all_for_print.append([
            float(np.mean(sizes)),
            float(np.mean(psnrs)),
            float(np.mean(ms_ssims)),
            None if len(vgg_losses) == 0 else float(np.mean(vgg_losses))
        ])

    print(all_for_print)

# Usage
# original_video_path = '/data22/aho/MCL_JCV_dataset/'
original_video_path = '/data22/aho/UVG_dataset/'

compressed_video_path = './results/mp4/compressed_video.mp4'
quality_list = [i for i in range(6, 51, 2)]  # Different compression qualities
vgg_weights_path = None  # Optionally provide a local VGG19 weights path

main(original_video_path, compressed_video_path, quality_list)
