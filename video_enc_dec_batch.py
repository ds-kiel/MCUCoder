# Batch video encode/decode evaluation over a folder of videos.
# Reuses single-video utilities from video_enc_dec.py and aggregates metrics.

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms

from video_enc_dec import (
    MCUCoder,
    VGGPerceptualLoss,
    VideoDataset,
    create_codec,
    eval_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch video encode/decode and aggregate metrics.")
    parser.add_argument("--videos_dir", type=str, required=True, help="Folder containing videos to process.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (.pth or .ckpt).")
    parser.add_argument("--output_dir", type=str, default="./outputs_batch", help="Root output directory.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for dataloaders.")
    parser.add_argument("--resize", type=int, default=224, help="Resize square dimension for processing.")
    return parser.parse_args()


def get_video_list(videos_dir):
    exts = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg")
    videos = []
    for f in sorted(os.listdir(videos_dir)):
        if f.lower().endswith(exts):
            videos.append(os.path.join(videos_dir, f))
    return videos


def main():
    args = parse_args()
    torch.manual_seed(0)

    videos = get_video_list(args.videos_dir)
    if not videos:
        raise ValueError(f"No videos found in {args.videos_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MCUCoder().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    vgg_metric = VGGPerceptualLoss().to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize((args.resize, args.resize), antialias=True),
            transforms.ToTensor(),
        ]
    )

    agg = {uf: {"bpp": [], "psnr": [], "ms_ssim": [], "vgg": []} for uf in range(1, 13)}

    for used_filter in range(1, 13):
        print(f"\n=== Evaluating used_filter={used_filter} across all videos ===")
        model.p = used_filter / 12.0

        for video_path in videos:
            video_name = Path(video_path).stem
            dataset = VideoDataset(video_path=video_path, transform=transform)
            if len(dataset) == 0:
                print(f"Skipping {video_name}: no frames.")
                continue

            # Outputs per video under output_dir/video_name
            video_out_root = Path(args.output_dir) / video_name
            output_video_dir = video_out_root / "decoded_videos"
            output_video_dir.mkdir(parents=True, exist_ok=True)
            samples_dir = video_out_root / "decoded_frames_samples"
            samples_dir.mkdir(parents=True, exist_ok=True)

            codec = create_codec(dataset, model)

            frames_sizes, frames_psnrs, frames_ms_ssims, frames_vggs = eval_model(
                model,
                used_filter=used_filter,
                test_dataset=dataset,
                batch_size=args.batch_size,
                codec=codec,
                output_dir=str(video_out_root),
                output_video_dir=str(output_video_dir),
                vgg_metric=vgg_metric,
            )

            # Convert size list to bpp as in single-video script
            frames_sizes_bits = [size * 8 * 1024 for size in frames_sizes]
            frames_bpps = [size / (args.resize * args.resize) for size in frames_sizes_bits]

            agg[used_filter]["bpp"].extend(frames_bpps)
            agg[used_filter]["psnr"].extend(frames_psnrs)
            agg[used_filter]["ms_ssim"].extend(frames_ms_ssims)
            agg[used_filter]["vgg"].extend(frames_vggs)

        # Per-filter summary right after processing all videos
        bpps = agg[used_filter]["bpp"]
        psnrs = agg[used_filter]["psnr"]
        msssims = agg[used_filter]["ms_ssim"]
        vggs = agg[used_filter]["vgg"]
        if bpps:
            print(
                f"used_filter={used_filter} summary: "
                f"bpp={np.mean(bpps):.4f}, "
                f"PSNR={np.mean(psnrs):.2f}dB, "
                f"MS-SSIM={np.mean(msssims):.2f}dB, "
                f"VGG={np.mean(vggs) if vggs else float('nan'):.4f}"
            )
        else:
            print(f"used_filter={used_filter} summary: no data.")

    print("\nAggregated metrics across all videos:")
    for uf in range(1, 13):
        bpps = agg[uf]["bpp"]
        psnrs = agg[uf]["psnr"]
        msssims = agg[uf]["ms_ssim"]
        vggs = agg[uf]["vgg"]
        if not bpps:
            print(f"used_filter={uf}: no data.")
            continue
        print(
            f"used_filter={uf}: "
            f"bpp={np.mean(bpps):.4f}, "
            f"PSNR={np.mean(psnrs):.2f}dB, "
            f"MS-SSIM={np.mean(msssims):.2f}dB, "
            f"VGG={np.mean(vggs) if vggs else float('nan'):.4f}"
        )


if __name__ == "__main__":
    main()
