# Copyright 2024 Kiel University
#
# Evaluate MCUCoder compression on ADE20K semantic segmentation.
# Compress/decompress images with MCUCoder, run a pretrained segmenter,
# and report mean IoU (mIoU) plus estimated bpp using the
# quantization + Huffman scheme from video_enc_dec.py.

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from dahuffman import HuffmanCodec
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from mcucoder import MCUCoder


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MCUCoder on ADE20K semantic segmentation.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/data22/aho/ADEChallengeData2016",
        help="Root of ADE20K dataset containing images/ and annotations/ folders.",
    )
    parser.add_argument("--split", type=str, default="validation", choices=["training", "validation"], help="Dataset split.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to MCUCoder checkpoint (.pth/.ckpt).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--resize", type=int, default=224, help="Resize for MCUCoder input.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit number of images (0 = use all).")
    parser.add_argument(
        "--seg_model",
        type=str,
        default="nvidia/segformer-b5-finetuned-ade-640-640",
        help="HuggingFace SegFormer model id or local path (must output 150 classes).",
    )
    parser.add_argument(
        "--seg_local_only",
        action="store_true",
        help="Load segmentation model/processor from local cache only (no download); recommended in restricted environments.",
    )
    parser.add_argument(
        "--segment_resize",
        type=int,
        default=None,
        help="Optional spatial size (H=W) to upsample decoded images before feeding the segmenter "
        "(e.g., 512 or 640). Masks stay at --resize; predictions are rescaled back to mask size.",
    )
    parser.add_argument(
        "--reduce_labels",
        action="store_true",
        help="Apply ADE20K reduce_labels convention (background -> ignore=255, other labels shift down by 1). "
        "SegFormer ADE20K checkpoints expect this.",
    )
    return parser.parse_args()


class ADE20KDataset(Dataset):
    def __init__(self, root, split="validation", img_transform=None, mask_transform=None, reduce_labels=False):
        self.img_dir = Path(root) / "images" / split
        self.ann_dir = Path(root) / "annotations" / split
        if not self.img_dir.exists() or not self.ann_dir.exists():
            raise ValueError(f"Invalid ADE20K paths: {self.img_dir} or {self.ann_dir} not found")
        self.img_paths = sorted(self.img_dir.glob("*.jpg"))
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.reduce_labels = reduce_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.ann_dir / f"{img_path.stem}.png"
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        # convert mask to torch long tensor with class ids
        mask_arr = np.array(mask, dtype=np.int64)
        if self.reduce_labels:
            bg = mask_arr == 0
            mask_arr[mask_arr > 0] -= 1  # shift labels down by 1
            mask_arr[bg] = 255  # ignore background
        mask = torch.from_numpy(mask_arr)
        return image, mask


def build_segformer(model_name, device, local_only=False):
    try:
        processor = SegformerImageProcessor.from_pretrained(model_name, local_files_only=local_only)
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            local_files_only=local_only,
            use_safetensors=True,  # avoid torch.load on .bin when torch<2.6
        ).to(device).eval()
    except ValueError as exc:
        raise ValueError(
            f"Failed to load segmentation model '{model_name}'. "
            "Ensure safetensors weights are available locally (use --seg_local_only to avoid downloads) "
            "or upgrade torch to >=2.6 if only .bin weights exist. "
            f"Original error: {exc}"
        ) from exc
    return processor, model


def quantization(data, filter_number, codec, step=4):
    min_, max_ = codec["min"][filter_number], codec["max"][filter_number]
    data = (data - min_) / (max_ - min_)
    data = data * 255
    data = data.type(dtype=torch.uint8)

    quantization_step = step
    data = data / quantization_step
    data = data.type(dtype=torch.uint8)

    return data


def quantization_and_dequantization(data, filter_number, codec, step=4):
    min_, max_ = codec["min"][filter_number], codec["max"][filter_number]

    data = (data - min_) / (max_ - min_)
    data = data * 255
    data = data.type(dtype=torch.uint8)

    quantization_step = step
    data = data / quantization_step
    data = data.type(dtype=torch.uint8)
    data = data * quantization_step

    data = data / 255.0
    data = data * (max_ - min_) + min_
    return data


def quantization_and_huffman(data, filter_number, codec, step=4):
    data = data.reshape(-1)
    quantized_data = quantization(data, filter_number, codec, step).cpu().numpy()
    codec_obj = codec["codec"][filter_number]
    encoded = codec_obj.encode(quantized_data)
    return len(encoded) / 1024  # kilobytes


def create_codec(model, loader, device):
    codec_setting = {"min": {}, "max": {}, "codec": {}}
    try:
        sample_batch = next(iter(loader))
    except StopIteration:
        raise ValueError("No images available to build codec.")

    if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 1:
        images = sample_batch[0]
    else:
        images = sample_batch

    images = images.to(device)

    with torch.no_grad():
        encoded = model.encoder(images)

    for i in range(encoded.shape[1]):
        data = encoded[:, i, :, :].reshape(-1).detach().clone()
        min_, max_ = torch.min(data), torch.max(data)
        data = ((data - min_) / (max_ - min_) * 255).type(torch.uint8)
        data = (data / 4).type(torch.uint8).cpu().numpy()

        for j in range(0, 63):
            if j not in data:
                data = np.append(data, j)

        codec_obj = HuffmanCodec.from_data(data)
        codec_setting["min"][i], codec_setting["max"][i], codec_setting["codec"][i] = min_, max_, codec_obj

    return codec_setting


def update_confusion_matrix(conf_matrix, preds, targets, num_classes, ignore_index=255):
    # preds/targets are 2D tensors on CPU
    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes) & (preds >= 0) & (preds < num_classes)
    pred = preds[valid].to(torch.int64)
    targ = targets[valid].to(torch.int64)
    if pred.numel() == 0:
        return conf_matrix
    k = targ * num_classes + pred
    bincount = torch.bincount(k, minlength=num_classes * num_classes)
    bincount = bincount[: num_classes * num_classes]  # guard against stray labels inflating size
    conf_matrix += bincount.reshape((num_classes, num_classes))
    return conf_matrix


def compute_miou(conf_matrix):
    diag = torch.diag(conf_matrix)
    denom = conf_matrix.sum(1) + conf_matrix.sum(0) - diag
    valid = denom > 0
    iou = torch.zeros_like(diag, dtype=torch.float64)
    iou[valid] = diag[valid].double() / denom[valid].double()
    if valid.any():
        return iou[valid].mean().item()
    return float("nan")


def main():
    args = parse_args()
    torch.manual_seed(0)
    random.seed(10)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_transform = transforms.Compose(
        [
            transforms.Resize((args.resize, args.resize), antialias=True),
            transforms.ToTensor(),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize((args.resize, args.resize), interpolation=InterpolationMode.NEAREST),
        ]
    )
    # single-scale 224x224 (or args.resize) for both image and mask

    dataset = ADE20KDataset(
        root=args.dataset_root,
        split=args.split,
        img_transform=img_transform,
        mask_transform=mask_transform,
        reduce_labels=args.reduce_labels,
    )
    if len(dataset) == 0:
        raise ValueError(f"No images found in {args.dataset_root} for split {args.split}")
    if args.max_images and args.max_images > 0:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[: args.max_images]
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    model = MCUCoder().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    processor, seg_model = build_segformer(args.seg_model, device, local_only=args.seg_local_only)
    num_classes = seg_model.config.num_labels

    codec = create_codec(model, loader, device)
    used_filters = list(range(1, 13))
    to_pil = transforms.ToPILImage()
    segment_resize_tf = (
        transforms.Resize((args.segment_resize, args.segment_resize), interpolation=InterpolationMode.BILINEAR)
        if args.segment_resize
        else None
    )

    summary = {}

    for uf in used_filters:
        model.p = uf / 12.0
        model.replace_value = 0
        conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        bpp_values = []

        print(f"Evaluating used_filter={uf} on {len(dataset)} images...")
        with torch.no_grad():
            for images, masks in tqdm(loader, desc=f"uf={uf}", leave=False):
                images = images.to(device)
                masks = masks.to(device)

                encoded = model.encoder(images)
                encoded = model.rate_less(encoded)
                for b in range(encoded.shape[0]):
                    for ch in range(uf):
                        encoded[b, ch] = quantization_and_dequantization(encoded[b, ch], ch, codec)
                recon = model.decoder(encoded).clamp(0, 1)

                # BPP estimate via quantization + Huffman coding
                for b in range(encoded.shape[0]):
                    feature_sizes = []
                    for ch in range(uf):
                        feature_sizes.append(quantization_and_huffman(encoded[b, ch], ch, codec))
                    sample_bits = np.sum(feature_sizes) * 8 * 1024  # kb -> bits
                    sample_bpp = sample_bits / (args.resize * args.resize)
                    bpp_values.append(sample_bpp)

                recon_pils = [to_pil(img.cpu()) for img in recon]
                if segment_resize_tf:
                    recon_pils = [segment_resize_tf(pil) for pil in recon_pils]
                target_sizes = [(mask.shape[-2], mask.shape[-1]) for mask in masks]
                inputs = processor(images=recon_pils, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = seg_model(**inputs)
                reduce_applied = False
                try:
                    seg_maps = processor.post_process_semantic_segmentation(
                        outputs, target_sizes=target_sizes, reduce_labels=args.reduce_labels
                    )
                    reduce_applied = args.reduce_labels
                except TypeError:
                    seg_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

                if args.reduce_labels and not reduce_applied:
                    # Manually apply reduce_labels convention to predictions
                    adjusted_maps = []
                    for pred_map in seg_maps:
                        pred_map = pred_map.clone()
                        bg = pred_map == 0
                        pred_map[~bg] = pred_map[~bg] - 1
                        pred_map[bg] = 255
                        adjusted_maps.append(pred_map)
                    seg_maps = adjusted_maps

                for pred_map, gt_mask in zip(seg_maps, masks):
                    conf_matrix = update_confusion_matrix(conf_matrix, pred_map.cpu(), gt_mask.cpu(), num_classes)

        miou = compute_miou(conf_matrix)
        mean_bpp = float(np.mean(bpp_values)) if bpp_values else float("nan")
        summary[uf] = {"mIoU": miou, "bpp": mean_bpp}
        print(f"used_filter={uf}: mIoU={miou:.4f} | bpp≈{mean_bpp:.4f}")

    if summary:
        print("\n==== Summary over all filters ====")
        for uf in used_filters:
            if uf in summary:
                vals = summary[uf]
                print(f"used_filter={uf}: mIoU={vals['mIoU']:.4f} | bpp≈{vals['bpp']:.4f}")


if __name__ == "__main__":
    main()
