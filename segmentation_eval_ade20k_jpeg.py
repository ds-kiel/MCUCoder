# Copyright 2024 Kiel University
#
# Evaluate JPEG compression on ADE20K semantic segmentation.
# Compress each resized image to JPEG at given qualities, run a pretrained
# segmenter, and report mean IoU (mIoU) plus estimated bpp from JPEG size.

import argparse
import io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate JPEG baseline on ADE20K semantic segmentation.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/data22/aho/ADEChallengeData2016",
        help="Root of ADE20K dataset containing images/ and annotations/ folders.",
    )
    parser.add_argument("--split", type=str, default="validation", choices=["training", "validation"], help="Dataset split.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--resize", type=int, default=224, help="Resize for detector input.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit number of images (0 = use all).")
    parser.add_argument(
        "--qualities",
        type=str,
        default="1,6,11,21,36",
        help="Comma-separated JPEG qualities to evaluate.",
    )
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
    return parser.parse_args()


class ADE20KDataset(Dataset):
    def __init__(self, root, split="validation", img_transform=None, mask_transform=None):
        self.img_dir = Path(root) / "images" / split
        self.ann_dir = Path(root) / "annotations" / split
        if not self.img_dir.exists() or not self.ann_dir.exists():
            raise ValueError(f"Invalid ADE20K paths: {self.img_dir} or {self.ann_dir} not found")
        self.img_paths = sorted(self.img_dir.glob("*.jpg"))
        self.img_transform = img_transform
        self.mask_transform = mask_transform

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
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask


def build_segformer(model_name, device, local_only=False):
    try:
        processor = SegformerImageProcessor.from_pretrained(model_name, local_files_only=local_only)
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            local_files_only=local_only,
            use_safetensors=True,
        ).to(device).eval()
    except ValueError as exc:
        raise ValueError(
            f"Failed to load segmentation model '{model_name}'. "
            "Ensure safetensors weights are available locally (use --seg_local_only) "
            "or upgrade torch to >=2.6 if only .bin weights exist. "
            f"Original error: {exc}"
        ) from exc
    return processor, model


def parse_quality_list(qualities_str):
    qualities = []
    for item in qualities_str.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            qualities.append(int(item))
        except ValueError:
            raise ValueError(f"Invalid quality value: {item}")
    if not qualities:
        raise ValueError("No valid JPEG quality values provided.")
    return qualities


def compress_pil_to_jpeg(pil_img, quality):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    size_bytes = buf.tell()
    buf.seek(0)
    comp_pil = Image.open(buf).convert("RGB")
    return comp_pil, size_bytes


def update_confusion_matrix(conf_matrix, preds, targets, num_classes, ignore_index=255):
    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes) & (preds >= 0) & (preds < num_classes)
    pred = preds[valid].to(torch.int64)
    targ = targets[valid].to(torch.int64)
    if pred.numel() == 0:
        return conf_matrix
    k = targ * num_classes + pred
    bincount = torch.bincount(k, minlength=num_classes * num_classes)
    bincount = bincount[: num_classes * num_classes]
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
    qualities = parse_quality_list(args.qualities)

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

    dataset = ADE20KDataset(root=args.dataset_root, split=args.split, img_transform=img_transform, mask_transform=mask_transform)
    if len(dataset) == 0:
        raise ValueError(f"No images found in {args.dataset_root} for split {args.split}")
    if args.max_images and args.max_images > 0:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[: args.max_images]
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    processor, seg_model = build_segformer(args.seg_model, device, local_only=args.seg_local_only)
    num_classes = seg_model.config.num_labels

    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    summary = {}

    for quality in qualities:
        conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        bpp_values = []

        print(f"Evaluating JPEG quality={quality} on {len(dataset)} images...")
        with torch.no_grad():
            for images, masks in tqdm(loader, desc=f"Q={quality}", leave=False):
                images = images.to(device)
                masks = masks.to(device)

                orig_pils = [to_pil(img.cpu()) for img in images]
                comp_pils = []
                for pil_img in orig_pils:
                    comp_img, size_bytes = compress_pil_to_jpeg(pil_img, quality)
                    comp_pils.append(comp_img)
                    bpp_values.append(size_bytes * 8.0 / (args.resize * args.resize))

                target_sizes = [(mask.shape[-2], mask.shape[-1]) for mask in masks]
                inputs = processor(images=comp_pils, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = seg_model(**inputs)
                seg_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

                for pred_map, gt_mask in zip(seg_maps, masks):
                    conf_matrix = update_confusion_matrix(conf_matrix, pred_map.cpu(), gt_mask.cpu(), num_classes)

        miou = compute_miou(conf_matrix)
        mean_bpp = float(np.mean(bpp_values)) if bpp_values else float("nan")
        summary[quality] = {"mIoU": miou, "bpp": mean_bpp}
        print(f"JPEG quality={quality}: mIoU={miou:.4f} | bpp≈{mean_bpp:.4f}")

    if summary:
        print("\n==== Summary over all qualities ====")
        for quality in qualities:
            if quality in summary:
                vals = summary[quality]
                print(f"JPEG quality={quality}: mIoU={vals['mIoU']:.4f} | bpp≈{vals['bpp']:.4f}")


if __name__ == "__main__":
    main()
