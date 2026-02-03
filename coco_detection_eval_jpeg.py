# Copyright 2024 Kiel University
#
# Evaluate object detection on JPEG-compressed COCO images. Compress each image
# to the given JPEG qualities, run a pretrained detector, and report AP50 and
# approximate bpp (based on compressed size after resizing).

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import io
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from tqdm import tqdm
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate COCO detection on JPEG-compressed images.")
    parser.add_argument("--images_dir", type=str, required=True, help="COCO images directory (e.g., val2017).")
    parser.add_argument(
        "--ann_file",
        type=str,
        required=True,
        help="COCO annotation file (e.g., annotations/instances_val2017.json).",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--resize", type=int, default=224, help="Resize for detector input.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit number of images (0 = use all).")
    parser.add_argument(
        "--qualities",
        type=str,
        default="1,6,11,21,36",
        help="Comma-separated JPEG qualities to evaluate.",
    )
    return parser.parse_args()


class CocoDetWithIds(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root=root, annFile=annFile, transform=None)
        self.image_ids = self.ids
        self.user_transform = transform

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)  # PIL image
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        orig_w, orig_h = img_info["width"], img_info["height"]
        if self.user_transform:
            img = self.user_transform(img)
        return img, target, image_id, (orig_w, orig_h)


def collate_fn(batch):
    images, targets, ids, sizes = zip(*batch)
    return list(images), list(targets), list(ids), list(sizes)


def build_detector(device):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    return model.to(device).eval()


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


def to_coco_results(predictions, image_ids, sizes, resize):
    results = []
    for pred, img_id, (orig_w, orig_h) in zip(predictions, image_ids, sizes):
        boxes = pred["boxes"].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()
        labels = pred["labels"].detach().cpu().numpy()
        scale_x = orig_w / float(resize)
        scale_y = orig_h / float(resize)
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y
            results.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return results


def evaluate_coco(coco_gt, detections, iou_type="bbox"):
    if len(detections) == 0:
        return {"AP50": float("nan")}
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {"AP50": coco_eval.stats[1]}  # AP @ IoU=0.50


def main():
    args = parse_args()
    torch.manual_seed(0)
    random.seed(10)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    qualities = parse_quality_list(args.qualities)

    base_transform = transforms.Compose(
        [
            transforms.Resize((args.resize, args.resize), antialias=True),
            transforms.ToTensor(),
        ]
    )

    dataset = CocoDetWithIds(root=args.images_dir, annFile=args.ann_file, transform=base_transform)
    coco_gt = dataset.coco
    if args.max_images and args.max_images > 0:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        dataset = Subset(dataset, indices[: args.max_images])
        coco_gt = dataset.dataset.coco

    collate = collate_fn
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    detector = build_detector(device)
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    summary = {}

    for quality in qualities:
        print(f"Evaluating JPEG quality={quality} on {len(dataset)} images...")
        detections = []
        bpp_values = []

        with torch.no_grad():
            for images, targets, image_ids, sizes in tqdm(loader, desc=f"Q={quality}", leave=False):
                orig_pils = [to_pil(img.cpu()) for img in images]
                comp_tensors = []
                for pil_img in orig_pils:
                    comp_pil, size_bytes = compress_pil_to_jpeg(pil_img, quality)
                    comp_tensors.append(to_tensor(comp_pil))
                    bpp_values.append(size_bytes * 8.0 / (args.resize * args.resize))

                comp_tensors = [img.to(device) for img in comp_tensors]
                preds = detector(comp_tensors)
                detections.extend(to_coco_results(preds, image_ids, sizes, args.resize))

        metrics = evaluate_coco(coco_gt, detections)
        mean_bpp = float(np.mean(bpp_values)) if bpp_values else float("nan")
        print(f"JPEG quality={quality}: AP50={metrics['AP50']:.4f} | bpp≈{mean_bpp:.4f}")
        summary[quality] = {"AP50": metrics["AP50"], "bpp": mean_bpp}

    if summary:
        print("\n==== Summary over all qualities ====")
        for quality in qualities:
            if quality in summary:
                vals = summary[quality]
                print(f"JPEG quality={quality}: AP50={vals['AP50']:.4f} | bpp≈{vals['bpp']:.4f}")


if __name__ == "__main__":
    main()
