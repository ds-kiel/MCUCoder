# Copyright 2024 Kiel University
#
# Evaluate MCUCoder on COCO detection by compressing/decompressing images,
# running a pretrained detector, and reporting mAP plus estimated bpp
# using the same quantization + Huffman scheme as video_enc_dec.py.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from dahuffman import HuffmanCodec
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from tqdm import tqdm

from mcucoder import MCUCoder



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MCUCoder on COCO object detection.")
    parser.add_argument("--images_dir", type=str, required=True, help="COCO images directory (e.g., val2017).")
    parser.add_argument(
        "--ann_file",
        type=str,
        required=True,
        help="COCO annotation file (e.g., annotations/instances_val2017.json).",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to MCUCoder checkpoint (.pth/.ckpt).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--resize", type=int, default=224, help="Resize for MCUCoder input.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit number of images (0 = use all).")
    parser.add_argument(
        "--detector",
        type=str,
        default="fasterrcnn_resnet50_fpn",
        choices=["fasterrcnn_resnet50_fpn"],
        help="Detection model to evaluate.",
    )
    return parser.parse_args()


class CocoDetWithIds(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        # Avoid double-transform: store user transform separately and keep base class transform=None
        super().__init__(root=root, annFile=annFile, transform=None)
        self.image_ids = self.ids
        self.user_transform = transform

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)  # returns PIL image because base transform=None
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        orig_w, orig_h = img_info["width"], img_info["height"]
        if self.user_transform:
            img = self.user_transform(img)
        return img, target, image_id, (orig_w, orig_h)


def collate_fn(batch):
    images, targets, ids, sizes = zip(*batch)
    return list(images), list(targets), list(ids), list(sizes)


def build_detector(name, device):
    if name == "fasterrcnn_resnet50_fpn":
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported detector: {name}")
    return model.to(device).eval()


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

    images, _, _, _ = sample_batch
    images = torch.stack(images).to(device)

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

    codec_loader = DataLoader(dataset, batch_size=min(args.batch_size, 16), shuffle=False, num_workers=2, collate_fn=collate)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    model = MCUCoder().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    detector = build_detector(args.detector, device)

    codec = create_codec(model, codec_loader, device)
    used_filters = list(range(1, 13))
    summary = {}

    for uf in used_filters:
        model.p = uf / 12.0
        model.replace_value = 0

        bpp_values = []
        recon_results = []

        print(f"Evaluating used_filter={uf} on {len(dataset)} images...")
        with torch.no_grad():
            for images, targets, image_ids, sizes in tqdm(eval_loader, desc=f"uf={uf}", leave=False):
                images = [img.to(device) for img in images]
                stacked = torch.stack(images)

                encoded = model.encoder(stacked)
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

                recon_inputs = [img for img in recon]  # already on device
                recon_preds = detector(recon_inputs)
                recon_results.extend(to_coco_results(recon_preds, image_ids, sizes, args.resize))

        recon_metrics = evaluate_coco(coco_gt, recon_results)
        mean_bpp = float(np.mean(bpp_values)) if bpp_values else float("nan")
        print(f"used_filter={uf}: recon AP50={recon_metrics['AP50']:.4f} | bpp≈{mean_bpp:.4f}")
        summary[uf] = {"AP50": recon_metrics["AP50"], "bpp": mean_bpp}

    if summary:
        print("\n==== Summary over all filters ====")
        for uf in used_filters:
            if uf in summary:
                vals = summary[uf]
                print(f"used_filter={uf}: AP50={vals['AP50']:.4f} | bpp≈{vals['bpp']:.4f}")


if __name__ == "__main__":
    main()
