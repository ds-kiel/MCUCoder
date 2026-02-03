import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import io
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from mcucoder import CustomImageDataset
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate classifier robustness on JPEG-compressed images.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory of images to evaluate.")
    parser.add_argument(
        "--classifier_models",
        type=str,
        default="deit_base_patch16_224,swin_base_patch4_window7_224",
        help="Comma-separated timm model names.",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation.")
    parser.add_argument("--resize", type=int, default=224, help="Spatial resize for classifier input.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit number of images (0 = use all).")
    parser.add_argument(
        "--qualities",
        type=str,
        default="1,6,11,21,36",
        help="Comma-separated JPEG quality levels (default matches folder_mjpeg).",
    )
    return parser.parse_args()


def build_classifiers(model_names, device):
    import timm
    from timm.data import resolve_model_data_config
    from timm.data.transforms_factory import create_transform

    classifiers = {}
    for name in model_names:
        model = timm.create_model(name, pretrained=True).to(device)
        model.eval()
        config = resolve_model_data_config(model)
        transform = create_transform(**config, is_training=False)
        classifiers[name] = {"model": model, "transform": transform}
    return classifiers


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
    return Image.open(buf).convert("RGB"), size_bytes


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
    dataset_type = "CustomImageDataset"
    has_labels = False
    try:
        dataset = ImageFolder(root=args.images_dir, transform=base_transform)
        dataset_type = "ImageFolder"
        has_labels = True
    except Exception:
        dataset = CustomImageDataset(root_dir=args.images_dir, transform=base_transform)
        has_labels = False

    if len(dataset) == 0:
        raise ValueError(f"No images found in {args.images_dir}")
    if args.max_images and args.max_images > 0:
        if hasattr(dataset, "samples"):
            random.shuffle(dataset.samples)
            dataset.samples = dataset.samples[: args.max_images]
        elif hasattr(dataset, "image_files"):
            random.shuffle(dataset.image_files)
            dataset.image_files = dataset.image_files[: args.max_images]

    print(f"Dataset type: {dataset_type}, size={len(dataset)}, labels_available={has_labels}")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    classifier_names = [name.strip() for name in args.classifier_models.split(",") if name.strip()]
    classifiers = build_classifiers(classifier_names, device)

    to_pil = transforms.ToPILImage()
    summary = {}

    for quality in qualities:
        print(f"Evaluating JPEG quality={quality} on {len(dataset)} images...")
        agreement = {name: [] for name in classifiers}
        acc_orig = {name: [] for name in classifiers}
        acc_comp = {name: [] for name in classifiers}
        compressed_sizes_bytes = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Q={quality}", leave=False):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, labels = batch
                else:
                    images, labels = batch, None
                images = images.to(device)
                labels = labels.to(device) if labels is not None else None

                orig_pils = [to_pil(img.cpu()) for img in images]
                compressed_pairs = [compress_pil_to_jpeg(pil_img, quality) for pil_img in orig_pils]
                compressed_pils = [pair[0] for pair in compressed_pairs]
                compressed_sizes_bytes.extend([pair[1] for pair in compressed_pairs])

                for name, payload in classifiers.items():
                    clf = payload["model"]
                    tform = payload["transform"]

                    orig_batch = torch.stack([tform(pil_img) for pil_img in orig_pils]).to(device)
                    comp_batch = torch.stack([tform(pil_img) for pil_img in compressed_pils]).to(device)

                    orig_pred = clf(orig_batch).argmax(dim=1)
                    comp_pred = clf(comp_batch).argmax(dim=1)
                    matches = (orig_pred == comp_pred).int().cpu().tolist()
                    agreement[name].extend(matches)
                    if labels is not None:
                        acc_orig[name].extend((orig_pred == labels).int().cpu().tolist())
                        acc_comp[name].extend((comp_pred == labels).int().cpu().tolist())

        avg_size_kb = (np.mean(compressed_sizes_bytes) / 1024) if compressed_sizes_bytes else float("nan")
        avg_bpp = (np.mean(compressed_sizes_bytes) * 8 / (args.resize * args.resize)) if compressed_sizes_bytes else float("nan")
        summary[quality] = {}
        for name, matches in agreement.items():
            agr = np.mean(matches) * 100 if matches else float("nan")
            if acc_orig[name]:
                acc_o = np.mean(acc_orig[name]) * 100
                acc_c = np.mean(acc_comp[name]) * 100
                summary[quality][name] = {
                    "agreement": agr,
                    "orig_acc": acc_o,
                    "comp_acc": acc_c,
                    "avg_size_kb": avg_size_kb,
                    "avg_bpp": avg_bpp,
                }
                print(
                    f"classifier: {name}, top-1 agreement orig vs JPEG(Q={quality}): {agr:.2f}% "
                    f"| orig acc: {acc_o:.2f}% | JPEG acc: {acc_c:.2f}% | avg size={avg_size_kb:.2f} KB | avg bpp={avg_bpp:.4f}"
                )
            else:
                summary[quality][name] = {
                    "agreement": agr,
                    "orig_acc": None,
                    "comp_acc": None,
                    "avg_size_kb": avg_size_kb,
                    "avg_bpp": avg_bpp,
                }
                print(
                    f"classifier: {name}, top-1 agreement orig vs JPEG(Q={quality}): {agr:.2f}% "
                    f"| avg size={avg_size_kb:.2f} KB | avg bpp={avg_bpp:.4f} (no labels)"
                )

    print("\n==== Summary over all qualities ====")
    for quality in qualities:
        if quality in summary:
            print(f"JPEG quality={quality}:")
            for name, vals in summary[quality].items():
                if vals["orig_acc"] is not None:
                    print(
                        f"  {name}: agreement={vals['agreement']:.2f}% | "
                        f"orig_acc={vals['orig_acc']:.2f}% | JPEG_acc={vals['comp_acc']:.2f}% | "
                        f"avg size={vals['avg_size_kb']:.2f} KB | avg bpp={vals['avg_bpp']:.4f}"
                    )
                else:
                    print(
                        f"  {name}: agreement={vals['agreement']:.2f}% | "
                        f"avg size={vals['avg_size_kb']:.2f} KB | avg bpp={vals['avg_bpp']:.4f} (no labels)"
                    )


if __name__ == "__main__":
    main()
