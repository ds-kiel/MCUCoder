# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from dahuffman import HuffmanCodec
from mcucoder import MCUCoder, CustomImageDataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torchvision.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate classifier robustness on compressed images.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory of images to evaluate.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to MCUCoder checkpoint (.pth).")
    parser.add_argument("--classifier_models", type=str, default="deit_base_patch16_224,swin_base_patch4_window7_224",
                        help="Comma-separated timm model names.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation.")
    parser.add_argument("--resize", type=int, default=224, help="Spatial resize for MCUCoder input.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit number of images (0 = use all).")
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
    return len(encoded) / 1024


def create_codec(model, loader, device):
    codec_setting = {"min": {}, "max": {}, "codec": {}}

    try:
        sample_batch = next(iter(loader))
    except StopIteration:
        raise ValueError("No images available to build codec.")

    if isinstance(sample_batch, (list, tuple)) and len(sample_batch) == 2:
        images, _ = sample_batch
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

    # Prefer ImageFolder (with labels) if subfolders exist; fallback to unlabeled custom dataset.
    dataset_type = "CustomImageDataset"
    if any(Path(args.images_dir).iterdir()):
        try:
            dataset = ImageFolder(root=args.images_dir, transform=base_transform)
            has_labels = True
            dataset_type = "ImageFolder"
        except Exception:
            dataset = CustomImageDataset(root_dir=args.images_dir, transform=base_transform)
            has_labels = False
    else:
        dataset = CustomImageDataset(root_dir=args.images_dir, transform=base_transform)
        has_labels = False

    if len(dataset) == 0:
        raise ValueError(f"No images found in {args.images_dir}")
    if args.max_images and args.max_images > 0:
        # For ImageFolder, the samples list lives in dataset.samples
        if hasattr(dataset, "samples"):
            random.shuffle(dataset.samples)
            dataset.samples = dataset.samples[: args.max_images]
        elif hasattr(dataset, "image_files"):
            random.shuffle(dataset.image_files)
            dataset.image_files = dataset.image_files[: args.max_images]

    print(f"Dataset type: {dataset_type}, size={len(dataset)}, labels_available={has_labels}")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    model = MCUCoder().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    codec = create_codec(model, loader, device)
    used_filters = list(range(1, 13))

    classifier_names = [name.strip() for name in args.classifier_models.split(",") if name.strip()]
    classifiers = build_classifiers(classifier_names, device)

    to_pil = transforms.ToPILImage()
    sample_dir = Path("class_eval_samples")
    sample_dir.mkdir(exist_ok=True)

    summary = {}

    for uf in used_filters:
        model.p = uf / 12.0
        model.replace_value = 0
        agreement = {name: [] for name in classifiers}
        acc_orig = {name: [] for name in classifiers}
        acc_recon = {name: [] for name in classifiers}
        bpp_values = []

        print(f"Evaluating used_filter={uf} on {len(dataset)} images...")
        saved_sample = False

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"uf={uf}", leave=False):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, labels = batch
                else:
                    images, labels = batch, None

                images = images.to(device)
                labels = labels.to(device) if labels is not None else None

                encoded = model.encoder(images)
                encoded = model.rate_less(encoded)
                for b in range(encoded.shape[0]):
                    for ch in range(uf):
                        encoded[b, ch] = quantization_and_dequantization(encoded[b, ch], ch, codec)
                recon = model.decoder(encoded).clamp(0, 1)

                # Estimate bpp using quantization + Huffman coding (same as video_enc_dec)
                for b in range(encoded.shape[0]):
                    feature_sizes = []
                    for ch in range(uf):
                        feature_sizes.append(quantization_and_huffman(encoded[b, ch], ch, codec))
                    sample_bits = np.sum(feature_sizes) * 8 * 1024  # kb -> bits
                    sample_bpp = sample_bits / (args.resize * args.resize)
                    bpp_values.append(sample_bpp)

                if not saved_sample:
                    save_image(images[0].detach().cpu(), sample_dir / f"uf{uf}_input.png")
                    save_image(recon[0].detach().cpu(), sample_dir / f"uf{uf}_decoded.png")
                    saved_sample = True

                for name, payload in classifiers.items():
                    clf = payload["model"]
                    tform = payload["transform"]
                    orig_batch = torch.stack([tform(to_pil(img.cpu())) for img in images]).to(device)
                    rec_batch = torch.stack([tform(to_pil(img.cpu())) for img in recon]).to(device)
                    orig_pred = clf(orig_batch).argmax(dim=1)
                    rec_pred = clf(rec_batch).argmax(dim=1)
                    matches = (orig_pred == rec_pred).int().cpu().tolist()
                    agreement[name].extend(matches)
                    if labels is not None:
                        acc_orig[name].extend((orig_pred == labels).int().cpu().tolist())
                        acc_recon[name].extend((rec_pred == labels).int().cpu().tolist())
                # live print after each batch for quick monitoring
                for name, matches in agreement.items():
                    if matches:
                        acc = np.mean(matches) * 100
                        # print(f"running: used_filter={uf}, classifier={name}, top-1 agreement={acc:.2f}%")

        summary[uf] = {}
        summary[uf]["bpp"] = float(np.mean(bpp_values)) if bpp_values else float("nan")
        for name, matches in agreement.items():
            acc = np.mean(matches) * 100 if matches else float("nan")
            if acc_orig[name]:
                acc_o = np.mean(acc_orig[name]) * 100
                acc_r = np.mean(acc_recon[name]) * 100
                summary[uf][name] = {
                    "agreement": acc,
                    "orig_acc": acc_o,
                    "recon_acc": acc_r,
                }
                print(
                    f"classifier: {name}, top-1 agreement original vs reconstructed: {acc:.2f}% "
                    f"| orig top-1 acc: {acc_o:.2f}% | recon top-1 acc: {acc_r:.2f}% "
                    f"| bpp≈{summary[uf]['bpp']:.4f} (used_filter={uf})"
                )
            else:
                summary[uf][name] = {
                    "agreement": acc,
                    "orig_acc": None,
                    "recon_acc": None,
                }
                print(
                    f"classifier: {name}, top-1 agreement original vs reconstructed: {acc:.2f}% "
                    f"| bpp≈{summary[uf]['bpp']:.4f} (used_filter={uf}) [no labels]"
                )

    # Final summary
    print("\n==== Summary over all filters ====")
    for uf in used_filters:
        if uf in summary:
            print(f"used_filter={uf}: bpp≈{summary[uf]['bpp']:.4f}")
            for name, vals in summary[uf].items():
                if name == "bpp":
                    continue
                if vals["orig_acc"] is not None:
                    print(
                        f"  {name}: agreement={vals['agreement']:.2f}% | "
                        f"orig_acc={vals['orig_acc']:.2f}% | recon_acc={vals['recon_acc']:.2f}%"
                    )
                else:
                    print(f"  {name}: agreement={vals['agreement']:.2f}% (no labels)")


if __name__ == "__main__":
    main()
