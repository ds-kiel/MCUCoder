# Copyright 2024 Kiel University
#
# Evaluate MCUCoder compression on LFW face verification using ArcFace (InsightFace).
# Uses match/mismatch DevTrain to choose a threshold, then reports accuracy on DevTest
# plus estimated bpp (quantization + Huffman, same as video_enc_dec).

import argparse
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from dahuffman import HuffmanCodec
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from mcucoder import MCUCoder


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MCUCoder on LFW verification with ArcFace.")
    parser.add_argument("--dataset_root", type=str, default="/data22/aho/LFW", help="Root of LFW dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to MCUCoder checkpoint (.pth/.ckpt).")
    parser.add_argument(
        "--arcface_model",
        type=str,
        default="glint360k_r100fc_1.0",
        help="Path to ArcFace ONNX or model_zoo id (InsightFace). Defaults to glint360k_r100fc_1.0 and will download if missing.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding/decoding.")
    parser.add_argument("--resize", type=int, default=224, help="Resize for MCUCoder input.")
    parser.add_argument("--max_images", type=int, default=0, help="Optional limit on number of unique images (0 = all).")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    return parser.parse_args()


# ---------------- LFW pairs ----------------
def read_match_pairs(csv_path):
    pairs = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            i1 = int(row["imagenum1"])
            i2 = int(row["imagenum2"])
            pairs.append((f"{name}_{i1:04d}.jpg", f"{name}_{i2:04d}.jpg"))
    return pairs


def read_mismatch_pairs(csv_path):
    pairs = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = list(row.keys())
            # Typical: name, imagenum1, name.1, imagenum2
            if len(keys) >= 4:
                n1 = row[keys[0]]
                i1 = int(row[keys[1]])
                n2 = row[keys[2]]
                i2 = int(row[keys[3]])
            else:
                # Fallback to explicit names if present
                n1 = row.get("name", "")
                i1 = int(row.get("imagenum1", "0"))
                n2 = row.get("name.1", row.get("name2", row.get("name_1", "")))
                i2 = int(row.get("imagenum2", "0"))
            pairs.append((f"{n1}_{i1:04d}.jpg", f"{n2}_{i2:04d}.jpg"))
    return pairs


def build_image_path(root, fname):
    # LFW structure: root/lfw-deepfunneled/lfw-deepfunneled/{person}/{person_0001.jpg}
    base = Path(root) / "lfw-deepfunneled"
    if (base / "lfw-deepfunneled").exists():
        base = base / "lfw-deepfunneled"
    person = "_".join(fname.split("_")[:-1])  # drop trailing imaginenum
    img_dir = base / person
    return img_dir / fname


# ---------------- Dataset ----------------
class LFWImageDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(path)


# ---------------- MCUCoder quantization helpers ----------------
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

    images = sample_batch[0].to(device)

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


# ---------------- ArcFace ----------------
def load_arcface(model_id, device):
    import insightface
    import urllib.request
    import shutil

    def download_arcface_onnx(dest_path, model_name):
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        urls = [
            # Current InsightFace hosting (Hugging Face mirror)
            f"https://huggingface.co/deepinsight/insightface/resolve/main/models/{model_name}/model.onnx",
            f"https://huggingface.co/deepinsight/insightface/resolve/main/models/{model_name}/{model_name}.onnx",
            # Legacy GitHub release locations (keep a few versions for safety)
            f"https://github.com/deepinsight/insightface/releases/download/v1.0.2/models/{model_name}/model.onnx",
            f"https://github.com/deepinsight/insightface/releases/download/v1.0.2/{model_name}.onnx",
            f"https://github.com/deepinsight/insightface/releases/download/v1.0.1/models/{model_name}/model.onnx",
            f"https://github.com/deepinsight/insightface/releases/download/v1.0.1/{model_name}.onnx",
            f"https://github.com/deepinsight/insightface/releases/download/v1.0.0/models/{model_name}/model.onnx",
            f"https://github.com/deepinsight/insightface/releases/download/v1.0.0/{model_name}.onnx",
        ]
        for url in urls:
            try:
                print(f"Attempting to download ArcFace ONNX from {url} ...")
                with urllib.request.urlopen(url) as resp, open(dest_path, "wb") as out_f:
                    shutil.copyfileobj(resp, out_f)
                if dest_path.exists() and dest_path.stat().st_size > 0:
                    print(f"Downloaded ArcFace model to {dest_path}")
                    return dest_path
            except Exception as e:
                print(f"Download failed from {url}: {e}")
        return None

    # If a path is provided but missing, fall back to default zoo id
    model_arg = model_id
    if Path(model_id).suffix.lower() in {".onnx", ".pkl", ".pt"} and not Path(model_id).exists():
        print(f"ArcFace path '{model_id}' not found; falling back to default zoo id 'glint360k_r100fc_1.0'.")
        model_arg = "glint360k_r100fc_1.0"

    model = insightface.model_zoo.get_model(model_arg)
    if model is None:
        # Try manual download if model_arg looks like a zoo id
        if Path(model_arg).suffix.lower() not in {".onnx", ".pkl", ".pt"}:
            cache_dir = Path.home() / ".insightface" / "models" / model_arg
            dest = cache_dir / "model.onnx"
            downloaded = download_arcface_onnx(dest, model_arg)
            if downloaded and downloaded.exists():
                model = insightface.model_zoo.get_model(str(downloaded))

    if model is None:
        # Try a stable, smaller ArcFace checkpoint if the requested one is unavailable.
        fallback_id = "arcface_r100_v1"
        if model_arg != fallback_id:
            print(f"Falling back to ArcFace model id '{fallback_id}'.")
            model = insightface.model_zoo.get_model(fallback_id)
            if model is None:
                cache_dir = Path.home() / ".insightface" / "models" / fallback_id
                dest = cache_dir / "model.onnx"
                downloaded = download_arcface_onnx(dest, fallback_id)
                if downloaded and downloaded.exists():
                    model = insightface.model_zoo.get_model(str(downloaded))

    if model is None:
        raise RuntimeError(
            f"Failed to load ArcFace model '{model_arg}'. "
            "If network is restricted, download an ArcFace ONNX (e.g., glint360k_r100fc_1.0 or arcface_r100_v1) on another machine, "
            "place it under ~/.insightface/models/<model_id>/model.onnx, or pass --arcface_model /path/to/model.onnx. "
            "onnxruntime/onnxruntime-gpu must be installed."
        )
    ctx_id = 0 if torch.cuda.is_available() else -1
    model.prepare(ctx_id=ctx_id)
    return model


def arcface_embed(model, pil_img):
    # Resize to 112x112, RGB->BGR, float32
    img = pil_img.resize((112, 112), resample=Image.BILINEAR)
    img = np.asarray(img, dtype=np.float32)
    img = img[:, :, ::-1]  # BGR
    embedding = model.get_embedding(img).flatten()
    norm = np.linalg.norm(embedding) + 1e-12
    return embedding / norm


# ---------------- Metrics ----------------
def cosine_similarity(a, b):
    return float(np.dot(a, b))


def best_threshold(pos_scores, neg_scores):
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    best_acc = -1.0
    best_t = 0.0
    for t in scores:
        preds = (scores >= t).astype(np.float32)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, best_acc


def eval_accuracy(pos_scores, neg_scores, threshold):
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    preds = (scores >= threshold).astype(np.float32)
    return float((preds == labels).mean())


# ---------------- Main ----------------
def main():
    args = parse_args()
    torch.manual_seed(0)
    random.seed(10)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read pairs
    root = Path(args.dataset_root)
    pos_train = read_match_pairs(root / "matchpairsDevTrain.csv")
    neg_train = read_mismatch_pairs(root / "mismatchpairsDevTrain.csv")
    pos_test = read_match_pairs(root / "matchpairsDevTest.csv")
    neg_test = read_mismatch_pairs(root / "mismatchpairsDevTest.csv")

    # Collect unique image paths
    all_fnames = set()
    for a, b in pos_train + pos_test:
        all_fnames.add(a)
        all_fnames.add(b)
    for a, b in neg_train + neg_test:
        all_fnames.add(a)
        all_fnames.add(b)
    all_paths = [build_image_path(root, fname) for fname in all_fnames]
    missing_paths = [p for p in all_paths if not p.exists()]
    if missing_paths:
        print(f"Warning: {len(missing_paths)} files referenced in pairs are missing on disk; they will be skipped.")
    all_paths = [p for p in all_paths if p.exists()]
    if args.max_images and args.max_images > 0:
        random.shuffle(all_paths)
        all_paths = all_paths[: args.max_images]

    base_transform = transforms.Compose(
        [
            transforms.Resize((args.resize, args.resize), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
        ]
    )

    dataset = LFWImageDataset(all_paths, transform=base_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # MCUCoder model + codec
    model = MCUCoder().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    codec = create_codec(model, loader, device)

    # ArcFace
    arcface_model = load_arcface(args.arcface_model, device)
    to_pil = transforms.ToPILImage()

    used_filters = list(range(1, 13))
    summary = {}

    # Helper: cache embeddings per filter
    for uf in used_filters:
        model.p = uf / 12.0
        model.replace_value = 0

        embeddings = {}
        bpp_values = []

        print(f"Evaluating used_filter={uf} on {len(all_paths)} unique images...")
        with torch.no_grad():
            for images, paths in tqdm(loader, desc=f"uf={uf}", leave=False):
                images = images.to(device)

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
                for pil_img, path in zip(recon_pils, paths):
                    embeddings[path] = arcface_embed(arcface_model, pil_img)

        # Build similarities
        def pair_scores(pair_list):
            sims = []
            missing = 0
            for fname1, fname2 in pair_list:
                p1 = str(build_image_path(root, fname1))
                p2 = str(build_image_path(root, fname2))
                if p1 not in embeddings or p2 not in embeddings:
                    missing += 1
                    continue
                sims.append(cosine_similarity(embeddings[p1], embeddings[p2]))
            if missing:
                print(f"Warning: {missing} pairs missing embeddings (likely filtered by --max_images).")
            return np.array(sims, dtype=np.float64)

        pos_train_sims = pair_scores(pos_train)
        neg_train_sims = pair_scores(neg_train)
        pos_test_sims = pair_scores(pos_test)
        neg_test_sims = pair_scores(neg_test)

        if len(pos_train_sims) == 0 or len(neg_train_sims) == 0:
            raise ValueError("Insufficient pairs to compute threshold.")

        threshold, train_acc = best_threshold(pos_train_sims, neg_train_sims)
        test_acc = eval_accuracy(pos_test_sims, neg_test_sims, threshold) if len(pos_test_sims) and len(neg_test_sims) else float("nan")
        mean_bpp = float(np.mean(bpp_values)) if bpp_values else float("nan")

        summary[uf] = {"threshold": threshold, "train_acc": train_acc, "test_acc": test_acc, "bpp": mean_bpp}
        print(
            f"used_filter={uf}: threshold={threshold:.4f} | train_acc={train_acc:.4f} | "
            f"test_acc={test_acc:.4f} | bpp≈{mean_bpp:.4f}"
        )

    if summary:
        print("\n==== Summary over all filters ====")
        for uf in used_filters:
            vals = summary[uf]
            print(
                f"used_filter={uf}: threshold={vals['threshold']:.4f} | "
                f"train_acc={vals['train_acc']:.4f} | test_acc={vals['test_acc']:.4f} | bpp≈{vals['bpp']:.4f}"
            )


if __name__ == "__main__":
    main()
