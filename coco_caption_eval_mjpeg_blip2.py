# Evaluate MJPEG (JPEG per frame) on COCO captions with BLIP-2.
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import random
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from tqdm import tqdm

HF_HUB_AVAILABLE = False
try:
    from huggingface_hub import snapshot_download

    HF_HUB_AVAILABLE = True
except ImportError:
    snapshot_download = None

try:
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
except ImportError as exc:
    raise ImportError("transformers is required for BLIP-2 evaluation.") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="COCO caption evaluation of MJPEG (JPEG) reconstructions with BLIP-2.")
    parser.add_argument("--images_dir", type=str, required=True, help="COCO images directory (e.g., val2017).")
    parser.add_argument(
        "--ann_file",
        type=str,
        required=True,
        help="COCO caption annotation file (e.g., annotations/captions_val2017.json).",
    )
    parser.add_argument(
        "--quality_levels",
        type=str,
        default="5,10,20,30,40,50,60,70,80,90",
        help="Comma-separated JPEG quality levels to evaluate (1-100).",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for BLIP-2 (memory heavy).")
    parser.add_argument("--resize", type=int, default=224, help="Resize for JPEG input.")
    parser.add_argument(
        "--dataset_fraction",
        type=float,
        default=1.,
        help="Fraction of the dataset to evaluate (0<f<=1). Default 0.1 = 10%.",
    )
    parser.add_argument("--max_images", type=int, default=0, help="Limit number of images (0 = use all).")
    parser.add_argument("--caption_prompt", type=str, default="A photo of", help="Text prompt fed to BLIP-2.")
    parser.add_argument("--num_beams", type=int, default=3, help="Number of beams for BLIP-2 generation.")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Max generated tokens for BLIP-2 captions.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--output_dir", type=str, default="results_mjpeg", help="Directory to save caption outputs.")
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="Optional HF cache dir; triggers an explicit snapshot download of the BLIP-2 weights.",
    )
    parser.add_argument(
        "--blip2_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        help="BLIP-2 model id.",
    )
    parser.add_argument(
        "--blip2_device",
        type=str,
        default=None,
        help="Optional device override for BLIP-2 (e.g., cuda:0). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--blip2_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Computation dtype for BLIP-2.",
    )
    return parser.parse_args()


class CocoCaptionDataset(CocoCaptions):
    def __getitem__(self, index):
        img, captions = super().__getitem__(index)
        img_id = self.ids[index]
        return img, captions, img_id


def collate_fn(batch):
    images, targets, ids = zip(*batch)
    return list(images), list(targets), list(ids)


def parse_quality_levels(spec: str):
    values = []
    for token in spec.split(","):
        tok = token.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            raise ValueError(f"Invalid quality level: {tok}")
        if v < 1 or v > 100:
            raise ValueError(f"Quality must be in [1,100], got {v}")
        values.append(v)
    if not values:
        values = [50]
    return sorted(set(values))


def resolve_dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def token_f1(a: str, b: str) -> float:
    a_tokens = a.lower().strip().split()
    b_tokens = b.lower().strip().split()
    if not a_tokens or not b_tokens:
        return 0.0
    a_counts = Counter(a_tokens)
    b_counts = Counter(b_tokens)
    common = sum((a_counts & b_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(b_tokens)
    recall = common / len(a_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ensure_blip2_weights(model_name: str, cache_dir: str | None):
    if not HF_HUB_AVAILABLE:
        print("huggingface_hub not installed; relying on transformers cache.")
        return None
    try:
        path = snapshot_download(repo_id=model_name, cache_dir=cache_dir)
        print(f"BLIP-2 weights available in {path}")
        return path
    except Exception as exc:
        print(f"Warning: could not snapshot_download {model_name}: {exc}")
        return None


def build_blip2(model_name: str, device: torch.device, dtype: torch.dtype, cache_dir: str | None):
    processor = Blip2Processor.from_pretrained(model_name, cache_dir=cache_dir)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype, cache_dir=cache_dir)
    model = model.to(device)
    model.eval()
    return processor, model


def resolve_images_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"images_dir does not exist: {path}")

    def has_images(p: Path) -> bool:
        return next(p.glob("*.jpg"), None) is not None or next(p.glob("*.jpeg"), None) is not None

    if path.is_dir() and has_images(path):
        return path

    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and has_images(subdirs[0]):
        print(f"No images directly under {path}; using nested dir {subdirs[0]}")
        return subdirs[0]

    for sub in subdirs:
        if sub.name == path.name and has_images(sub):
            print(f"No images directly under {path}; using nested dir {sub}")
            return sub

    raise ValueError(f"No image files found in {path}")


def evaluate_coco_captions(coco_gt: COCO, results, image_ids=None):
    try:
        from pycocoevalcap.eval import COCOEvalCap
    except ImportError:
        print("pycocoevalcap not installed; skipping caption metrics.")
        return None

    if len(results) == 0:
        return None

    coco_res = coco_gt.loadRes(results)
    coco_eval = COCOEvalCap(coco_gt, coco_res)
    if image_ids:
        coco_eval.params["image_id"] = image_ids
    coco_eval.evaluate()
    return coco_eval.eval


def jpeg_reencode(image: torch.Tensor, quality: int, to_pil) -> tuple[torch.Tensor, float]:
    pil = to_pil(image.cpu().clamp(0, 1))
    buffer = BytesIO()
    pil.save(buffer, format="JPEG", quality=quality, optimize=True)
    size_bytes = buffer.tell()
    buffer.seek(0)
    recon = transforms.ToTensor()(Image.open(buffer).convert("RGB"))
    bpp = (size_bytes * 8) / (image.shape[1] * image.shape[2])
    return recon, bpp


def main():
    args = parse_args()
    torch.manual_seed(0)
    random.seed(10)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vlm_device = torch.device(args.blip2_device) if args.blip2_device else device
    vlm_dtype = resolve_dtype(args.blip2_dtype)
    if vlm_device.type == "cpu":
        vlm_dtype = torch.float32

    images_root = resolve_images_dir(args.images_dir)

    base_transform = transforms.Compose(
        [
            transforms.Resize((args.resize, args.resize), antialias=True),
            transforms.ToTensor(),
        ]
    )

    dataset = CocoCaptionDataset(root=str(images_root), annFile=args.ann_file, transform=base_transform)
    if len(dataset) == 0:
        raise ValueError(f"No images found in {args.images_dir}")

    selected_ids = list(dataset.ids)
    if args.dataset_fraction and 0 < args.dataset_fraction < 1.0:
        sample_size = max(1, int(len(selected_ids) * args.dataset_fraction))
        selected_ids = random.sample(selected_ids, sample_size)
        print(f"Using {sample_size}/{len(dataset)} images ({args.dataset_fraction*100:.1f}%).")
    if args.max_images and args.max_images > 0 and args.max_images < len(selected_ids):
        random.shuffle(selected_ids)
        selected_ids = selected_ids[: args.max_images]
        print(f"Further limited to {len(selected_ids)} images via max_images.")
    dataset.ids = selected_ids
    coco_gt = dataset.coco
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    quality_levels = parse_quality_levels(args.quality_levels)
    ensure_blip2_weights(args.blip2_model, args.hf_cache_dir)
    processor, blip2 = build_blip2(args.blip2_model, vlm_device, vlm_dtype, args.hf_cache_dir)
    to_pil = transforms.ToPILImage()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for q in quality_levels:
        bpp_values = []
        agreement_scores = []
        orig_results = []
        recon_results = []

        print(f"Evaluating JPEG quality={q} on {len(dataset)} images...")
        with torch.no_grad():
            for images, _, img_ids in tqdm(loader, desc=f"Q={q}", leave=False):
                images_tensor = torch.stack(images).to(device, non_blocking=True)

                recon_list = []
                for img in images_tensor:
                    recon, bpp = jpeg_reencode(img, q, to_pil)
                    recon_list.append(recon)
                    bpp_values.append(bpp)
                recon = torch.stack(recon_list).to(device)

                pil_orig = [to_pil(img.cpu()) for img in images_tensor]
                pil_recon = [to_pil(img.cpu()) for img in recon]
                prompts = [args.caption_prompt] * len(pil_orig)

                orig_inputs = processor(images=pil_orig, text=prompts, return_tensors="pt", padding=True)
                orig_inputs = {k: v.to(vlm_device) if isinstance(v, torch.Tensor) else v for k, v in orig_inputs.items()}
                orig_ids = blip2.generate(**orig_inputs, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
                orig_caps = processor.batch_decode(orig_ids, skip_special_tokens=True)

                recon_inputs = processor(images=pil_recon, text=prompts, return_tensors="pt", padding=True)
                recon_inputs = {
                    k: v.to(vlm_device) if isinstance(v, torch.Tensor) else v for k, v in recon_inputs.items()
                }
                recon_ids = blip2.generate(**recon_inputs, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
                recon_caps = processor.batch_decode(recon_ids, skip_special_tokens=True)

                for img_id, orig_cap, recon_cap in zip(img_ids, orig_caps, recon_caps):
                    iid = int(img_id)
                    orig_cap = orig_cap.strip()
                    recon_cap = recon_cap.strip()
                    orig_results.append({"image_id": iid, "caption": orig_cap})
                    recon_results.append({"image_id": iid, "caption": recon_cap})
                    agreement_scores.append(token_f1(orig_cap, recon_cap))

        image_ids_used = [r["image_id"] for r in recon_results]
        orig_metrics = evaluate_coco_captions(coco_gt, orig_results, image_ids_used)
        recon_metrics = evaluate_coco_captions(coco_gt, recon_results, image_ids_used)

        q_summary = {
            "bpp": float(np.mean(bpp_values)) if bpp_values else float("nan"),
            "caption_agreement_f1": float(np.mean(agreement_scores)) if agreement_scores else float("nan"),
            "orig_metrics": orig_metrics,
            "recon_metrics": recon_metrics,
        }
        summary[q] = q_summary

        with open(output_dir / f"blip2_orig_q{q}.json", "w") as f:
            json.dump(orig_results, f)
        with open(output_dir / f"blip2_recon_q{q}.json", "w") as f:
            json.dump(recon_results, f)

        print(
            f"Q={q} | bpp≈{q_summary['bpp']:.4f} | "
            f"caption_agreement_f1={q_summary['caption_agreement_f1']:.3f}"
        )
        if recon_metrics:
            print("  recon metrics: " + ", ".join([f"{k}={v:.3f}" for k, v in recon_metrics.items()]))
        if orig_metrics:
            print("  orig  metrics: " + ", ".join([f"{k}={v:.3f}" for k, v in orig_metrics.items()]))

    print("\n==== Summary over all qualities ====")
    for q in quality_levels:
        if q not in summary:
            continue
        entry = summary[q]
        print(
            f"Q={q}: bpp≈{entry['bpp']:.4f}, caption_agreement_f1={entry['caption_agreement_f1']:.3f}"
        )
        for tag, metrics in [("orig", entry["orig_metrics"]), ("recon", entry["recon_metrics"])]:
            if metrics:
                metric_str = ", ".join([f"{k}={v:.3f}" for k, v in metrics.items()])
                print(f"  {tag}: {metric_str}")


if __name__ == "__main__":
    main()
