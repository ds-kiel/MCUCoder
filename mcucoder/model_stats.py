"""Param and FLOP counter for the MCUCoder encoder/decoder.

Uses a single dummy forward pass on CPU and counts Conv/Linear multiply-adds
(counted as two FLOPs)."""

import argparse
import importlib.util
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


def _format_big(num: int) -> str:
    value = float(num)
    for unit in ("", "K", "M", "B", "T"):
        if abs(value) < 1000:
            text = f"{value:.2f}".rstrip("0").rstrip(".")
            return f"{text}{unit}"
        value /= 1000.0
    return f"{value:.2f}P".rstrip("0").rstrip(".") + "P"


def _conv2d_flops(module: nn.Conv2d, output: torch.Tensor) -> int:
    out_h, out_w = output.shape[2], output.shape[3]
    batch = output.shape[0]
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (
        module.in_channels / module.groups
    )
    ops_per_element = kernel_ops * 2
    if module.bias is not None:
        ops_per_element += 1
    return int(batch * module.out_channels * out_h * out_w * ops_per_element)


def _conv_transpose2d_flops(module: nn.ConvTranspose2d, output: torch.Tensor) -> int:
    out_h, out_w = output.shape[2], output.shape[3]
    batch = output.shape[0]
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (
        module.in_channels / module.groups
    )
    ops_per_element = kernel_ops * 2
    if module.bias is not None:
        ops_per_element += 1
    return int(batch * module.out_channels * out_h * out_w * ops_per_element)


def _linear_flops(module: nn.Linear, output: torch.Tensor) -> int:
    batch = output.shape[0]
    ops_per_element = module.in_features * 2
    if module.bias is not None:
        ops_per_element += 1
    return int(batch * module.out_features * ops_per_element)


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _count_flops(model: nn.Module, example_input: torch.Tensor) -> Tuple[int, torch.Tensor]:
    flops = 0
    handles = []

    def hook(module: nn.Module, inputs, output):
        nonlocal flops
        if isinstance(module, nn.Conv2d):
            flops += _conv2d_flops(module, output)
        elif isinstance(module, nn.ConvTranspose2d):
            flops += _conv_transpose2d_flops(module, output)
        elif isinstance(module, nn.Linear):
            flops += _linear_flops(module, output)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        out = model(example_input)

    for handle in handles:
        handle.remove()

    return flops, out


def _load_model_module():
    module_path = Path(__file__).resolve().parent / "model.py"
    spec = importlib.util.spec_from_file_location("mcucoder_model", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Compute params and FLOPs for MCUCoder encoder/decoder.")
    parser.add_argument("--height", type=int, default=224, help="Input height.")
    parser.add_argument("--width", type=int, default=224, help="Input width.")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size.")
    parser.add_argument("--latent-channels", type=int, default=196, help="N used to build Decoder.")
    args = parser.parse_args()

    model_module = _load_model_module()
    Encoder = model_module.Encoder
    Decoder = model_module.Decoder

    device = torch.device("cpu")
    encoder = Encoder().to(device).eval()
    decoder = Decoder(N=args.latent_channels).to(device).eval()

    dummy = torch.randn(args.batch_size, 3, args.height, args.width, device=device)

    enc_flops, encoded = _count_flops(encoder, dummy)
    dec_flops, _ = _count_flops(decoder, encoded)

    enc_params = _count_parameters(encoder)
    dec_params = _count_parameters(decoder)

    print(f"Input shape: {tuple(dummy.shape)}")
    print(f"Latent shape: {tuple(encoded.shape)}")
    print(f"Encoder params: {enc_params:,} ({_format_big(enc_params)})")
    print(f"Encoder FLOPs: {enc_flops:,} ({_format_big(enc_flops)} | {enc_flops / 1e9:.3f} GFLOPs)")
    print(f"Decoder params: {dec_params:,} ({_format_big(dec_params)})")
    print(f"Decoder FLOPs: {dec_flops:,} ({_format_big(dec_flops)} | {dec_flops / 1e9:.3f} GFLOPs) for input latent above")


if __name__ == "__main__":
    main()
