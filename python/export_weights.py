import argparse
import json
from pathlib import Path

import numpy as np
import torch


def absmax_quantize(x):
    x = x.detach().cpu().float()
    max_abs = torch.max(torch.abs(x)).item()

    if max_abs == 0:
        scale = 1.0
        q = torch.zeros_like(x, dtype=torch.int8)
    else:
        scale = max_abs / 127.0
        q = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)

    zero_point = 0
    return q, scale, zero_point


def export_tensor(name, tensor, output_dir):
    values = tensor.detach().cpu()

    if "weight" in name:
        quantized_tensor, scale, zero_point = absmax_quantize(values)
        arr = quantized_tensor.numpy().astype(np.int8)
        dtype = "int8"
    else:
        arr = values.numpy().astype(np.float32)
        scale = 1.0
        zero_point = 0
        dtype = "float32"

    bin_path = output_dir / f"{name}.bin"
    json_path = output_dir / f"{name}.json"

    arr.tofile(bin_path)

    metadata = {
        "shape": list(arr.shape),
        "dtype": dtype,
        "scale": float(scale),
        "zero_point": int(zero_point),
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Wrote {bin_path}")
    print(f"Wrote {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="python/lenet5_mnist.pth",
        help="Path to the trained LeNet-5 .pth file",
    )
    parser.add_argument(
        "--out-dir",
        default="weights",
        help="Directory where .bin and .json files will be written",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.out_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Some checkpoints are saved as {"model_state_dict": ...}
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    for name, tensor in state_dict.items():
        export_tensor(name, tensor, output_dir)


if __name__ == "__main__":
    main()