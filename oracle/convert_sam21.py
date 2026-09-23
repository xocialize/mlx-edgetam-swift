"""SAM 2.1 (Hiera) image-mode weights → MLX safetensors for the Swift SAM2-family core (AB-T-0173).

Source: Meta's official checkpoints (https://dl.fbaipublicfiles.com/segment_anything_2/092824/<name>.pt).
Keeps the IMAGE path only — image_encoder (Hiera trunk + FpnNeck), sam_prompt_encoder, sam_mask_decoder,
no_mem_embed — because the Swift port serves `promptSegment` (SAM 2.1's video memory stack differs from
EdgeTAM's and is not ported). Key names are upstream's, unchanged; the decoder/prompt-encoder keys are
identical to EdgeTAM's, so the Swift decoder is shared.

Layout: conv (O,I,kH,kW) → NHWC (O,kH,kW,I); ConvTranspose (output_upscaling.0/.3) (I,O,kH,kW) → (O,kH,kW,I);
the trunk's learned positional embeddings `pos_embed` (1,C,7|14,…) and `pos_embed_window` (1,C,8,8) are
params, not convs → (1,H,W,C) NHWC (the Swift side builds the 256² embedding from them).

    oracle/.venv/bin/python oracle/convert_sam21.py --ckpt /Volumes/Satechi/Models/sam2.1-oracle/sam2.1_hiera_small.pt \
        --out oracle/weights/sam21_hiera_small_fp32.safetensors [--dtype float16]
"""
import argparse
import os

import mlx.core as mx
import numpy as np
import torch

KEEP = ("image_encoder", "sam_prompt_encoder", "sam_mask_decoder", "no_mem_embed")
CONV_T = {"sam_mask_decoder.output_upscaling.0.weight", "sam_mask_decoder.output_upscaling.3.weight"}
POS = {"image_encoder.trunk.pos_embed", "image_encoder.trunk.pos_embed_window"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"]
    dt = mx.float16 if a.dtype == "float16" else mx.float32
    out = {}
    for k, v in sd.items():
        if not k.startswith(KEEP) or k.endswith("num_batches_tracked"):
            continue
        x = v.float().numpy()
        if k in POS:
            x = np.transpose(x, (0, 2, 3, 1))                        # (1,C,H,W) → (1,H,W,C)
        elif x.ndim == 4:
            x = np.transpose(x, (1, 2, 3, 0)) if k in CONV_T else np.transpose(x, (0, 2, 3, 1))
        out[k] = mx.array(x.astype(np.float32)).astype(dt)
    mx.eval(list(out.values()))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    mx.save_safetensors(a.out, out, metadata={"format": "mlx", "source": os.path.basename(a.ckpt),
                                             "model": "sam2.1-image"})
    n = sum(int(np.prod(v.shape)) for v in out.values())
    print(f"[convert_sam21] {len(out)} tensors, {n / 1e6:.1f} M params ({a.dtype}) → {a.out}")


if __name__ == "__main__":
    main()
