"""EdgeTAM image-mode weights → canonical MLX safetensors (NHWC) + Swift parity fixtures.

Conv (O,I,kH,kW)→(O,kH,kW,I); ConvTranspose (output_upscaling.0/.3) (I,O,kH,kW)→(O,kH,kW,I);
Linear/2D kept; drop num_batches_tracked. Flat torch keys (Swift WeightLoading maps them).
Only image-mode keys (image_encoder, sam_prompt_encoder, sam_mask_decoder, no_mem_embed).
"""
import argparse
import os
import numpy as np
import torch
import mlx.core as mx

HERE = os.path.dirname(__file__)
CKPT = "/Users/dustinnielson/Development/porting_dev_opportunities/_eval/EdgeTAM/checkpoints/edgetam.pt"
CONV_T = {"sam_mask_decoder.output_upscaling.0.weight", "sam_mask_decoder.output_upscaling.3.weight"}
KEEP = ("image_encoder", "sam_prompt_encoder", "sam_mask_decoder", "no_mem_embed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--out", default=f"{HERE}/weights/edgetam_fp32.safetensors")
    args = ap.parse_args()
    dt = mx.float16 if args.dtype == "float16" else mx.float32
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]

    out = {}
    for k, v in sd.items():
        if not k.startswith(KEEP) or k.endswith("num_batches_tracked"):
            continue
        a = v.float().numpy()
        if a.ndim == 4:
            a = np.transpose(a, (1, 2, 3, 0)) if k in CONV_T else np.transpose(a, (0, 2, 3, 1))
        out[k] = mx.array(a.astype(np.float32)).astype(dt)
    mx.eval(list(out.values()))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    mx.save_safetensors(args.out, out)
    print(f"[convert] {len(out)} tensors ({args.dtype}) -> {args.out}")

    # Swift parity fixture (NHWC inputs + goldens)
    g = f"{HERE}/goldens"
    fx = {
        "enc_input": mx.array(np.transpose(np.load(f"{g}/enc_input.npy"), (0, 2, 3, 1)).astype(np.float32)),
        "image_embed": mx.array(np.transpose(np.load(f"{g}/image_embed.npy"), (0, 2, 3, 1)).astype(np.float32)),
        "unnorm_coords": mx.array(np.load(f"{g}/unnorm_coords.npy")[0].astype(np.float32)),  # (1,2)
        "labels": mx.array(np.load(f"{g}/labels.npy")[0].astype(np.float32)),                # (1,)
        "masks_raw": mx.array(np.load(f"{g}/dec_masks_raw.npy")[0].astype(np.float32)),       # (3,256,256)
        "scores": mx.array(np.load(f"{g}/scores.npy").astype(np.float32)),                    # (3,)
    }
    mx.save_safetensors(f"{HERE}/weights/parity.safetensors", fx)
    print(f"[fixture] {[ (k,tuple(v.shape)) for k,v in fx.items() ]}")


if __name__ == "__main__":
    main()
