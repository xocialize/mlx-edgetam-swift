"""Publish SAM 2.1-small fp16 image-mode weights to mlx-community/SAM2.1-hiera-small-fp16 (AB-T-0173), and refresh
the EdgeTAM-fp16 model card with the v0.5.0 parity numbers (card only; its weights are unchanged).

    oracle/.venv/bin/python oracle/convert_sam21.py --ckpt /Volumes/Satechi/Models/sam2.1-oracle/sam2.1_hiera_small.pt \
        --out oracle/weights/sam21_hiera_small_fp16.safetensors --dtype float16
    oracle/.venv/bin/python oracle/publish_sam21.py [--edgetam-card-only]
"""
import argparse
import os

from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
RID = "mlx-community/SAM2.1-hiera-small-fp16"

CARD = """---
library_name: mlx
license: apache-2.0
license_link: https://github.com/facebookresearch/sam2/blob/main/LICENSE
base_model: facebook/sam2.1-hiera-small
pipeline_tag: image-segmentation
tags:
  - mlx
  - segmentation
  - promptable-segmentation
  - sam2
  - sam2.1
---

# mlx-community/SAM2.1-hiera-small-fp16

[SAM 2.1](https://github.com/facebookresearch/sam2) Hiera-Small, **image mode** (point / box → mask),
converted to Apple **MLX** (fp16) from Meta's official `sam2.1_hiera_small.pt` for the
[`mlx-edgetam-swift`](https://github.com/xocialize/mlx-edgetam-swift) Swift package (`SAM21Package`, an MLXEngine
`promptSegment` ModelPackage).

- **Contents:** the image path only — Hiera trunk + FPN neck + SAM prompt encoder + mask decoder + `no_mem_embed`
  (359 tensors, 38.5 M params). The video memory stack is not included.
- **Keys:** upstream's names, unchanged. Convolutions are NHWC `(O,kH,kW,I)`. The trunk's `pos_embed` and
  `pos_embed_window` are `(1,H,W,C)`. Converter: `oracle/convert_sam21.py`.
- **Parity:** the Swift port matches PyTorch `SAM2ImagePredictor` **through `set_image`** on the CPU fp32 stream.
  On a 1024² flat-shaded image, a 1024² captioned copy and a 1200×800 synthetic page:
  - image_embed relative error ≤ 5.3e-6;
  - every click and box mask matches at IoU 1.0000, with predicted-IoU Δ ≤ 1e-5.
  With these fp16 weights and fp16 activations on the GPU, mask IoU vs fp32 PyTorch is ≥ 0.9996 on
  object-sized prompts.
- **Footprint (M5 Max):** 1024² encode ~0.1 s warm; each further prompt on the same image 11–18 ms; process
  `phys_footprint` 2.4–2.8 GB.

## Why use it next to EdgeTAM
[EdgeTAM](https://huggingface.co/mlx-community/EdgeTAM-fp16) (13.9 M params) is the fast default and handles
video. On single clicks over flat-shaded or synthetic images, SAM 2.1-S is much better:
- a click on a cartoon fox's belly selects the whole fox (IoU 0.968 vs 0.008 for EdgeTAM);
- a click on one small square of a grey page selects only that square, where EdgeTAM also adds a corner
  square every time.

## Use
```swift
// .package(url: "https://github.com/xocialize/mlx-edgetam-swift", from: "0.6.0")
import EdgeTAM
let p = try EdgeTAMPredictor.fromPretrained(weightsPath, dtype: .float16)   // Hiera detected from the weights
p.setImage(sourceCGImage)
let r = p.predict(point: (471, 635))            // r.mask, r.soft (anti-aliased), r.score
let boxed = p.predict(points: [], labels: [], box: [40, 245, 745, 910])
```
Or through MLXEngine: `SAM21Package` (`MLXEdgeTAM`), with `mode: softMatte` for an anti-aliased matte.

Weights: Apache-2.0 (facebookresearch/sam2). Port code: MIT.
"""

EDGETAM_CARD = """---
library_name: mlx
license: apache-2.0
license_link: https://github.com/facebookresearch/EdgeTAM/blob/main/LICENSE
base_model: facebookresearch/EdgeTAM
pipeline_tag: image-segmentation
tags:
  - mlx
  - segmentation
  - promptable-segmentation
  - sam2
  - edgetam
---

# mlx-community/EdgeTAM-fp16

[EdgeTAM](https://github.com/facebookresearch/EdgeTAM) is on-device SAM 2 for promptable segmentation and video
tracking. This repo holds it converted to Apple **MLX** (fp16) for the
[`mlx-edgetam-swift`](https://github.com/xocialize/mlx-edgetam-swift) Swift package (`EdgeTAMPackage`, an MLXEngine
`promptSegment` + `trackObject` ModelPackage). There are 874 tensors (image + video), checked against the official
`edgetam.pt` (fp16 rounding only).

**Parity (v0.5.0).** Measured on the CPU fp32 stream against upstream PyTorch, through the public predictors:
- **image:** image_embed relative error ≤ 2.4e-4, and every click and box mask matches at IoU 1.0000 on square
  and non-square images;
- **video:** 5-frame point and box tracks stay at IoU ≥ 0.999.

**Expect weak single clicks on flat-shaded or synthetic images.** Upstream EdgeTAM behaves the same way; a box
prompt is its reliable input. For a better click-selection tier see
[SAM2.1-hiera-small-fp16](https://huggingface.co/mlx-community/SAM2.1-hiera-small-fp16).

## Use
```swift
// .package(url: "https://github.com/xocialize/mlx-edgetam-swift", from: "0.6.0")
import EdgeTAM
let p = try EdgeTAMPredictor.fromPretrained(weightsPath, dtype: .float16)
p.setImage(sourceCGImage)                       // encoder once per image
let r = p.predict(point: (500, 375))            // r.mask, r.soft (anti-aliased), r.score
let vp = try EdgeTAMVideoPredictor.fromPretrained(weightsPath, dtype: .float16)
```

Weights: Apache-2.0 (facebookresearch/EdgeTAM). Port code: MIT.
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--edgetam-card-only", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    if not a.edgetam_card_only:
        api.create_repo(RID, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=f"{HERE}/weights/sam21_hiera_small_fp16.safetensors",
                        path_in_repo="model.safetensors", repo_id=RID)
        api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=RID)
        print(f"[publish] → https://huggingface.co/{RID}")
    api.upload_file(path_or_fileobj=EDGETAM_CARD.encode(), path_in_repo="README.md", repo_id="mlx-community/EdgeTAM-fp16",
                    commit_message="Card: v0.5.0 upstream-parity numbers; link the SAM 2.1-S quality tier")
    print("[publish] EdgeTAM-fp16 card refreshed")


if __name__ == "__main__":
    main()
