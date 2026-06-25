"""Publish EdgeTAM fp16 image-mode weights to mlx-community/EdgeTAM-fp16."""
import os, shutil
from huggingface_hub import HfApi

HERE = os.path.dirname(__file__)
API = HfApi()
RID = "mlx-community/EdgeTAM-fp16"

CARD = """---
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

[EdgeTAM](https://github.com/facebookresearch/EdgeTAM) — on-device SAM 2 for promptable segmentation —
converted to **Apple MLX** (`-fp16`) for the [`mlx-edgetam-swift`](https://github.com/xocialize/mlx-edgetam-swift)
Swift package (MLXEngine `promptSegment` ModelPackage). **Image-mode** (point/box → mask); 22× faster
than SAM 2, 16 FPS on iPhone 15 Pro Max.

From-scratch MLX-Swift architecture port: RepViT-M1 image encoder + FPN + SAM prompt encoder + two-way
mask decoder. Parity-locked vs the PyTorch oracle on the CPU stream (image_embed 9.7e-6, mask logits
8.9e-5). This `-fp16` build is functionally identical (end-to-end mask **IoU 0.99** vs PyTorch; the
thresholded mask absorbs fp16 weight-rounding).

## Use

```swift
// Package.swift → .package(url: "https://github.com/xocialize/mlx-edgetam-swift", from: "0.1.0")
import EdgeTAM
let predictor = try EdgeTAMPredictor.fromPretrained(weightsPath, dtype: .float16)
predictor.setImage(sourceCGImage)
let (mask, score, _, _) = predictor.predict(point: (500, 375))   // click → object mask
```

Or as an MLXEngine `promptSegment` ModelPackage (`MLXEdgeTAM.EdgeTAMPackage`), resolving this repo via the Hub.

Weights: Apache-2.0 (facebookresearch/EdgeTAM). Port code: MIT.
"""


def main():
    pub = f"{HERE}/publish_edgetam"
    os.makedirs(pub, exist_ok=True)
    shutil.copy(f"{HERE}/weights/edgetam_fp16.safetensors", f"{pub}/model.safetensors")
    API.create_repo(RID, repo_type="model", exist_ok=True)
    API.upload_file(path_or_fileobj=f"{pub}/model.safetensors", path_in_repo="model.safetensors", repo_id=RID)
    API.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=RID)
    print(f"[publish] → https://huggingface.co/{RID}")


if __name__ == "__main__":
    main()
