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

[EdgeTAM](https://github.com/facebookresearch/EdgeTAM) — on-device SAM 2 for promptable segmentation +
video tracking — converted to **Apple MLX** (`-fp16`) for the
[`mlx-edgetam-swift`](https://github.com/xocialize/mlx-edgetam-swift) Swift package (MLXEngine
`promptSegment` + `trackObject` ModelPackage). 22× faster than SAM 2, 16 FPS on iPhone 15 Pro Max.

From-scratch MLX-Swift architecture port. **Image-mode** (point/box → mask): RepViT-M1 encoder + FPN +
SAM prompt encoder + two-way mask decoder — parity-locked vs the PyTorch oracle on the CPU stream
(image_embed 9.7e-6, mask logits 8.9e-5; end-to-end mask **IoU 0.99** vs PyTorch). **Video-mode**
(`trackObject`, click on one frame → per-frame masklet): adds the video memory stack — PerceiverResampler
+ MemoryEncoder + MemoryAttention (RoPE-2D) + the SAM2 memory-bank state machine — every op parity-locked
vs the oracle; full masklet propagation min-IoU 0.92. This single `-fp16` file carries both (874 tensors).

## Use

```swift
// Package.swift → .package(url: "https://github.com/xocialize/mlx-edgetam-swift", from: "0.1.0")
import EdgeTAM
// Image: click → object mask
let p = try EdgeTAMPredictor.fromPretrained(weightsPath, dtype: .float16)
p.setImage(sourceCGImage)
let (mask, score, _, _) = p.predict(point: (500, 375))
// Video: click on a frame → per-frame masklet
let vp = try EdgeTAMVideoPredictor.fromPretrained(weightsPath, dtype: .float16)
let track = vp.track(frames: cgImages, clickFrame: 0, points: [[210, 350]], labels: [1])
```

Or as an MLXEngine ModelPackage (`MLXEdgeTAM.EdgeTAMPackage`) — `promptSegment` (image) + `trackObject`
(video) surfaces — resolving this repo via the Hub.

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
