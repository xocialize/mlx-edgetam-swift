# mlx-edgetam-swift

EdgeTAM (on-device SAM 2) promptable segmentation on Apple-Silicon MLX-Swift — a from-scratch
architecture port for an MLXEngine `promptSegment` ModelPackage. **Image-mode** (point/box → mask);
the shared click/box-select lane for Extract cutouts and Erase masks.

> **Status:** image-mode **parity-locked** (image_embed 9.7e-6, masks 8.9e-5; e2e mask IoU 0.99 vs
> PyTorch) + conformant `MLXEdgeTAM` ModelPackage. Video masklet tracking is a future phase.
> See [SCOPING.md](SCOPING.md).

## Layout
- `Sources/EdgeTAM` — MLX core: RepViT-M1 encoder + FPN + SAM prompt encoder + mask decoder + predictor.
- `Sources/MLXEdgeTAM` — conformant `promptSegment` ModelPackage (image + point/box → `Matte` + score).
- `Sources/Smoke`, `Sources/PackageSmoke` — parity + package-drive gates.
- `oracle/` — PyTorch parity harness + weight converter/publisher.

## Use
```swift
import EdgeTAM
let predictor = try EdgeTAMPredictor.fromPretrained(weightsPath, dtype: .float16)
predictor.setImage(sourceCGImage)
let (mask, score, _, _) = predictor.predict(point: (500, 375))   // click → object mask (H×W bool)
```
Weights: [`mlx-community/EdgeTAM-fp16`](https://huggingface.co/mlx-community/EdgeTAM-fp16) (18 MB).

## License
Port code MIT. EdgeTAM / SAM 2 / RepViT Apache-2.0. See NOTICE.
