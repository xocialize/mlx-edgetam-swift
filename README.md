# mlx-edgetam-swift

EdgeTAM (on-device SAM 2) promptable segmentation and video tracking on Apple-Silicon MLX-Swift. It is a
from-scratch architecture port of [facebookresearch/EdgeTAM](https://github.com/facebookresearch/EdgeTAM)
(official Meta code and checkpoint), packaged as an MLXEngine `promptSegment` + `trackObject` ModelPackage.

> **Status (v0.5.0):** parity-locked **end to end through `setImage`** against the upstream PyTorch
> predictors (CPU fp32), on square, non-square (1200×800, 1800×1200), flat/synthetic and natural images:
> - the preprocessed input matches exactly (≤ 6e-7);
> - image_embed and the high-res features match to relative error ≤ 2.4e-4;
> - every click and box mask matches at IoU **1.0000**, with predicted IoUs Δ ≤ 3e-4;
> - the video track stays ≥ 0.999 IoU over 5 frames, for both point and box prompts.
>
> The gates are `Tests/EdgeTAMParityTests` (committed non-square golden), `edgetam-smoke` and
> `edgetam-video-smoke`. See AB-T-0171.

## Use
```swift
import EdgeTAM
let predictor = try EdgeTAMPredictor.fromPretrained(weightsPath, dtype: .float16)
predictor.setImage(sourceCGImage)                      // encoder runs ONCE per image
let click = predictor.predict(point: (500, 375))       // single click → best of 3 multimask outputs
let boxed = predictor.predict(points: [], labels: [], box: [425, 600, 700, 875])   // single-mask output
click.mask   // (H,W) {0,1} at source resolution
click.soft   // (H,W) anti-aliased matte (~1–2 px ramp; its 0.5 level is exactly `mask`'s edge)
click.score  // predicted IoU; click.scores / click.fullLogits = all three multimask outputs
```
Once `setImage` has run, each prompt only runs the decoder: 19 ms per repeat click on an M5 Max (Release).
`MaskMode.auto`, the default, follows SAM2: a single point returns the best-IoU multimask output. Two or
more points, or a box, return the single-mask output, with SAM2's dynamic-stability fallback.

The package (`MLXEdgeTAM`) supports:
- point prompts, box prompts, and both together;
- `mode: EdgeTAMPackage.softMatte`, which returns a `.softAlpha` matte in place of `.binary`;
- a repeat request on the same image bytes, which reuses the cached features.

It declares `WeightSourcing`, so an engine at contract 1.24 or later materializes the weights into the
canonical store directory `models--mlx-community--EdgeTAM-fp16/` before load. That is the directory the
install marker lives in.

Weights: [`mlx-community/EdgeTAM-fp16`](https://huggingface.co/mlx-community/EdgeTAM-fp16). They were
checked against upstream `edgetam.pt`: 874/874 tensors match, and the maximum relative difference is fp16
rounding.

## What to expect from one click (measured upstream behaviour, not port error)
EdgeTAM is a 13.9 M-parameter model distilled for phones. **Single clicks on flat-shaded or synthetic images
are weak, and upstream PyTorch shows exactly the same numbers:**

- **Flat anime (fox, 1024²):** a click's best predicted IoU is usually 0.3–0.55, and the highest-scoring
  mask is often a part (a sliver). Even so, one of the other two multimask outputs usually overlaps the
  whole fox far better: 0.41–0.84 vs 0.005–0.84 for the chosen one (IoU vs a BiRefNet subject mask).
- **Grey page with small squares (1200×800):** each click selects its square (score ~0.83). **The top-left
  square is also in every mask**, and other squares sometimes join. This comes from EdgeTAM itself.
  SAM 2.1-small on the same page returns only the clicked square.

Cheap levers, measured on the captioned fox (IoU vs the whole-fox mask):

| prompt | EdgeTAM | SAM 2.1-S (reference) |
|---|---|---|
| 1 click on the belly (471,635) | 0.008 | 0.968 |
| 1 click on the body (300,700) | 0.843 | 0.964 |
| 3 clicks | 0.886 | 0.958 |
| **box around the fox** | **0.904** (score 0.939) | 0.922 |

For EdgeTAM, **a box is the reliable prompt**. Extra clicks help. Feeding the previous logits back as
`mask_input` adds only +0.015, and it is not ported. Reproduce with `oracle/eval_levers.py`.

## Parity tooling (`oracle/`)
`oracle/upstream/`, `weights/`, `goldens/` and `.venv/` are gitignored.
```bash
git clone https://github.com/facebookresearch/EdgeTAM oracle/upstream/EdgeTAM   # ships checkpoints/edgetam.pt
uv venv --python 3.11 oracle/.venv && VIRTUAL_ENV=oracle/.venv uv pip install torch torchvision numpy hydra-core iopath pillow mlx safetensors
oracle/.venv/bin/python oracle/run_oracle.py && oracle/.venv/bin/python oracle/run_video_oracle.py && oracle/.venv/bin/python oracle/convert.py
oracle/.venv/bin/python oracle/make_parity_golden.py        # Tests/EdgeTAMParityTests golden (committed)
# any image, stage by stage (input → image_embed → high-res feats → low-res logits → masks):
oracle/.venv/bin/python oracle/run_click_oracle.py --image X.png --clicks 471,635 --boxes 40,245,745,910 --out oracle/goldens/click
edgetam-smoke --weights oracle/weights/edgetam_fp32.safetensors --image X.png --clicks "471,635" --boxes "40,245,745,910" --stages-out oracle/goldens/click/X.swift.safetensors
oracle/.venv/bin/python oracle/compare_stages.py oracle/goldens/click/X
```

## Layout
- `Sources/EdgeTAM` — the MLX core: RepViT-M1 encoder, FPN, SAM prompt encoder, mask decoder, image and
  video predictors, and the video memory stack.
- `Sources/MLXEdgeTAM` — the conformant ModelPackage.
- `Sources/EdgeTAM*Smoke` — the parity gates and package-drive gates.
- `Tests/` — the non-square e2e parity test, the MAT and CAN conformance tests, and store resolution.

## License
The port code is MIT. EdgeTAM, SAM 2 and RepViT are Apache-2.0. See NOTICE.
