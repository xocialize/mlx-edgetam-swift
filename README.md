# mlx-edgetam-swift

SAM2-family promptable segmentation on Apple-Silicon MLX-Swift. These are from-scratch ports of Meta's official
code and checkpoints, packaged as MLXEngine ModelPackages:
- **EdgeTAM** ([facebookresearch/EdgeTAM](https://github.com/facebookresearch/EdgeTAM)). This is the fast
  default: `promptSegment` (image) + `trackObject` (video), 13.9 M params.
- **SAM 2.1-Small** ([facebookresearch/sam2](https://github.com/facebookresearch/sam2)), image only (v0.6.0).
  This is the quality tier for click and box selection: `promptSegment`, 38.5 M params. It shares EdgeTAM's
  prompt encoder, mask decoder and predictor. Only the backbone differs (Hiera instead of RepViT), and it is
  picked from the weights.

> **Status (EdgeTAM, v0.5.0+):** parity-locked **end to end through `setImage`** against the upstream PyTorch
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

## SAM 2.1-Small: the quality tier (v0.6.0, AB-T-0173)
- **Weights:** [`mlx-community/SAM2.1-hiera-small-fp16`](https://huggingface.co/mlx-community/SAM2.1-hiera-small-fp16),
  the image path of Meta's `sam2.1_hiera_small.pt` (`oracle/convert_sam21.py`).
- **Loading:** `EdgeTAMPredictor.fromPretrained` detects Hiera from the weights. `HieraSpec` also covers the
  tiny, base+ and large configs. Base+ is verified on CPU fp32 on the fox: image_embed rel 1.0e-4, masks
  IoU 1.0000. On the GPU its image_embed drifts to rel 0.34 over its 24 blocks (TF32), yet the masks stay at
  IoU ≥ 0.9993. Tiny and large are untested.
- **Engine package:** `MLXEdgeTAM.SAM21Package` (`promptSegment`, `softMatte` mode, per-image feature cache,
  `WeightSourcing`).

**Parity (CPU fp32, through `setImage`, `oracle/compare_stages.py`)** on the fox, the captioned fox and the
1200×800 gate page:
- image_embed relative error ≤ 5.3e-6;
- all 21 click and box masks match at IoU 1.0000;
- predicted-IoU Δ ≤ 1e-5.

The committed GPU test `SAM21ParityTests` covers gate, truck and the captioned fox. It uses wider tolerances
because MLX's GPU fp32 matmul on M5 is TF32-like (measured rel 8e-4 per GEMM). Over 16 Hiera blocks that
compounds to image_embed rel 1.4e-2, while masks still match at IoU ≥ 0.991.

**Runtime and memory (M5 Max, Release, fp16):** the Hiera path computes in the weight dtype.
- Memory: MLX peak 0.98 / 1.13 GB (1024² / 1800×1200); `phys_footprint` 2.35 / 2.79 GB.
- Time: ~0.1–0.25 s per new image; later prompts on the same image take 11–22 ms.

| on the captioned fox / gate page | EdgeTAM | SAM 2.1-S |
|---|---|---|
| belly click (IoU vs whole fox) | 0.008 | **0.968** |
| best of 3 masks, every click | 0.41–0.84 | **0.94–0.97** |
| click on square B…E | also selects the top-left square A | **only the clicked square** |
| box around the fox | 0.904 | 0.922 |

Use EdgeTAM for video and speed, and SAM 2.1-S when a single click has to land first time.

## What to expect from one EdgeTAM click (measured upstream behaviour, not port error)
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
# SAM 2.1-S: official checkpoint → MLX, then the same stage comparison (the fork carries configs/sam2.1/*)
curl -LO https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
oracle/.venv/bin/python oracle/convert_sam21.py --ckpt sam2.1_hiera_small.pt --out oracle/weights/sam21_hiera_small_fp32.safetensors
oracle/.venv/bin/python oracle/run_click_oracle.py --image X.png --clicks 471,635 --out oracle/goldens/click \
    --config configs/sam2.1/sam2.1_hiera_s.yaml --ckpt $PWD/sam2.1_hiera_small.pt --tag .sam21s
edgetam-smoke --weights oracle/weights/sam21_hiera_small_fp32.safetensors --image X.png --clicks "471,635" --stages-out oracle/goldens/click/X.sam21s.swift.safetensors
oracle/.venv/bin/python oracle/compare_stages.py oracle/goldens/click/X.sam21s
```

## Layout
- `Sources/EdgeTAM` — the MLX core: the RepViT-M1 or Hiera (`Hiera.swift`) backbone, FPN, SAM prompt encoder,
  mask decoder, image and video predictors, and the EdgeTAM video memory stack.
- `Sources/MLXEdgeTAM` — the conformant ModelPackages: `EdgeTAMPackage` and `SAM21Package`, which share
  `PromptSegmentSession` and `StoreWeights`.
- `Sources/EdgeTAM*Smoke` — the parity gates and package-drive gates.
- `Tests/` — the non-square e2e parity test, the MAT and CAN conformance tests, and store resolution.

## License
The port code is MIT. EdgeTAM, SAM 2 / SAM 2.1, Hiera and RepViT are Apache-2.0. See NOTICE.
