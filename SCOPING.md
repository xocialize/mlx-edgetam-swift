# EdgeTAM — MLX-Swift Port Scoping (feasibility recon)

**Model:** facebookresearch/EdgeTAM — on-device SAM 2 variant for promptable segmentation + video
tracking. **22× faster than SAM 2, 16 FPS on iPhone 15 Pro Max** (no quant). **Apache-2.0** ✅.
**Checkpoint:** `checkpoints/edgetam.pt` — **54 MB total** (whole model). Ships a **CoreML export**
(`coreml/export_to_coreml.py`) → Apple-HW viability proven + a cross-check reference.

**Why we want it:** the shared **promptable-segmentation** layer both Extract (Stage 2: "click the
object") and Erase (Stage 3: click-to-erase + video masklet tracking) have been waiting on. One port
unblocks two capabilities. (Per `BACKGROUND-REMOVAL-PLAN.md` / `ERASE-PLAN.md`.)

## Architecture = SAM 2 codebase + EdgeTAM efficiency swaps

It literally vendors the `sam2/` package. Config `configs/edgetam.yaml`. Components + LOC:

| Component | LOC | Role | Port stance |
|---|---|---|---|
| **RepViT-M1 image encoder** (`backbones/timm.py` → `repvit_m1.dist_in1k`) | timm | trunk, 4 levels ch [48,96,192,384] | **NET-NEW** — reparam mobile-ViT (conv+SE+GELU, no exotic ops). The main image-encoder work. |
| **FpnNeck** (`backbones/image_encoder.py`) | 134 | FPN + sine pos-enc, d_model 256 | standard (small) |
| **PromptEncoder** (`sam/prompt_encoder.py`) | 182 | points/boxes/masks → embeddings | **stock SAM** — donate from mlx-examples SAM / `mlx-community/sam3-image` |
| **MaskDecoder + two-way transformer** (`sam/mask_decoder.py`, `sam/transformer.py`) | 295+435 | 3 multimask outputs + IoU | **stock SAM** — donate (near-identical) |
| **Memory attention** (`memory_attention.py`, RoPE-2D) | 182 | video temporal memory | **NET-NEW** (video) |
| **Memory encoder** (`memory_encoder.py`) | 181 | encode mask+feat → memory | **NET-NEW** (video) |
| **2D spatial Perceiver** (`perceiver.py`) | 319 | EdgeTAM novelty: compress memory; standard latent cross-attn (Linear q/kv, LayerNorm, softmax) — NOT exotic | **NET-NEW** (video) |
| **SAM2Base** (`sam2_base.py`) | 927 | orchestration: image predict + video state/memory-bank | glue (image subset is small; video state is the big part) |

## The decisive phasing — image-mode is a tractable SAM-image port

- **IMAGE promptable segmentation** ("click/box → mask", what Extract/Erase Stage 2 need first) uses
  ONLY: RepViT encoder + FpnNeck + PromptEncoder + MaskDecoder + a thin image-predictor. The
  perceiver / memory-attention / memory-encoder / video-state are **not on this path**. → medium
  effort; net-new = RepViT (standard conv) + FPN; the SAM prompt/decoder are donatable.
- **VIDEO masklet tracking** (Stage 3/4) adds the memory machinery (RoPE-2D attention, memory encoder,
  perceiver) + the stateful memory-bank orchestration in `sam2_base` (~the 927 LOC). This is the
  **challenging, novel, stateful** part — the real reason this is a "hard one."

## Reuse map (skill: reuse, don't port)

- **Donate** prompt-encoder + mask-decoder + two-way transformer from an existing MLX SAM port
  (`mlx-examples` segment-anything / `mlx-community/sam3-image`) — they're stock SAM, byte-shaped the
  same. *(To confirm in Phase 0: which donor is closest to SAM2's decoder + has clean MLX-Swift code.)*
- **Net-new**: RepViT-M1 trunk (no SAM port has it; standard conv — check for an existing MLX RepViT
  first), FpnNeck, and the whole video memory stack.
- **Cross-check oracle**: the shipped CoreML export + the PyTorch reference (both Apache, in-repo).

## Silent-failure surfaces to watch (Step-1 traps)
- RepViT **structural reparameterization** (RepVGG-style: train-time multi-branch fused to single conv
  at inference). timm's `repvit` may load already-fused or need `reparameterize()` — confirm the
  checkpoint's conv shapes are the *deployed/fused* form before porting.
- **Prompt/coordinate conventions**: point/box normalization, the +0.5 pixel-center offset, the image
  preprocessing (1024² resize, ImageNet norm) — classic SAM parity traps.
- **RoPE-2D** in memory attention (`feat_sizes [32,32]`, asymmetric q/k sizes 64/16) — 2D axial RoPE,
  a known parity trap (per `references/spatial-and-rope-ops.md`).
- **Mask-decoder output**: multimask vs single, the IoU head, low-res→full-res upscaling + threshold.

## Proposed phased plan
- **Phase 0 — de-risk + donor pick (cheap):** stand up the PyTorch oracle (load edgetam.pt, run the
  image predictor on a click → golden mask); identify the closest MLX SAM donor for prompt/decoder;
  confirm RepViT checkpoint is fused; find/decide MLX RepViT. Gate: clean image-mask oracle + donor chosen.
- **Phase 1 — image-mode port (PRIMARY):** RepViT-M1 + FpnNeck + donated PromptEncoder + MaskDecoder in
  MLX-Swift; parity-lock the mask logits vs the PyTorch oracle (CPU fp32 <1e-3); `imageSegment`-style
  surface (image + point/box → mask). Unblocks Extract/Erase **promptable Stage 2**.
- **Phase 2 — video tracking:** memory attention (RoPE-2D) + memory encoder + perceiver + the
  memory-bank state machine; masklet propagation across frames. The hard, stateful part.
- **Phase 3 — engine + publish:** ModelPackage (likely the existing `matting`/a new promptable surface),
  footprint, C0–C13; weights → mlx-community (54 MB → tiny), code → xocialize.

## Phase 0 de-risk — DONE 2026-06-24 (GO)

| Gate | Result |
|---|---|
| **PyTorch oracle (image mode)** | `oracle/run_oracle.py` — `build_sam2("configs/edgetam.yaml", edgetam.pt)` + `SAM2ImagePredictor`; click (500,375) on truck.jpg → **correct mask** (segmented the cab window the click hit, best score 0.77). Goldens dumped: `image_embed (1,256,64,64)`, `high_res_feats [(1,32,256,256),(1,64,128,128)]`, `masks_logits (3,H,W)`, scores. ✅ |
| **RepViT-M1 fusion** | Checkpoint backbone is **deployed/inference form** — `Conv2d_BN` (`.c`+`.bn`, foldable), **no train-time multi-branch** (`rbr`/`branch`/`identity`) keys → no `reparameterize()` step. Standard conv+BN port. ✅ |
| **Component key counts** | image_encoder 684 · sam_prompt_encoder **17** · sam_mask_decoder **131** · spatial_perceiver 44 · memory_attention 54 · memory_encoder 40 (982 total) — stock-SAM sizes confirm donatability. ✅ |
| **SAM donor** | `mlx-examples/segment_anything` has `prompt_encoder.py` + `mask_decoder.py` + `transformer.py` (MLX-Python, SAM1 — near-identical two-way transformer + IoU head + prompt encoder to SAM2's; SAM2 adds the high-res-feat path + obj-ptr). Clean **translation reference** for the Swift port. `mlx-community/sam3-image` is a fallback. ✅ |

**Verdict: GO.** Image-mode is tractable and de-risked: oracle produces correct masks, RepViT is a
standard conv+BN port, the SAM prompt/decoder math is donatable from mlx-examples. Net-new risk =
RepViT-M1 trunk (standard) + FpnNeck (small) + image-predictor glue (1024² resize / ImageNet norm /
coord conventions — the classic SAM parity traps to pin). Video stack (perceiver + memory) stays Phase 2.

## Phase 1 progress

- **P1a — RepViT-M1 + FpnNeck encoder: PARITY-LOCKED 2026-06-24 (`oracle/mlx_encoder.py`, image_embed
  max_abs 9.7e-6 vs golden, CPU fp32).** Trunk bit-perfect first try (all 4 stages ~2e-5) — RepViT
  legacy RepVggDw (`conv3x3dw + conv1x1dw + identity`, no trailing BN), SE by key-presence, FPN top-down
  sum-fuse at levels [2,3]. **Gotcha:** `image_embed = FPN out[2] + no_mem_embed` — SAM2 adds a learned
  "no memory" embedding to image features in image mode (7e-2 error until added). The main net-new
  risk (the efficient encoder) is retired.
- **P1b — prompt encoder + mask decoder: PARITY-LOCKED 2026-06-24 (`oracle/mlx_decoder.py`, masks
  max_abs 8.2e-5 + iou 1.25e-6 vs RAW decoder output).** SAM PE-random + prompt tokens + 2-layer
  TwoWayTransformer (8 heads, 256→128 cross-attn downsample) + ConvT upscaling fused w/ conv_s0/s1 +
  hypernetwork MLPs + IoU head. **Four SAM gotchas found:** (1) `skip_first_layer_pe` layer-0 self-attn
  REPLACES queries (no residual); (2) `iou_prediction_head` has SIGMOID output; (3) `LayerNorm2d` eps
  **1e-6**; (4) `predict()` post-processes low_res → parity vs the RAW decoder masks, not predict's return.
  **→ entire image-mode forward validated in MLX-Python.**
- **P1c — Swift core: PARITY-LOCKED 2026-06-24 (`Sources/EdgeTAM/EdgeTAMModel.swift`, `edgetam-smoke`:
  image_embed 9.7e-6 · masks 8.9e-5 · iou 1.3e-6).** Full RepViT+FPN+prompt+decoder transcribed from the
  validated references, bit-faithful first build. `oracle/convert.py` → NHWC safetensors (727 tensors).
- **P1c — image predictor: DONE 2026-06-24 (`EdgeTAMPredictor`/`EdgeTAMImage`).** CGImage→1024² bilinear
  + ImageNet-norm preprocess, coord scale `coord/orig*1024`, raw-256-masks → torch-matching bilinear
  (align_corners=False) postprocess + threshold, multimask best-by-IoU. `edgetam-smoke --image`:
  **postproc bilinear 8.7e-5** (exact vs torch), **e2e IoU-vs-PyTorch 0.9902** on truck click → correct
  cab-window mask. **→ PHASE 1 (image-mode promptable segmentation) COMPLETE in Swift.**
- **P3 — image-mode ModelPackage + publish: DONE 2026-06-25** (promptSegment 1.10.0, MLXEdgeTAM, mlx-community/EdgeTAM-fp16, xocialize v0.1.0). See CONFORMANCE.md.

## P2 — video memory stack (in progress)

**Config (edgetam.yaml):** num_maskmem=7 (1 cond + 6 prev), use_obj_ptrs_in_encoder, max_obj_ptrs 16,
only_obj_ptrs_in_the_past, directly_add_no_mem_embed, sigmoid_scale 20 / bias −10 for mem-enc, mem_dim 64.

**Per-frame flow (after frame 0):** encode frame (RepViT+FPN, image-mode) → **memory-attend** current feats
(queries) to the memory bank (keys = past frames' compressed memory + obj-ptrs) via MemoryAttention (2 layers,
RoPE-2D, num_heads 1) → mask decoder → **encode new memory**: MaskDownSampler(k3 s2) + Fuser(2× CXBlock dw-k7)
→ **PerceiverResampler compresses** to 512 latents → store with temporal pos-enc; mask-token → obj-ptr.

**Components / de-risk status:**
- **PerceiverResampler — DE-RISKED 2026-06-25 (`oracle/mlx_perceiver.py`, 2.2e-5 vs golden).** 256 1D
  global latents (cross-attend all 4096 positions + pos added to k&v) + 256 2D windowed latents (16×16
  windows of the 64×64, 1 latent/window, shared layers) → concat 512. heads 1, dim 64, FF=LN+Lin+GELU+Lin.
- **MemoryEncoder — DE-RISKED 2026-06-25 (`oracle/mlx_mem_encoder.py`, 2.9e-6).** MaskDownSampler (4×
  conv-s2 + LN2d + GELU → 1×1) + pix_feat_proj + Fuser (2× CXBlock dw-k7) + out_proj→64. (caller does
  the scaled-sigmoid; encoder gets skip_mask_sigmoid=True.)
- **MemoryAttention (RoPE-2D) — DE-RISKED 2026-06-25 (`oracle/mlx_mem_attn.py`, 3.5e-6, first try).**
  2 layers (self-attn rope-q&k 64² + cross-v2: q rope 64², keys [256 1D no-rope, 256 2D rope-16², 4
  obj-ptr no-rope] = memory+pos, MLP relu). `output = curr + 0.1·curr_pos`. internal 256, heads 1.
  **→ ALL THREE NOVEL VIDEO OPS DE-RISKED** (perceiver 2.2e-5 · encoder 2.9e-6 · attention 3.5e-6).
- **Memory-bank orchestration — SPEC'D 2026-06-25 (`VIDEO-ORCHESTRATION.md`, via background explorer agent):**
  memory-slot temporal indexing `maskmem_tpos_enc[6−t_pos]`, obj-ptr split into 4 tokens (mem_dim 64<C 256),
  spatial-then-pointer concat, scaled-sigmoid memory, directly_add_no_mem_embed first frame, obj_ptr from
  multimask tokens + fixed_no_obj_ptr gating. The hard stateful algorithm — mapped, ready to port.
- **Swift port — DONE 2026-06-25 (`Sources/EdgeTAM/EdgeTAMVideo.swift` + `EdgeTAMVideoPredictor.swift`).**
  All 3 ops + RoPE-2D + memory-bank state machine + tracking-decode + propagate loop transcribed from the
  validated oracles; `convert.py` extended to 874 tensors (147 video; 9 real convs → NHWC, `maskmem_tpos_enc`
  raw). Position encodings (PositionEmbeddingSine) generated in Swift, validated vs goldens to ~5e-7.
  `edgetam-video-smoke` (CPU fp32): every op = Python-oracle precision; **full 5-frame masklet min-IoU 0.92**
  (frame-0 IoU 1.0; f1–4 diverge a few boundary px vs the bf16-offload reference — Swift keeps fp32). The
  click must be normalized `(point/[W,H])·1024` before the prompt encoder. **→ P2 video tracking validated
  in Swift.**
- **P3-video — ENGINE INTEGRATION DONE 2026-06-25.** New `trackObject` capability in MLXToolKit **contract
  1.11.0** (engine `415c4af`): `TrackObjectRequest` (Video + point/box on `promptFrame`) → `TrackObjectResponse`
  (`[Matte]` per frame + per-frame scores); `CanonicalOutput.matteSequence` (lossless per-frame, not a
  re-encoded mask video). `EdgeTAMPackage` gains the second surface (`edgetam-track`); `Video` bytes are
  decoded via `FrameStreamNative.decode` (new frames-in seam, `27e767f`) → `EdgeTAMVideoPredictor.track` →
  `[Matte]`. `edgetam-video-package-smoke` (full surface, GPU): ProRes .mov → decode → masklet → 5 mattes,
  per-frame IoU 0.92–0.98, **measured peak 1.07 GB @ 5f / 1.79 GB @ 30f** (~0.9 GB fixed + ~30 MB/frame).
  Weights republished mlx-community/EdgeTAM-fp16 (874 tensors, fp16 round-trip validated for video).

- **P3-video ENHANCEMENT(v2) — multi-object + box + streaming DONE 2026-06-25.** All three additive, NO
  contract bump (still 1.11.0).
  - **Multi-object**: `EdgeTAMModel.propagate` / `EdgeTAMVideoPredictor.track` take `[ObjectPrompt]`, share
    the per-frame RepViT+FPN encode ONCE, run a per-object memory bank + decode + memory-encode → one track
    per object. SURFACE = request-per-object (one `TrackObjectRequest` = one object; the contract's documented
    "lane-ready" interpretation), or drive `track(frames:objects:)` directly for the shared-encode win in one
    pass. `edgetam-video-smoke` 2-object case: boy+girl min-IoU **0.9186**; **object-0 (boy) under the
    shared-encode multi-object pass == the single-object track bit-for-bit (independence IoU 1.0000)**.
  - **Box prompt**: `req.box` wired `forwardSamHeads`→`embedPrompt` (SAM corner tokens, labels 2/3, +0.5, no
    not_a_point pad). **Gotcha (SAM2 `_use_multimask`):** a box contributes 2 corner points → exceeds
    `multimask_max_pt_num=1` → the box-prompted init frame emits the SINGLE (un-ambiguous) mask, not the
    best-of-3 multimask; tracked frames still multimask. Wiring it as always-multimask gave a wrong f0 (IoU
    0.70); the single-mask gate fixes it to **f0 0.886 / track min 0.886** vs the box golden.
  - **Long-clip streaming**: `propagate` pulls frames one at a time, emits each Matte as it lands (PNG-encode +
    drop the MLX mask), `MLX.GPU.clearCache()` per frame → flat GPU footprint independent of clip length OR
    object count. `edgetam-video-package-smoke` (full surface, GPU, 5f): boy **0.917** / girl **0.969** /
    box **0.888**, **peak 1.02 GB**.
  - Oracle: `run_video_oracle.py` captures `vid_mo_obj{0,1}_f*` (boy reuses single-object run; girl point
    [400,240]) + `vid_box_f*` (bbox of the boy frame-0 mask). NB: the upstream 2D-perceiver `.view` chokes on
    batch>1, so the oracle decodes each object in its own B=1 pass (SAM2 memory banks are per-object
    independent → identical to batched). Remaining: multi-object surface-batching field (deferred to honor
    "no contract bump"); a future `.rawRGBA16Half` 16-bit step is unrelated.

## First read: feasible, well-phased, image-mode is the affordable win
54 MB, Apache, CoreML-proven on Apple HW, and the hard part (video memory) is cleanly separable from a
tractable image-mode port that immediately unblocks promptable selection for two capabilities. The
SAM stock parts are donatable. Net-new risk concentrates in RepViT (standard) and the video memory stack.
