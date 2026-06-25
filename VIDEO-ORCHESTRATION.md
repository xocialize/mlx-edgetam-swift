# EdgeTAM video memory-bank orchestration — port spec

Condensed from a full read of `sam2_base.py` + `sam2_video_predictor.py` (EdgeTAM config). The
**stateful** half of P2 (the ops — perceiver/encoder/attention — are de-risked separately). Shapes
use C=256 (hidden), mem_dim=64, H=W=64 (feat), image 1024, num_maskmem=7.

## Per-frame track_step
1. **Encode frame** → vision feats `(HW=4096, B, 256)` seq-first + pos.
2. **`_prepare_memory_conditioned_features`**:
   - **First/init-cond frame** (`directly_add_no_mem_embed=true`): `pix_feat = feats[-1] + no_mem_embed`
     → `(B,256,64,64)`. **No memory attention.** (Same path as image-mode.)
   - **Else**: build memory bank (below) → `pix_feat = memory_attention(curr=feats, memory, curr_pos,
     memory_pos, num_obj_ptr_tokens, num_spatial_mem)` → `(B,256,64,64)`.
3. **`_forward_sam_heads`** (memory-conditioned pix_feat as image_embed; prompts only on click frames;
   `multimask_output_for_tracking` → 3 masks, pick best by IoU) → masks + obj_ptr + object_score_logits.
4. **`_encode_new_memory`** (if run_mem_encoder): `mask_for_mem = sigmoid(hi_res_mask)*20 − 10`;
   `memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)` → `(B,64,64,64)`; then
   **`spatial_perceiver`** → `(B,512,64)` compressed memory + pos. Store in output_dict.

## Memory-bank assembly (the part to port carefully)
`to_cat_memory` (seq-first, dim 64), in order:
1. **Spatial memory** — for `t_pos` in slots: conditioning frames `t_pos=0`; then prev frames
   `t_pos∈[1..6]`, `t_rel=7−t_pos`, frame = `frame_idx−1` (t_rel=1) else `((frame_idx−2)//stride)*stride
   −(t_rel−2)*stride` (stride=1 eval). Each stored memory `(B,512,64)` → `(512,B,64)`; **pos += `maskmem_tpos_enc[7−t_pos−1]`** (i.e. index `6−t_pos`, shape `(1,1,64)`).
2. **Object pointers** — gather past obj_ptrs (cond + up to 15 non-cond past), `only_obj_ptrs_in_the_past`.
   Each `(B,256)`; since `mem_dim(64) < C(256)`: reshape `(num_ptr,B,4,64)`→`(num_ptr*4,B,64)` (split into
   4 tokens); pos = **zeros** (`add_tpos_enc_to_obj_ptrs=false`). Append after spatial.
`num_spatial_mem` = #spatial slots; `num_obj_ptr_tokens` = #ptr tokens. `memory = cat(...)`,
`memory_pos = cat(...)`, both `(total, B, 64)`.

## memory_attention (2 layers, RoPE-2D, heads 1) — de-risk separately
- `output = curr + 0.1*curr_pos` (pos_enc_at_input). Per layer: self-attn (RoPEAttention v1, no pos-add,
  rope on q&k full 64×64) → cross-attn (RoPEAttentionv2: q rope 64×64; **keys: first 256 = 1D latents NO
  rope, next 256 = 2D latents rope 16×16 with repeat=num_spatial_mem, last `num_obj_ptr_tokens` excluded**)
  with `k = memory + memory_pos` (pos_enc_at_cross_attn_keys) → MLP. Final norm.
- Validated memory I/O golden: curr (4096,1,256), memory (516,1,64)=512 spatial(1 frame)+4 obj-ptr → (4096,1,256).

## obj_ptr extraction (in _forward_sam_heads)
`use_multimask_token_for_obj_ptr` → from the 3 multimask tokens; `obj_ptr_proj` (MLP); `pred_obj_scores` +
`fixed_no_obj_ptr`: `obj_ptr = is_obj*obj_ptr + (1−is_obj)*no_obj_ptr`, `is_obj = object_score_logits>0`.

## Op de-risk status (MLX-Python) — **WHOLE VIDEO FORWARD DE-RISKED 2026-06-25**
- ✅ PerceiverResampler 2.2e-5 (`mlx_perceiver.py`)
- ✅ MemoryEncoder 2.9e-6 (`mlx_mem_encoder.py`)
- ✅ MemoryAttention (RoPE-2D) 3.5e-6 (`mlx_mem_attn.py`)
- ✅ Tracking decoder (`mlx_track_decode.py`) — propagated-frame path: empty prompt (2 not-a-point tokens),
  multimask=True (EdgeTAM `multimask_min_pt_num=0` + `multimask_output_for_tracking=True`), best-by-IoU,
  object-score hard-gate (`pred_obj_score_head` 3-layer MLP off token 0), obj_ptr from best multimask token
  (`obj_ptr_proj` 3-MLP) + `fixed_no_obj_ptr` gate. **mask 1.9e-5 / obj_ptr 1.5e-6 / obj_score 2.9e-6** vs f1.
- ✅ Memory-bank assembly verified against captured `vid_ma_memory/pos`: spatial[:512] = perceiver out
  (**stored bf16** by the predictor's eval offload — diff 0.0 vs bf16-rounded); obj-ptr = 4×64 split tokens;
  obj-ptr pos = **zeros exactly** (`add_tpos_enc_to_obj_ptrs=False`); spatial pos = perceiver pos + tpos.
  **PORT NOTE:** on-device MLX needs no CPU offload → keep fp16/fp32 (more precise than the reference bf16).
- ✅ Linkage: `memory_attention out` reshaped == decoder `backbone_features` **bit-identical (0.0)**.
- ✅ **SWIFT PORT DONE 2026-06-25** (`Sources/EdgeTAM/EdgeTAMVideo.swift` + `EdgeTAMVideoPredictor.swift`).
  All 3 ops + RoPE-2D + the memory-bank state machine + tracking-decode + propagate loop transcribed
  from the validated oracles. **Position encodings generated in Swift** (PositionEmbeddingSine — content-
  independent), validated vs the `vid_*_pos` goldens to ~5e-7. `edgetam-video-smoke` (CPU fp32) parity:
  perceiver 2.2e-5 · mem-encoder 2.9e-6 · mem-attention 3.1e-6 · track mask 1.9e-5 / obj_ptr 1.2e-6 /
  obj_score 2.9e-6 (all = the Python-oracle precision) · mem-pos assembly 5e-7 · **full 5-frame masklet
  propagation min-IoU 0.92** (frame-0 IoU 1.0; f1–4 0.92–0.96 — the `vid_mask_f` goldens were generated
  with the predictor's **bf16** memory offload, Swift keeps fp32 = more precise, so the masklet legitimately
  diverges a few boundary px on the ~0.4%-coverage object).
- **Gotcha found in Swift e2e:** the click is in original-video px → must normalize by `(video_W,video_H)`
  then ×1024 before the prompt encoder (`add_new_points_or_box` does this; missing it → 8× mask).
- **Surface:** `EdgeTAMVideoPredictor` (`[CGImage]+click → [mask]`, pure CoreGraphics+MLX). Video-FILE
  decode (URL→frames) stays in the **Forge shell** via `FrameStreamNative` (FFmpeg-free, RIFE/SeedVR2
  path) — NOT in the SPM. Engine `ModelPackage` capability deferred pending a video-tracking contract
  in MLXToolKit (`promptSegment` is image-only; new capability = product decision).
