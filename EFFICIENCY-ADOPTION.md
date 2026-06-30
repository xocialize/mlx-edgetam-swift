# Efficiency Adoption Brief — `mlx-edgetam-swift` (EdgeTAM, `promptSegment` + `trackObject`)

> **For a session-specific agent.** Adopt engine 1.14 efficiency (engine 0.17.0+). Load the
> `mlx-swift-integration` skill; read references/package-efficiency.md (four levers + **"Measurement
> findings"**, esp. *in-app phys vs smoke MLX-peak*) + references/memory-harness.md. This is a LIGHT
> **split + unload-clearCache** adoption (single encoder+decoder, no encoder-evict). Audited 2026-06-30.

## Package at a glance
- Two products: `EdgeTAM` (core, mlx-swift only) + `MLXEdgeTAM` (`EdgeTAMPackage: ModelPackage`). Capabilities
  **`promptSegment`** (image) + **`trackObject`** (video, streaming). Engine pinned `from: "0.11.0"`; also
  depends on `frame-stream-native` 0.2.0 (video decode).
- **Footprint today (FLAT residentBytes only, NO transient):** `QuantFootprint(.fp16, 1.5 GB)`. Single quant.
- Architecture: image **encoder** + mask **decoder** (shared SAM2-style). The encoder runs **per-frame**
  (it's the core op, NOT evictable mid-track) → no encoder-evict lever; the win is the split + unload.
- **Already good — don't regress:** the video path (`EdgeTAMVideo.swift:411`) already does a per-frame
  `MLX.GPU.clearCache()` → flat ~1.0 GB + ~0.37 MB/frame streaming. Keep that.
- `unload()` is `predictor = nil; videoPredictor = nil` — **no `MLX.Memory.clearCache()`**.

## Audit vs. the four levers
| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡 | from 0.11.0 → 0.17.0 | **P0** |
| 1. Split footprint | ❌ | flat 1.5 GB, no transient declared | **P1 (headline)** |
| 2. Per-stage evict | ➖ N/A | single encoder+decoder; encoder is the per-frame core (can't evict mid-track). Per-frame clearCache already present. | note N/A |
| 3. mmap/lazy | 🟡 verify | confirm lazy weight load (floor ≈ on-disk ~1 GB) | note |
| 4. BudgetAware | ➖ | single fp16, no dtype lever | defer |

## Plan
- **P0:** `swift package update` → 0.17.0; build + fix any drift (both capability surfaces stable; verify).
- **P1 (headline):** split the flat 1.5 GB. `residentBytes` = the encoder+decoder weights floor (≈ on-disk,
  ~1.0–1.5 GB); `peakActivationBytes` = **one frame's encode+decode transient** (the working set the
  per-frame `clearCache` already bounds — measure that single-frame peak). Adopt `QuantConfigured` (single
  fp16). The split lets EdgeTAM co-reside cheaply (1 GB resident) under the shared transient reserve.
- **P2:** N/A — note the encoder is the per-frame core, not an evictable upfront stage, and the per-frame
  `clearCache` discipline already keeps the streaming footprint flat (don't regress it).
- **`unload()` must add `MLX.Memory.clearCache()`** after niling `predictor`/`videoPredictor`. The wrapper
  uses MLX already (the core does), so the product is linked.

## Measurement — IMPORTANT (in-app phys lesson)
EdgeTAM is light, so a smoke is more tractable than the heavy video models, but the in-app `phys_footprint`
(R-MEM-1/admission basis) still reads higher than a smoke MLX-peak (~2.5–2.9×). Declare `residentBytes` from
the measured weight floor (solid) + a **FLAGGED** best-effort `peakActivationBytes` from the single-frame
smoke peak, pending an in-app phys re-baseline (promptSegment is image-mode → measurable in `MLXEngineImage`;
trackObject via the video app). Note the per-frame transient is one-frame-bounded (the streaming is flat).

## Definition of done
- [ ] engine 0.17.0; `QuantConfigured`; P1 split declared; `unload()` clearCache; per-frame clearCache kept.
- [ ] residentBytes = measured weight floor; peakActivationBytes = single-frame transient (FLAGGED smoke est).
- [ ] Smoke green for promptSegment (valid mask) — IoU sanity; split recorded; activation flagged.
- [ ] Registry: edgetam row Eff ⬜→✅ (note "activation = smoke est, phys re-baseline pending"), Eng→0.17.0.

## Report back
flat→split, the single-frame transient (flagged for phys), drift since 0.11.0, effort, commit SHA. STAY IN
SCOPE — four-lever adoption + this brief + registry row only; no testing-app/shell/xcodeproj changes;
stop-and-report if bigger.
