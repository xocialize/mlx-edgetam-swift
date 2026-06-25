# EdgeTAM `promptSegment` ModelPackage — C0–C13 conformance review

Reviewed 2026-06-25 against `EngineeringDocs/MLXEngineDocs/conformance.md` (contract **1.10.0**, which
introduces `.promptSegment`). Evidence cites `Sources/MLXEdgeTAM/{EdgeTAMPackage,EdgeTAMConfiguration}.swift`.
Parity (mlx-porting domain): image_embed 9.7e-6 · mask logits 8.9e-5 · postproc bilinear 8.7e-5 vs the
PyTorch oracle; end-to-end click→mask IoU **0.99** vs SAM2ImagePredictor (`edgetam-smoke --image`); package
run live-verified (`edgetam-package-smoke` → correct cab-window matte, peak 0.42 GB).

**Verdict: PASS (C0–C13).** Offline contract build green; package envelope (license gate → load → run →
Matte) live-verified on a real image.

| C | Item | Verdict | Evidence |
|---|---|---|---|
| **C0** | Contract version | ✅ | `manifest.contractVersion` = `ContractVersion.current` (1.10.0 — introduced `.promptSegment`). |
| **C1** | Capability registration | ✅ | One capability `.promptSegment`; one `surfaces` entry (`PromptSegmentContract.descriptor`), one model. |
| **C2** | Canonical schema | ✅ | `PromptSegmentRequest` (image, points, pointLabels, box, mode) → `PromptSegmentResponse(matte, score)`; contract's own types. |
| **C3** | Canonical artifact I/O | ✅ | Input `Image` (PNG/JPEG bytes) → output `Matte` (PNG bytes, kind `.binary`), serialized round-trip. |
| **C4** | Mode-as-parameter | ✅ | No modes declared (single image-mode surface); prompts are canonical request fields, not steering tags. |
| **C5** | metaData hygiene | ✅ | `metaData` unused; prompts are canonical fields. |
| **C6** | Specialty declaration | ✅ | `specialties: []` — none claimed. |
| **C7** | Weight license gate | ✅ | `weightLicense: .apache2` (facebookresearch/EdgeTAM Apache-2.0) — permissive; verified `.admitted` in `edgetam-package-smoke`. |
| **C8** | Port-code license gate | ✅ | `portCodeLicense: .mit` (from-scratch Swift port; RepViT/SAM2 arch Apache, port code MIT) — permissive. |
| **C9** | PackageConfiguration | ✅ | `EdgeTAMConfiguration: PackageConfiguration, ModelStorable`; session-stable (repo/weightsFile/quant/store root). Per-request prompts ride the request. |
| **C10** | Requirements manifest | ✅ | `footprints: [.fp16 1.0 GB]` (measured peak 0.42 GB @1800×1200 + margin), `requiredBackends: [.metalGPU]`, `minMacOS 26`. |
| **C11** | MCPBridge introspection | ✅ | `PromptSegmentContract.descriptor` (image + points + pointLabels + box params) — introspectable. |
| **C12** | Forward-compat discipline | ✅ | No closed-`Capability` switch; `run()` guards `request.capability == .promptSegment` else-throws. |
| **C13** | Runtime governance | ✅ | Engine-constructed via `PackageRegistration.of` + `nonisolated init` (no compute in init); `@InferenceActor` load/run/unload; `unload()` nils the predictor; cancellation checked around the forward. |

## Notes
- **fp16 validated, not assumed** (the LaMa lesson): fp16 weight-rounding shifts mask logits ~0.1 but the
  thresholded mask is unchanged — e2e IoU 0.99 vs PyTorch. Ship fp16 (18 MB).
- **Image-mode only.** Video masklet tracking (perceiver + RoPE-2D memory + memory bank) is a future
  package surface; the prompt encoder + mask decoder are built SAM-generic for a future standalone
  `segment_anything` capability.
- **Box prompts** are a follow-up (V1 = point prompts; the contract already carries `box`).
