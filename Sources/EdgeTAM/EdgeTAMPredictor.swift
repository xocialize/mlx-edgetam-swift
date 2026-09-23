import Foundation
import CoreGraphics
import MLX

/// End-to-end EdgeTAM image predictor: set a CGImage once, then prompt it any number of times → mask at
/// source resolution. Mirrors SAM2ImagePredictor (verified stage-by-stage through `setImage` against the
/// upstream PyTorch predictor, AB-T-0171):
///   • preprocess — torchvision `Resize((1024,1024))` on the tensor (antialiased; a non-square image is
///     STRETCHED, not letterboxed) + ImageNet normalisation;
///   • `setImage` runs the encoder ONCE and caches image_embed + the two high-res skips (`_features`);
///   • prompts — coords scaled `x/W·1024, y/H·1024`; a box is merged into the point list as corner points
///     labelled 2/3 (see `EdgeTAMModel.embedPrompt`);
///   • masks — 256² low-res logits → bilinear (align_corners=False) to source res; `logit > 0` = mask.
public final class EdgeTAMPredictor: @unchecked Sendable {

    /// Which decoder output to return.
    public enum MaskMode: Sendable {
        /// SAM2's recommendation: multimask for an ambiguous prompt (a single point, no box), single-mask
        /// (with the dynamic-stability fallback) once the prompt is unambiguous (2+ points, or a box).
        case auto
        /// Always the best-IoU of the three multimask outputs (`multimask_output=True`, v0.4.x behaviour).
        case multimask
        /// Always the single-mask output (`multimask_output=False` + `_dynamic_multimask_via_stability`).
        case single
    }

    /// One prompt's result. `mask`, `score`, `fullLogits`, `scores` keep the v0.4 tuple's names.
    public struct Prediction {
        /// `(H,W)` float {0,1}: the chosen mask, `logit > 0` at source resolution.
        public let mask: MLXArray
        /// Predicted IoU of the chosen mask.
        public let score: Float
        /// `(H,W)` float in [0,1]: the chosen mask ANTI-ALIASED — `clip(0.5 + L / 2|∇L|)` over the source-res
        /// logits L, i.e. coverage from the signed distance (px) to the logit-0 contour: a ~1–2 px ramp at the
        /// edge, exact 0/1 elsewhere, and its 0.5 level IS `mask`'s edge. (A plain `sigmoid(L)` is NOT a
        /// matte: SAM logits are uncalibrated and, upsampled 4× from 256², change only ~0.5/px at the edge
        /// → a ~20 px blur plus background haze — measured on the fox, AB-T-0171.)
        public let soft: MLXArray
        /// `(3,H,W)` source-res logits of the three MULTIMASK outputs (tokens 1…3), for diagnostics.
        public let fullLogits: MLXArray
        /// Predicted IoUs of the three multimask outputs.
        public let scores: [Float]
        /// Decoder token the mask came from: 0 = single-mask, 1…3 = multimask.
        public let token: Int
        /// `(4,256,256)` raw low-res logits of all four tokens + their IoUs `(4,)` — the parity surface.
        public let lowRes: MLXArray
        public let lowResIoU: MLXArray
    }

    public let model: EdgeTAMModel
    private var features: EdgeTAMModel.ImageFeatures?
    /// The last `setImage`'s preprocessed encoder input `(1,1024,1024,3)` NHWC (kept for parity dumps).
    public private(set) var input: MLXArray?
    public private(set) var origH = 0, origW = 0
    private let mean = MLXArray([0.485, 0.456, 0.406] as [Float], [1, 1, 1, 3])
    private let std = MLXArray([0.229, 0.224, 0.225] as [Float], [1, 1, 1, 3])

    public init(weights: [String: MLXArray]) { self.model = EdgeTAMModel(weights: weights) }

    public static func fromPretrained(_ path: String, dtype: DType = .float32) throws -> EdgeTAMPredictor {
        let w = try MLX.loadArrays(url: URL(fileURLWithPath: path)).mapValues { $0.asType(dtype) }
        return EdgeTAMPredictor(weights: w)
    }

    /// Signed-distance anti-aliasing of a logit map `(H,W)`: `clip(0.5 + L / (2·max(|∇L|, ε)), 0, 1)` with
    /// central differences (edge-replicated). `L/|∇L|` ≈ distance in px to the `L = 0` contour.
    public static func antialiasedMatte(_ logits: MLXArray) -> MLXArray {
        let L = logits.asType(.float32)
        let (H, W) = (L.dim(0), L.dim(1))
        let px = MLX.concatenated([L[0..., 0 ..< 1], L, L[0..., (W - 1) ..< W]], axis: 1)
        let py = MLX.concatenated([L[0 ..< 1, 0...], L, L[(H - 1) ..< H, 0...]], axis: 0)
        let gx = (px[0..., 2...] - px[0..., ..<W]) * 0.5
        let gy = (py[2..., 0...] - py[..<H, 0...]) * 0.5
        let g = MLX.maximum(MLX.sqrt(gx * gx + gy * gy), 1e-3)
        return MLX.clip(0.5 + L / (2 * g), min: 0, max: 1)
    }

    /// Cached encoder features of the current image (nil before `setImage`).
    public var imageFeatures: EdgeTAMModel.ImageFeatures? { features }

    /// Preprocess + encode once. Every later `predict` on this image is decoder-only.
    public func setImage(_ cg: CGImage) {
        origH = cg.height; origW = cg.width
        let rgb = EdgeTAMImage.rgb(from: cg, width: origW, height: origH)        // (1,H,W,3) native
        let resized = EdgeTAMImage.resize(rgb, outH: 1024, outW: 1024, antialias: true)
        var x = (resized - mean) / std                         // fp32 input: EdgeTAM's fp16 weights promote to fp32
        // SAM 2.1 (Hiera) computes in the WEIGHT dtype: fp16 activations cut the peak 2.18 → ~1.1 GB at 1024² with
        // no accuracy cost (fox clicks: mask IoU vs PyTorch fp32 0.9996–0.9999 either way; AB-T-0173).
        if model.hiera != nil, let d = model.weightDType, d != .float32 { x = x.asType(d) }
        let f = model.features(x)
        f.eval()
        input = x; features = f
    }

    /// Single positive click → mask (convenience).
    public func predict(point: (Float, Float), label: Int = 1, mode: MaskMode = .auto) -> Prediction {
        predict(points: [[point.0, point.1]], labels: [label], mode: mode)
    }

    /// Point and/or box prompt → `Prediction`. `points` are `[x,y]` in source px with `labels` 1 = foreground /
    /// 0 = background (one per point); `box` is `[x0,y0,x1,y1]` in source px.
    public func predict(points: [[Float]], labels: [Int], box: [Float]? = nil, mode: MaskMode = .auto) -> Prediction {
        guard let features else { preconditionFailure("EdgeTAMPredictor.predict before setImage") }
        precondition(!points.isEmpty || box != nil, "EdgeTAMPredictor.predict needs a point or a box")
        let sx = 1024 / Float(origW), sy = 1024 / Float(origH)
        var scaled = [Float]()
        for p in points { scaled.append(p[0] * sx); scaled.append(p[1] * sy) }
        let coords = MLXArray(scaled, [points.count, 2])
        let boxPx = box.map { [$0[0] * sx, $0[1] * sy, $0[2] * sx, $0[3] * sy] }
        let (masks, iou) = model.decode(features, coordsPx: coords, labels: labels, box: boxPx)  // (4,256,256),(4,)
        let lowRes = masks.asType(.float32), lowIoU = iou.asType(.float32)
        MLX.eval(lowRes, lowIoU)
        let ious = lowIoU.asArray(Float.self)

        let multimask: Bool
        switch mode {
        case .multimask: multimask = true
        case .single: multimask = false
        case .auto: multimask = box == nil && points.count <= 1
        }
        let token = multimask ? (1 ..< 4).max(by: { ious[$0] < ious[$1] })!
                              : model.dynamicMultimaskIndex(lowRes, lowIoU)

        // postprocess: 256 → source res (bilinear, align_corners=False), tokens as channels
        let m4 = lowRes.transposed(1, 2, 0).reshaped([1, 256, 256, 4])
        let full = EdgeTAMImage.resize(m4, outH: origH, outW: origW, antialias: false)[0]   // (H,W,4)
        let chosen = full[0..., 0..., token]
        let mask = (chosen .> 0).asType(.float32)
        let soft = Self.antialiasedMatte(chosen)
        let fullLogits = full[0..., 0..., 1 ..< 4].transposed(2, 0, 1)
        MLX.eval(mask, soft)
        return Prediction(mask: mask, score: ious[token], soft: soft, fullLogits: fullLogits,
                          scores: Array(ious[1 ..< 4]), token: token, lowRes: lowRes, lowResIoU: lowIoU)
    }
}
