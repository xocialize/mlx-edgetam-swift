import Foundation
import CoreGraphics
import MLX

/// End-to-end EdgeTAM image predictor: set a CGImage, then click a point → mask at source resolution.
/// Mirrors SAM2ImagePredictor: 1024² bilinear + ImageNet-norm preprocess, coord scaling, raw 256 masks →
/// bilinear postprocess to source res (align_corners=False), multimask best by IoU.
public final class EdgeTAMPredictor: @unchecked Sendable {

    private let model: EdgeTAMModel
    private var input: MLXArray?
    private var origH = 0, origW = 0
    private let mean = MLXArray([0.485, 0.456, 0.406] as [Float], [1, 1, 1, 3])
    private let std = MLXArray([0.229, 0.224, 0.225] as [Float], [1, 1, 1, 3])

    public init(weights: [String: MLXArray]) { self.model = EdgeTAMModel(weights: weights) }

    public static func fromPretrained(_ path: String, dtype: DType = .float32) throws -> EdgeTAMPredictor {
        let w = try MLX.loadArrays(url: URL(fileURLWithPath: path)).mapValues { $0.asType(dtype) }
        return EdgeTAMPredictor(weights: w)
    }

    public func setImage(_ cg: CGImage) {
        origH = cg.height; origW = cg.width
        let rgb = EdgeTAMImage.rgb(from: cg, width: origW, height: origH)        // (1,H,W,3) native
        let resized = EdgeTAMImage.bilinear(rgb, outH: 1024, outW: 1024)
        input = (resized - mean) / std
    }

    /// Single positive click → mask (convenience).
    public func predict(point: (Float, Float), label: Int = 1)
        -> (mask: MLXArray, score: Float, fullLogits: MLXArray, scores: [Float]) {
        predict(points: [[point.0, point.1]], labels: [label])
    }

    /// Multi-point prompt → (binary mask `(H,W)`, score, full-res logits `(3,H,W)`, all scores).
    /// `points` are `[x,y]` in source px; `labels` 1=foreground / 0=background, one per point.
    public func predict(points: [[Float]], labels: [Int])
        -> (mask: MLXArray, score: Float, fullLogits: MLXArray, scores: [Float]) {
        var scaled = [Float]()
        for p in points { scaled.append(p[0] / Float(origW) * 1024); scaled.append(p[1] / Float(origH) * 1024) }
        let coords = MLXArray(scaled, [points.count, 2])
        let (masks, iou) = model.segment(input: input!, coordsPx: coords, labels: labels)  // (3,256,256),(3,)
        // postprocess: 256 → source res (bilinear), treating the 3 masks as channels
        let m4 = masks.transposed(1, 2, 0).reshaped([1, 256, 256, 3])
        let full = EdgeTAMImage.bilinear(m4, outH: origH, outW: origW)            // (1,H,W,3)
        let scores = iou.asArray(Float.self)
        let best = scores.enumerated().max(by: { $0.element < $1.element })!.offset
        let logits = full[0, 0..., 0..., 0...].transposed(2, 0, 1)                // (3,H,W)
        let bestMask = full[0, 0..., 0..., best] .> 0                             // (H,W) bool→{0,1}
        bestMask.eval()
        return (bestMask, scores[best], logits, scores)
    }
}
