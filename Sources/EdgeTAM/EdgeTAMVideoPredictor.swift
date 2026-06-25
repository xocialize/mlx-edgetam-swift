import Foundation
import CoreGraphics
import MLX

/// End-to-end EdgeTAM **video masklet tracker**: hand it the decoded frames + one click on a frame,
/// get back a per-frame binary mask at source resolution. Mirrors `EdgeTAMPredictor` (1024² bilinear +
/// ImageNet-norm preprocess) but drives the stateful `EdgeTAMModel.propagate` memory bank.
///
/// **Boundary (Forge plugin contract):** this stays a pure `frames → masks` plugin — CoreGraphics + MLX
/// only, no AVFoundation. Video-FILE decode (URL → frames) and mask muxing are the **shell's** job,
/// bridged with `FrameStreamNative` (the same FFmpeg-free native path RIFE/SeedVR2 use): decode each
/// `CVPixelBuffer` → `CGImage`, collect, call `track`, then write the masks back out shell-side.
public final class EdgeTAMVideoPredictor: @unchecked Sendable {

    private let model: EdgeTAMModel
    private let mean = MLXArray([0.485, 0.456, 0.406] as [Float], [1, 1, 1, 3])
    private let std = MLXArray([0.229, 0.224, 0.225] as [Float], [1, 1, 1, 3])

    public init(weights: [String: MLXArray]) { self.model = EdgeTAMModel(weights: weights) }

    public static func fromPretrained(_ path: String, dtype: DType = .float32) throws -> EdgeTAMVideoPredictor {
        let w = try MLX.loadArrays(url: URL(fileURLWithPath: path)).mapValues { $0.asType(dtype) }
        return EdgeTAMVideoPredictor(weights: w)
    }

    public struct FrameMask {
        public let mask: MLXArray   // (H,W) bool→{0,1} at source resolution
        public let logits: MLXArray // (H,W) raw upsampled logits
        public let score: Float     // object-score logit (>0 ≈ present; ≤0 ≈ occluded/absent)
    }

    /// Track a single object across `frames` from point clicks on `clickFrame`. `points` are `[x,y]` in
    /// source px (foreground/background per `labels`: 1/0); all frames must share the source resolution.
    /// Returns one `FrameMask` per input frame (same order).
    public func track(frames: [CGImage], clickFrame: Int = 0, points: [[Float]], labels: [Int]) -> [FrameMask] {
        precondition(!frames.isEmpty, "track requires at least one frame")
        precondition(!points.isEmpty, "track requires at least one point prompt")
        let (origW, origH) = (frames[0].width, frames[0].height)
        // Preprocess each frame → (1,1024,1024,3) ImageNet-norm, stack → (T,1024,1024,3).
        var pre: [MLXArray] = []
        pre.reserveCapacity(frames.count)
        for cg in frames {
            let rgb = EdgeTAMImage.rgb(from: cg, width: origW, height: origH)
            let resized = EdgeTAMImage.bilinear(rgb, outH: 1024, outW: 1024)
            pre.append((resized - mean) / std)
        }
        let stacked = MLX.concatenated(pre, axis: 0)                    // (T,1024,1024,3)
        let (logits, scores) = model.propagate(frames: stacked, clickFrame: clickFrame,
                                               points: points, labels: labels, origH: origH, origW: origW)
        return zip(logits, scores).map { FrameMask(mask: ($0.0 .> 0), logits: $0.0, score: $0.1) }
    }

    /// Convenience: per-frame mask overlays as opaque RGB `CGImage`s (white = object).
    public func trackMasks(frames: [CGImage], clickFrame: Int = 0, points: [[Float]], labels: [Int]) -> [CGImage] {
        track(frames: frames, clickFrame: clickFrame, points: points, labels: labels)
            .map { EdgeTAMImage.maskCGImage($0.mask) }
    }
}
