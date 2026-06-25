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

    /// Preprocess one source frame → `(1,1024,1024,3)` ImageNet-normalized model input.
    private func preprocess(_ cg: CGImage, origW: Int, origH: Int) -> MLXArray {
        let rgb = EdgeTAMImage.rgb(from: cg, width: origW, height: origH)
        let resized = EdgeTAMImage.bilinear(rgb, outH: 1024, outW: 1024)
        return (resized - mean) / std
    }

    /// **Streaming multi-object** track (flat GPU footprint). The per-frame encode is shared across every
    /// object; each `(objIdx, frameMask)` is handed to `emit` the moment it lands — nothing is pre-stacked
    /// and no output is retained inside the predictor. Each `ObjectPrompt` carries its own point/box (source
    /// px) + `clickFrame`; all frames must share the source resolution. Pair with a consumer that converts
    /// each mask out of MLX immediately (e.g. PNG-encode) so the GPU working set never accumulates.
    public func track(frames: [CGImage], objects: [ObjectPrompt],
                      emit: (_ objIdx: Int, _ frameIdx: Int, _ frameMask: FrameMask) throws -> Void) rethrows {
        precondition(!frames.isEmpty, "track requires at least one frame")
        precondition(!objects.isEmpty, "track requires at least one object")
        precondition(objects.allSatisfy { !$0.points.isEmpty || $0.box != nil },
                     "each object needs a point or box prompt")
        let (origW, origH) = (frames[0].width, frames[0].height)
        try model.propagate(frameCount: frames.count, objects: objects, origH: origH, origW: origW,
                             frame: { preprocess(frames[$0], origW: origW, origH: origH) },
                             emit: { o, idx, logits, score in
                                 try emit(o, idx, FrameMask(mask: (logits .> 0), logits: logits, score: score))
                             })
    }

    /// **Streaming single-object** track (point and/or box). Keeps the original `(clickFrame, points,
    /// labels)` call site; `box` is `[x0,y0,x1,y1]` in source px. Delegates to the multi-object core.
    public func track(frames: [CGImage], clickFrame: Int = 0, points: [[Float]] = [], labels: [Int] = [],
                      box: [Float]? = nil,
                      emit: (_ frameIdx: Int, _ frameMask: FrameMask) throws -> Void) rethrows {
        try track(frames: frames,
                  objects: [ObjectPrompt(clickFrame: clickFrame, points: points, labels: labels, box: box)],
                  emit: { _, idx, fm in try emit(idx, fm) })
    }

    /// Collecting single-object convenience → one `FrameMask` per input frame (same order). Holds every
    /// mask in memory — for the flat-footprint path, use an `emit:` overload and convert each mask as it lands.
    public func track(frames: [CGImage], clickFrame: Int = 0, points: [[Float]] = [], labels: [Int] = [],
                      box: [Float]? = nil) -> [FrameMask] {
        var out: [FrameMask] = []; out.reserveCapacity(frames.count)
        track(frames: frames, clickFrame: clickFrame, points: points, labels: labels, box: box) { _, fm in out.append(fm) }
        return out
    }

    /// Collecting multi-object convenience → one `[FrameMask]` track per object (object order preserved).
    public func track(frames: [CGImage], objects: [ObjectPrompt]) -> [[FrameMask]] {
        var out = [[FrameMask]](repeating: [], count: objects.count)
        for i in out.indices { out[i].reserveCapacity(frames.count) }
        track(frames: frames, objects: objects) { o, _, fm in out[o].append(fm) }
        return out
    }

    /// Convenience: per-frame mask overlays as opaque RGB `CGImage`s (white = object).
    public func trackMasks(frames: [CGImage], clickFrame: Int = 0, points: [[Float]], labels: [Int]) -> [CGImage] {
        track(frames: frames, clickFrame: clickFrame, points: points, labels: labels)
            .map { EdgeTAMImage.maskCGImage($0.mask) }
    }
}
