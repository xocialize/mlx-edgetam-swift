import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import Hub
import EdgeTAM
import FrameStreamNative

/// The conformant EdgeTAM ModelPackage. Two surfaces over one 54 MB checkpoint (the engine constructs /
/// loads / evicts it, C13):
///   • `promptSegment` — Image + point/box prompts → a `Matte` of the prompted object + IoU `score`.
///   • `trackObject` — `Video` + a click on one frame → a per-frame `Matte` track (masklet propagation),
///     EdgeTAM's video memory stack. The `Video` bytes are decoded to frames via `FrameStreamNative`
///     (FFmpeg-free native path); V1 single object.
@InferenceActor
public final class EdgeTAMPackage: ModelPackage {
    public typealias Configuration = EdgeTAMConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),  // EdgeTAM Apache / port MIT
            provenance: Provenance(sourceRepo: "mlx-community/EdgeTAM-fp16", revision: "main", tier: 2),
            requirements: RequirementsManifest(
                // Image-mode measured (M-Max, 1800×1200 source, fp32): peak 0.42 GB — EdgeTAM is
                // on-device-tiny (54 MB, RepViT encoder @ fixed 1024²). trackObject runs the same
                // per-frame forward sequentially; its working set scales with clip length — MEASURED on
                // GPU (960×540 source, fp32): 5 frames = 1.07 GB, 30 frames = 1.79 GB → ~0.9 GB fixed +
                // ~30 MB/frame (stacked input + retained per-frame mattes + memory bank + buffer cache).
                // The 2 GB envelope covers ~35-frame clips; longer clips want streaming (don't pre-stack
                // frames, encode mattes incrementally, mx.clear_cache per step) — a V2 optimization.
                footprints: [QuantFootprint(quant: .fp16, residentBytes: 2_000_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0))
            ),
            surfaces: [
                PromptSegmentContract.descriptor(
                    name: "edgetam",
                    summary: "EdgeTAM promptable segmentation — image + point/box → object mask. "
                        + "On-device SAM 2; click/box-select for cutout + erase masks."),
                TrackObjectContract.descriptor(
                    name: "edgetam-track",
                    summary: "EdgeTAM promptable video tracking — video + a click on one frame → a "
                        + "per-frame object mask track (masklet). On-device SAM 2; Erase/Extract across frames."),
            ])
    }

    private let configuration: Configuration
    private var predictor: EdgeTAMPredictor?
    private var videoPredictor: EdgeTAMVideoPredictor?

    public nonisolated init(configuration: Configuration) { self.configuration = configuration }

    public func load() async throws { if predictor == nil { predictor = try await build() } }
    public func unload() async { predictor = nil; videoPredictor = nil }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try Task.checkCancellation()
        switch request {
        case let req as PromptSegmentRequest: return try await runSegment(req)
        case let req as TrackObjectRequest:   return try await runTrack(req)
        default: throw EdgeTAMError.unsupportedCapability(request.capability)
        }
    }

    // MARK: promptSegment (image)

    private func runSegment(_ req: PromptSegmentRequest) async throws -> PromptSegmentResponse {
        guard !req.points.isEmpty else { throw EdgeTAMError.noPrompt }  // V1: point prompts (box = follow-up)
        if predictor == nil { predictor = try await build() }
        let cg = try Self.decode(req.image)
        predictor!.setImage(cg)
        let labels = req.pointLabels.count == req.points.count ? req.pointLabels : Array(repeating: 1, count: req.points.count)
        let r = predictor!.predict(points: req.points, labels: labels)
        try Task.checkCancellation()
        let png = try Self.encodePNG(EdgeTAMImage.maskCGImage(r.mask))
        return PromptSegmentResponse(
            matte: Matte(format: .png, data: png, width: cg.width, height: cg.height, kind: .binary),
            score: r.score)
    }

    // MARK: trackObject (video masklet)

    // ENHANCEMENT(v2) — trackObject hardening. Original three limitations, all additive (no contract bump):
    //   1. MULTI-OBJECT — DONE in the CORE. `EdgeTAMModel.propagate`/`EdgeTAMVideoPredictor.track` now take
    //      `[ObjectPrompt]`, share the per-frame RepViT+FPN encode once, and run a per-object memory bank +
    //      decode + memory-encode → one track per object. SURFACE stays request-per-object: one
    //      `TrackObjectRequest` = one object's prompt set → one `[Matte]` track, so the contract carries
    //      multi-object with NO bump (the documented "lane-ready" interpretation — a caller tracking N
    //      objects issues N requests, or drives `EdgeTAMVideoPredictor.track(frames:objects:)` directly for
    //      the shared-encode win in one pass). A future additive `objects` grouping field could batch the
    //      surface end-to-end if a caller needs it; intentionally deferred to honor "prefer no contract bump".
    //   2. BOX PROMPT — DONE. `req.box` is wired through `forwardSamHeads`→`embedPrompt` (SAM corner tokens).
    //   3. LONG-CLIP STREAMING — DONE. `propagate` pulls frames one at a time and emits each Matte as it
    //      lands (PNG-encoded here, MLX mask dropped), `MLX.GPU.clearCache()` per frame → flat GPU footprint.
    private func runTrack(_ req: TrackObjectRequest) async throws -> TrackObjectResponse {
        // One object per request (the lane-ready, no-bump multi-object surface): point and/or box prompt.
        guard !req.points.isEmpty || req.box != nil else { throw EdgeTAMError.noPrompt }
        // Decode the Video bytes to frames (FrameStreamNative, native containers only).
        let frames = try await Self.decodeVideo(req.video)
        guard req.promptFrame >= 0, req.promptFrame < frames.count else {
            throw EdgeTAMError.promptFrameOutOfRange(req.promptFrame, frames.count)
        }
        if videoPredictor == nil { videoPredictor = try await buildVideo() }
        let labels = req.pointLabels.count == req.points.count ? req.pointLabels : Array(repeating: 1, count: req.points.count)
        // Stream: each frame's Matte is PNG-encoded and the MLX mask dropped the instant it lands, so the
        // GPU working set stays flat across long clips (the predictor clears its per-step cache). The only
        // thing that grows is the output PNG list, which the contract returns in full regardless.
        var masks: [Matte] = []; masks.reserveCapacity(frames.count)
        var scores: [Float] = []; scores.reserveCapacity(frames.count)
        try videoPredictor!.track(frames: frames, clickFrame: req.promptFrame,
                                  points: req.points, labels: labels, box: req.box) { _, fm in
            try Task.checkCancellation()
            let png = try Self.encodePNG(EdgeTAMImage.maskCGImage(fm.mask))
            masks.append(Matte(format: .png, data: png, width: fm.mask.dim(1), height: fm.mask.dim(0), kind: .binary))
            scores.append(fm.score)
        }
        return TrackObjectResponse(masks: masks, scores: scores)
    }

    // MARK: build / weights

    private func build() async throws -> EdgeTAMPredictor {
        let url = try await weightsURL()
        return try EdgeTAMPredictor.fromPretrained(url.path, dtype: Self.dtype(configuration.quant))
    }
    private func buildVideo() async throws -> EdgeTAMVideoPredictor {
        let url = try await weightsURL()
        return try EdgeTAMVideoPredictor.fromPretrained(url.path, dtype: Self.dtype(configuration.quant))
    }
    private func weightsURL() async throws -> URL {
        if let o = configuration.weightsURL {
            guard FileManager.default.fileExists(atPath: o.path) else { throw EdgeTAMError.weightsMissing(o) }
            return o
        }
        let hub = HubApi(downloadBase: configuration.modelsRootDirectory)
        let dir = try await hub.snapshot(from: configuration.repo, matching: ["*.safetensors"]) { @Sendable p in
            WeightDownloadProgress.report(fraction: p.fractionCompleted)
        }
        let url = dir.appendingPathComponent(configuration.weightsFile)
        guard FileManager.default.fileExists(atPath: url.path) else { throw EdgeTAMError.weightsMissing(url) }
        return url
    }
    private nonisolated static func dtype(_ q: Quant) -> DType {
        switch q { case .fp32: return .float32; case .bf16: return .bfloat16; default: return .float16 }
    }

    private nonisolated static func decode(_ image: Image) throws -> CGImage {
        guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw EdgeTAMError.decodeFailed }
        return cg
    }

    /// Materialize the `Video` bytes to a temp file (FrameStreamNative reads native containers from a URL)
    /// and decode every frame to a `CGImage`. The temp file is removed before returning.
    private nonisolated static func decodeVideo(_ video: Video) async throws -> [CGImage] {
        let ext = video.format == .mov ? "mov" : "mp4"
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("edgetam-track-\(UUID().uuidString).\(ext)")
        try video.data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let frames = try await NativeFrameStream.decode(input: url)
        guard !frames.isEmpty else { throw EdgeTAMError.noVideoFrames }
        return frames
    }
    private nonisolated static func encodePNG(_ cg: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
        else { throw EdgeTAMError.encodeFailed }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw EdgeTAMError.encodeFailed }
        return data as Data
    }

    public enum EdgeTAMError: Error {
        case unsupportedCapability(Capability)
        case noPrompt, weightsMissing(URL), decodeFailed, encodeFailed
        case noVideoFrames, promptFrameOutOfRange(Int, Int)
    }
}

public extension EdgeTAMPackage {
    nonisolated static var registration: PackageRegistration { .of(EdgeTAMPackage.self) }
}
