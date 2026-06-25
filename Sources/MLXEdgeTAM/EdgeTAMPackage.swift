import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import Hub
import EdgeTAM

/// The conformant `promptSegment` ModelPackage over the EdgeTAM image-mode core. One surface: Image +
/// point/box prompts → a `Matte` of the prompted object + the model's IoU `score`. The engine constructs
/// / loads / evicts it (C13). Image-mode only (video masklet tracking is a future package surface).
@InferenceActor
public final class EdgeTAMPackage: ModelPackage {
    public typealias Configuration = EdgeTAMConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),  // EdgeTAM Apache / port MIT
            provenance: Provenance(sourceRepo: "mlx-community/EdgeTAM-fp16", revision: "main", tier: 2),
            requirements: RequirementsManifest(
                // Measured (M-Max, 1800×1200 source, fp32): peak 0.42 GB — EdgeTAM is on-device-tiny
                // (54 MB, RepViT encoder @ fixed 1024²). 1 GB envelope covers larger sources + fp16/fp32.
                footprints: [QuantFootprint(quant: .fp16, residentBytes: 1_000_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0))
            ),
            surfaces: [
                PromptSegmentContract.descriptor(
                    name: "edgetam",
                    summary: "EdgeTAM promptable segmentation — image + point/box → object mask. "
                        + "On-device SAM 2; click/box-select for cutout + erase masks.")
            ])
    }

    private let configuration: Configuration
    private var predictor: EdgeTAMPredictor?

    public nonisolated init(configuration: Configuration) { self.configuration = configuration }

    public func load() async throws { if predictor == nil { predictor = try await build() } }
    public func unload() async { predictor = nil }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard request.capability == .promptSegment, let req = request as? PromptSegmentRequest else {
            throw EdgeTAMError.unsupportedCapability(request.capability)
        }
        guard !req.points.isEmpty else { throw EdgeTAMError.noPrompt }  // V1: point prompts (box = follow-up)
        try Task.checkCancellation()
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

    // MARK: build / weights

    private func build() async throws -> EdgeTAMPredictor {
        let url = try await weightsURL()
        return try EdgeTAMPredictor.fromPretrained(url.path, dtype: Self.dtype(configuration.quant))
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
    }
}

public extension EdgeTAMPackage {
    nonisolated static var registration: PackageRegistration { .of(EdgeTAMPackage.self) }
}
