import Foundation
import MLX
import MLXToolKit
import EdgeTAM

/// Configuration for `SAM21Package` — SAM 2.1 (Hiera) image-mode weights converted from Meta's official
/// checkpoint by `oracle/convert_sam21.py` (image path only). `WeightSourcing` lets the engine materialize them
/// into `<root>/models--mlx-community--SAM2.1-hiera-small-fp16/` before load.
public struct SAM21Configuration: PackageConfiguration, ModelStorable, QuantConfigured {
    public var repo: String
    public var weightsFile: String
    public var quant: Quant
    public var modelsRootDirectory: URL?
    /// Direct weights path, bypassing the store (parity work / CLI smoke).
    public var weightsURL: URL?

    public init(repo: String = "mlx-community/SAM2.1-hiera-small-fp16",
                weightsFile: String = "model.safetensors",
                quant: Quant = .fp16,
                modelsRootDirectory: URL? = nil,
                weightsURL: URL? = nil) {
        self.repo = repo
        self.weightsFile = weightsFile
        self.quant = quant
        self.modelsRootDirectory = modelsRootDirectory
        self.weightsURL = weightsURL
    }
}

extension SAM21Configuration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [WeightSource(role: "weights", repo: repo, revision: nil, matching: [weightsFile])]
    }
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let weightsURL, FileManager.default.fileExists(atPath: weightsURL.path) { return [] }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }
}

/// SAM 2.1-small `promptSegment` — the QUALITY tier for interactive selection (AB-T-0173). Same request/response
/// semantics as `EdgeTAMPackage`'s image surface (point / box / point+box prompts, SAM2 multimask rule, the
/// `softMatte` mode, per-image feature cache) over the shared SAM2-family core with the Hiera backbone.
/// Measured vs EdgeTAM on the Forge fixtures (upstream PyTorch, which this port matches through `setImage`):
/// a belly click on the flat-shaded fox selects the whole fox (IoU 0.968 vs 0.008), and a click on one square
/// of the gate page selects only that square (EdgeTAM also adds the top-left one). EdgeTAM stays the fast and
/// video (`trackObject`) default; SAM 2.1's video memory stack is not ported.
@InferenceActor
public final class SAM21Package: ModelPackage {
    public typealias Configuration = SAM21Configuration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // SAM 2.1 code + checkpoints: Apache-2.0 (facebookresearch/sam2); port MIT.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "mlx-community/SAM2.1-hiera-small-fp16", revision: "main", tier: 2),
            requirements: RequirementsManifest(
                // SPLIT footprint, MEASURED (edgetam-package-smoke --sam21, Release, M5 Max, fp16 weights + fp16
                // activations, AB-T-0173): 38.5 M image-path params (77 MB on disk); one Hiera forward per image at
                // the fixed 1024² input, later prompts on the same image decoder-only (11–18 ms).
                //   MLX active after the forward 0.10 GB (weights + cached features) → resident 0.25 GB.
                //   MLX peak 0.98 GB (1024²) / 1.13 GB (1800×1200); process phys_footprint 2.35 / 2.79 GB (the pool
                //   keeps the transient) → activation = 2.79 − 0.25 ≈ 2.6 GB, declared against phys (R-MEM-1).
                //   (fp32 activations peaked at 2.18 GB MLX — why the Hiera path computes in the weight dtype.)
                footprints: [QuantFootprint(quant: .fp16, residentBytes: 250_000_000,
                                            peakActivationBytes: 2_600_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0))
            ),
            surfaces: [
                PromptSegmentContract.descriptor(
                    name: "sam2.1-small",
                    summary: "SAM 2.1 (Hiera-S) promptable segmentation — image + point/box → object mask. "
                        + "Quality tier for click/box selection (better single clicks than EdgeTAM on flat or "
                        + "synthetic images); image only.",
                    modes: [softMatte]),
            ])
    }

    /// Same mode as `EdgeTAMPackage.softMatte`: the anti-aliased `.softAlpha` matte.
    public nonisolated static let softMatte: Mode = EdgeTAMPackage.softMatte

    private let configuration: Configuration
    private var session: PromptSegmentSession?

    public nonisolated init(configuration: Configuration) { self.configuration = configuration }

    public func load() async throws { if session == nil { session = PromptSegmentSession(try await build()) } }
    public func unload() async {
        session = nil
        MLX.Memory.clearCache()                                        // release the pool so evict reclaims
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try Task.checkCancellation()                                   // CAN-1 entry checkpoint
        guard let req = request as? PromptSegmentRequest else {
            throw SAM21Error.unsupportedCapability(request.capability)
        }
        guard !req.points.isEmpty || req.box != nil else { throw SAM21Error.noPrompt }
        if session == nil { session = PromptSegmentSession(try await build()) }
        return try session!.run(req, softMode: Self.softMatte, profileName: "sam2.1")
    }

    private func build() async throws -> EdgeTAMPredictor {
        let url: URL
        do {
            url = try await StoreWeights.resolve(repo: configuration.repo, file: configuration.weightsFile,
                                                 root: configuration.modelsRootDirectory,
                                                 explicit: configuration.weightsURL)
        } catch let e as StoreWeights.WeightsMissing { throw SAM21Error.weightsMissing(e.url) }
        let dtype: DType = configuration.quant == .fp32 ? .float32 : (configuration.quant == .bf16 ? .bfloat16 : .float16)
        let p = try EdgeTAMPredictor.fromPretrained(url.path, dtype: dtype)
        guard p.model.hiera != nil else { throw SAM21Error.notSAM21Weights(url) }
        return p
    }

    nonisolated static func storedWeights(_ c: SAM21Configuration) -> URL? {
        StoreWeights.stored(repo: c.repo, file: c.weightsFile, root: c.modelsRootDirectory)
    }

    public enum SAM21Error: Error {
        case unsupportedCapability(Capability)
        case noPrompt, weightsMissing(URL), notSAM21Weights(URL)
    }
}

public extension SAM21Package {
    nonisolated static var registration: PackageRegistration { .of(SAM21Package.self) }
}
