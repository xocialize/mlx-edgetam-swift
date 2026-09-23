import Foundation
import MLXToolKit

/// Configuration for the EdgeTAM `promptSegment` / `trackObject` package. One 28 MB fp16 checkpoint; weights
/// resolve under the engine model store. `WeightSourcing` lets the engine (≥ contract 1.24) materialize them
/// into the canonical `<root>/models--mlx-community--EdgeTAM-fp16/` BEFORE load — the directory its install
/// marker lives in. (≤ v0.4.x declared no sources, so the engine wrote only the marker there while the package's
/// own swift-transformers download landed in `<root>/models/mlx-community/EdgeTAM-fp16/`: AB-T-0171.)
public struct EdgeTAMConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    public var repo: String
    public var weightsFile: String
    public var quant: Quant
    public var modelsRootDirectory: URL?
    /// Direct weights path, bypassing model-store resolution (pre-resolved caller / CLI smoke).
    public var weightsURL: URL?

    public init(repo: String = "mlx-community/EdgeTAM-fp16",
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

extension EdgeTAMConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [WeightSource(role: "weights", repo: repo, revision: nil, matching: [weightsFile])]
    }

    /// Honors the explicit `weightsURL` first, then the default store probe (MS-2: flat or snapshot layout).
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let weightsURL, FileManager.default.fileExists(atPath: weightsURL.path) { return [] }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }
}
