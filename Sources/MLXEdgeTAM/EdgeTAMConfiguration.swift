import Foundation
import MLXToolKit

/// Configuration for the EdgeTAM `promptSegment` package. Single 54 MB checkpoint (image-mode promptable
/// segmentation); weights resolve under the engine model store (`modelsRootDirectory` + repo + file).
public struct EdgeTAMConfiguration: PackageConfiguration, ModelStorable {
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
