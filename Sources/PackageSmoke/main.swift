import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import ArgumentParser
import MLX
import MLXToolKit
import MLXEdgeTAM

/// Drive the conformant EdgeTAMPackage as the engine would: license gate → init → load() →
/// run(PromptSegmentRequest) with a click → write the Matte. Proves the package envelope + footprint.
@main
struct PackageSmoke: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "edgetam-package-smoke",
        abstract: "Drive EdgeTAMPackage through load()/run() with a click.")

    @Option(name: .long) var weights: String
    @Option(name: .long) var image: String
    @Option(name: .long) var point: String = "500,375"
    @Option(name: .long) var out: String
    @Option(name: .long) var dtype: String = "float32"

    func run() async throws {
        let decl = EdgeTAMPackage.manifest.license
        let gate = LicensePolicy.permissiveOnly.evaluate(decl)
        print("[pkg] license weight=\(decl.weightLicense) port=\(decl.portCodeLicense) → \(gate)")
        guard gate.isAdmitted else { throw ExitCode(1) }

        let q: Quant = dtype == "float16" ? .fp16 : (dtype == "bfloat16" ? .bf16 : .fp32)
        let cfg = EdgeTAMConfiguration(quant: q, weightsURL: URL(fileURLWithPath: weights))
        let pkg = EdgeTAMPackage(configuration: cfg)
        try await pkg.load()

        let data = try Data(contentsOf: URL(fileURLWithPath: image))
        let c = point.split(separator: ",").map { Float($0)! }
        let req = PromptSegmentRequest(image: Image(format: .jpeg, data: data),
                                       points: [[c[0], c[1]]], pointLabels: [1])
        MLX.GPU.resetPeakMemory()
        let start = Date()
        let resp = try await pkg.run(req)
        let secs = Date().timeIntervalSince(start)
        guard let r = resp as? PromptSegmentResponse else { throw ExitCode(1) }
        try r.matte.data.write(to: URL(fileURLWithPath: out))
        print(String(format: "[pkg] run → matte %dx%d kind=%@ score=%.3f  (%.2fs, peak %.2f GB) → %@",
                     r.matte.width ?? 0, r.matte.height ?? 0, r.matte.kind.rawValue, r.score,
                     secs, Double(MLX.GPU.peakMemory) / 1e9, out))
    }
}
