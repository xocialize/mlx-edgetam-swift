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
    @Option(name: .long, help: "Also run a box prompt 'x0,y0,x1,y1' (source px) on the same image.") var box: String?
    @Flag(name: .long, help: "Request the soft matte (mode softMatte) for the box run.") var soft = false

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
                     secs, Double(MLX.Memory.peakMemory) / 1e9, out))
        // Same image again: the package keeps the encoder features (SAM2 set_image) → decoder-only.
        let t1 = Date()
        let again = try await pkg.run(req) as! PromptSegmentResponse
        print(String(format: "[pkg] repeat (same image, cached features) score=%.3f  %.3fs", again.score, Date().timeIntervalSince(t1)))
        if let box {
            let b = box.split(separator: ",").map { Float($0)! }
            let breq = PromptSegmentRequest(image: Image(format: .jpeg, data: data), box: b,
                                            mode: soft ? EdgeTAMPackage.softMatte : nil)
            let t2 = Date()
            let br = try await pkg.run(breq) as! PromptSegmentResponse
            let bout = (out as NSString).deletingPathExtension + "-box.png"
            try br.matte.data.write(to: URL(fileURLWithPath: bout))
            print(String(format: "[pkg] box %@ → kind=%@ score=%.3f  %.3fs → %@", box, br.matte.kind.rawValue,
                         br.score, Date().timeIntervalSince(t2), bout))
        }
    }
}
