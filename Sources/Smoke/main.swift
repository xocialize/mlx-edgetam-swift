import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import ArgumentParser
import MLX
import EdgeTAM

/// EdgeTAM image-mode parity gate: encoder image_embed + decoder masks/iou vs the PyTorch goldens (CPU fp32).
/// With --postproc / --image: also validate the predictor's bilinear postprocess + end-to-end click→mask.
@main
struct Smoke: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "edgetam-smoke",
        abstract: "EdgeTAM MLX-Swift image-mode parity gate + predictor.")

    @Option(name: .long) var weights: String
    @Option(name: .long) var parity: String
    @Option(name: .long, help: "Postprocess fixture (masks_full + scores) for the bilinear check.") var postproc: String?
    @Option(name: .long, help: "Image to run the end-to-end predictor on.") var image: String?
    @Option(name: .long, help: "Click 'x,y' in source px for --image.") var point: String = "500,375"
    @Option(name: .long, help: "Output mask PNG for --image.") var out: String?

    func run() throws {
        Device.setDefault(device: Device(.cpu))
        let w = try MLX.loadArrays(url: URL(fileURLWithPath: weights)).mapValues { $0.asType(.float32) }
        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: parity))
        let model = EdgeTAMModel(weights: w)

        let (emb, _, _) = model.encode(fx["enc_input"]!.asType(.float32))
        emb.eval()
        let de = MLX.abs(emb - fx["image_embed"]!.asType(.float32)).max().item(Float.self)
        report("image_embed", de, 1e-3)

        let labels = fx["labels"]!.asArray(Float.self).map { Int($0) }
        let (masks, iou) = model.segment(input: fx["enc_input"]!.asType(.float32),
                                         coordsPx: fx["unnorm_coords"]!.asType(.float32), labels: labels)
        masks.eval(); iou.eval()
        let dm = MLX.abs(masks - fx["masks_raw"]!.asType(.float32)).max().item(Float.self)
        let di = MLX.abs(iou - fx["scores"]!.asType(.float32)).max().item(Float.self)
        report("masks", dm, 1e-2)
        report("iou", di, 1e-3)
        let iv = iou.asArray(Float.self).map { (round($0 * 1000) / 1000) }
        print("  iou \(iv) vs gold \(fx["scores"]!.asArray(Float.self).map { round($0 * 1000) / 1000 })")
        var failed = de >= 1e-3 || dm >= 1e-2 || di >= 1e-3

        // Postprocess (bilinear) check: golden raw masks → bilinear to full-res vs masks_full golden.
        if let postproc {
            let pf = try MLX.loadArrays(url: URL(fileURLWithPath: postproc))
            let gFull = pf["masks_full"]!.asType(.float32)                 // (3,1200,1800)
            let (H, W) = (gFull.dim(1), gFull.dim(2))
            let m4 = masks.transposed(1, 2, 0).reshaped([1, 256, 256, 3])
            let full = EdgeTAMImage.bilinear(m4, outH: H, outW: W)[0, 0..., 0..., 0...].transposed(2, 0, 1)
            full.eval()
            let df = MLX.abs(full - gFull).max().item(Float.self)
            report("postproc", df, 1e-2); failed = failed || df >= 1e-2
        }

        // End-to-end: load image, click, segment → mask; IoU vs the golden best mask.
        if let image, let postproc {
            guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: image) as CFURL, nil),
                  let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw ExitCode(1) }
            let pf = try MLX.loadArrays(url: URL(fileURLWithPath: postproc))
            let c = point.split(separator: ",").map { Float($0)! }
            let predictor = EdgeTAMPredictor(weights: w)
            predictor.setImage(cg)
            let r = predictor.predict(point: (c[0], c[1]))
            // golden best mask
            let gScores = pf["scores"]!.asArray(Float.self)
            let gBest = gScores.enumerated().max(by: { $0.element < $1.element })!.offset
            let gMask = (pf["masks_full"]!.asType(.float32)[gBest] .> 0)
            let inter = (r.mask * gMask).sum().item(Float.self)
            let uni = ((r.mask + gMask) .> 0).asType(.float32).sum().item(Float.self)
            let iou2 = inter / uni
            print(String(format: "[edgetam-smoke] e2e  %dx%d click(%g,%g) score=%.3f  IoU-vs-PyTorch=%.4f  %@",
                         cg.width, cg.height, c[0], c[1], r.score, iou2, iou2 > 0.95 ? "OK ✅" : "FAIL ❌"))
            if let out {
                let png = NSMutableData()
                let d = CGImageDestinationCreateWithData(png, UTType.png.identifier as CFString, 1, nil)!
                CGImageDestinationAddImage(d, EdgeTAMImage.maskCGImage(r.mask), nil); CGImageDestinationFinalize(d)
                try (png as Data).write(to: URL(fileURLWithPath: out))
            }
            failed = failed || iou2 <= 0.95
        }
        if failed { throw ExitCode(1) }
    }

    private func report(_ name: String, _ err: Float, _ thr: Float) {
        print(String(format: "[edgetam-smoke] %-12s max_abs=%.3e  thr=%.0e  %@", (name as NSString).utf8String!, err, thr,
                     err < thr ? "OK ✅" : "FAIL ❌"))
    }
}
