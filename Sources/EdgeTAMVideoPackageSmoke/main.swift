import Foundation
import AVFoundation
import CoreGraphics
import ImageIO
import ArgumentParser
import MLX
import MLXToolKit
import MLXEdgeTAM
import EdgeTAM

/// End-to-end `trackObject` SURFACE smoke: encode N source frames → a real `.mov`, then drive the
/// conformant `EdgeTAMPackage` exactly as the engine would (license gate → run(TrackObjectRequest)) so
/// the whole runtime path exercises — `Video` bytes → `FrameStreamNative` decode → masklet propagation →
/// per-frame `[Matte]`. Reports the measured peak footprint and per-frame IoU vs the propagate goldens
/// (codec-tolerant: ProRes encode/decode drifts a few px from the raw-frame parity smoke).
@main
struct VideoPackageSmoke: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "edgetam-video-package-smoke",
        abstract: "Drive EdgeTAMPackage.trackObject end-to-end on a real video.")

    @Option(name: .long) var weights: String
    @Option(name: .long, help: "parity_video.safetensors (mask_f0..4 goldens)") var parity: String
    @Option(name: .long, help: "Dir of source frames 00000.jpg.. (bedroom).") var framesDir: String
    @Option(name: .long) var point: String = "210,350"
    @Option(name: .long) var frames: Int = 5

    func run() async throws {
        // License gate (as the engine does).
        let decl = EdgeTAMPackage.manifest.license
        let gate = LicensePolicy.permissiveOnly.evaluate(decl)
        print("[vpkg] license weight=\(decl.weightLicense) port=\(decl.portCodeLicense) → \(gate)  contract=\(EdgeTAMPackage.manifest.contractVersion)")
        guard gate.isAdmitted else { throw ExitCode(1) }

        // Encode the first N source frames → a near-lossless ProRes .mov.
        let cgs = try Self.loadFrames(framesDir, count: frames)
        let movURL = FileManager.default.temporaryDirectory.appendingPathComponent("edgetam-vpkg-\(UUID().uuidString).mov")
        try await Self.encodeMov(cgs, to: movURL)
        defer { try? FileManager.default.removeItem(at: movURL) }
        let videoBytes = try Data(contentsOf: movURL)
        print(String(format: "[vpkg] encoded %d frames %dx%d → %.2f MB .mov", cgs.count, cgs[0].width, cgs[0].height,
                     Double(videoBytes.count) / 1e6))

        // Drive the package surface.
        let cfg = EdgeTAMConfiguration(quant: .fp32, weightsURL: URL(fileURLWithPath: weights))
        let pkg = EdgeTAMPackage(configuration: cfg)
        let video = Video(format: .mov, data: videoBytes)
        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: parity)).mapValues { $0.asType(.float32) }
        let nGold = 5

        // Drive ONE TrackObjectRequest (the request-per-object multi-object surface) and score it per-frame
        // vs `goldPrefix`_f{i}. Returns (minIoU, seconds). Codec-tolerant: ProRes encode/decode drifts a few px.
        func track(_ label: String, points: [[Float]], box: [Float]?, goldPrefix: String) async throws -> Float {
            let req = TrackObjectRequest(video: video, promptFrame: 0, points: points,
                                         pointLabels: Array(repeating: 1, count: points.count), box: box)
            let resp = try await pkg.run(req)
            guard let r = resp as? TrackObjectResponse, r.masks.count == cgs.count else { throw ExitCode(1) }
            var minIoU: Float = 1
            for i in 0 ..< min(nGold, r.masks.count) {
                guard let goldArr = fx["\(goldPrefix)_f\(i)"] else { continue }
                let m = r.masks[i]
                let pred = Self.matteBinary(m.data, width: m.width ?? 0, height: m.height ?? 0)
                let gold = (goldArr .> 0).asType(.float32)
                let inter = (pred * gold).sum().item(Float.self)
                let uni = ((pred + gold) .> 0).asType(.float32).sum().item(Float.self)
                minIoU = min(minIoU, uni > 0 ? inter / uni : 1)
            }
            let lbl = label.padding(toLength: 22, withPad: " ", startingAt: 0)
            print("[vpkg] \(lbl) " + String(format: "min_IoU=%.4f  score f0=%.3f", minIoU, r.scores[0]))
            return minIoU
        }

        // Object 0 (the CLI point, default boy) — measure peak + timing on this primary request.
        let c = point.split(separator: ",").map { Float($0)! }
        MLX.GPU.resetPeakMemory()
        let start = Date()
        let boyIoU = try await track("obj0 point", points: [[c[0], c[1]]], box: nil, goldPrefix: "mask")
        let secs = Date().timeIntervalSince(start)
        // Object 1 (girl) — a second request, the request-per-object multi-object lane.
        let girlIoU = try await track("obj1 point (girl)", points: [[400, 240]], box: nil, goldPrefix: "mo_obj1")
        // Box prompt — points empty, box only (ENHANCEMENT v2 #2).
        let bx = fx["box_xyxy"]!
        let box: [Float] = (0 ..< 4).map { bx[$0].item(Float.self) }
        let boxIoU = try await track("box prompt", points: [], box: box, goldPrefix: "box")

        let ok = boyIoU > 0.80 && girlIoU > 0.80 && boxIoU > 0.80
        print(String(format: "[vpkg] trackObject end-to-end  boy=%.3f girl=%.3f box=%.3f thr=0.80  (boy %.2fs, peak %.2f GB)  %@",
                     boyIoU, girlIoU, boxIoU, secs, Double(MLX.GPU.peakMemory) / 1e9, ok ? "OK ✅" : "FAIL ❌"))
        if !ok { throw ExitCode(1) }
    }

    // MARK: helpers

    private static func loadFrames(_ dir: String, count: Int) throws -> [CGImage] {
        let files = (try FileManager.default.contentsOfDirectory(atPath: dir))
            .filter { $0.hasSuffix(".jpg") || $0.hasSuffix(".png") }.sorted().prefix(count)
        return try files.map { name in
            let url = URL(fileURLWithPath: dir).appendingPathComponent(name)
            guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw ExitCode(2) }
            return cg
        }
    }

    /// Near-lossless ProRes422 .mov so decode drifts minimally from the raw frames.
    private static func encodeMov(_ frames: [CGImage], to url: URL) async throws {
        let (w, h) = (frames[0].width, frames[0].height)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.proRes422,
            AVVideoWidthKey: w, AVVideoHeightKey: h,
        ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
        ])
        writer.add(input)
        guard writer.startWriting() else { throw writer.error ?? ExitCode(3) }
        writer.startSession(atSourceTime: .zero)
        let fps: Int32 = 24
        for (i, cg) in frames.enumerated() {
            while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 2_000_000) }
            guard let pool = adaptor.pixelBufferPool else { throw ExitCode(3) }
            var pbOut: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pbOut)
            guard let pb = pbOut else { throw ExitCode(3) }
            CVPixelBufferLockBaseAddress(pb, [])
            let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: w, height: h, bitsPerComponent: 8,
                                bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
            ctx?.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            CVPixelBufferUnlockBaseAddress(pb, [])
            adaptor.append(pb, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: fps))
        }
        input.markAsFinished()
        await writer.finishWriting()
        if writer.status == .failed { throw writer.error ?? ExitCode(3) }
    }

    /// Rasterize a binary matte PNG → `(H,W)` {0,1} MLXArray (channel 0 > 0.5).
    private static func matteBinary(_ png: Data, width: Int, height: Int) -> MLXArray {
        guard let src = CGImageSourceCreateWithData(png as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return MLXArray.zeros([height, width]) }
        let rgb = EdgeTAMImage.rgb(from: cg, width: width, height: height)   // (1,H,W,3) [0,1]
        return (rgb[0, 0..., 0..., 0] .> 0.5).asType(.float32)
    }
}
