import Foundation
import CryptoKit
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import Hub
import EdgeTAM
import MLXProfiling

/// The `promptSegment` path shared by every SAM2-family package here (EdgeTAM, SAM 2.1): one predictor plus the
/// SHA-256 of the image it holds features for. The encoder runs once per IMAGE (SAM2 `set_image`), so a repeat
/// request on the same image bytes — the interactive click / add-point / box loop — is decoder-only.
final class PromptSegmentSession: @unchecked Sendable {
    let predictor: EdgeTAMPredictor
    private var imageKey: String?
    init(_ predictor: EdgeTAMPredictor) { self.predictor = predictor }

    func run(_ req: PromptSegmentRequest, softMode: Mode, profileName: String) throws -> PromptSegmentResponse {
        guard !req.points.isEmpty || req.box != nil else { throw PromptSegmentError.noPrompt }
        let labels = req.pointLabels.count == req.points.count ? req.pointLabels : Array(repeating: 1, count: req.points.count)
        let key = SHA256.hash(data: req.image.data).description
        let prof = MLXProfiler.shared
        var cg: CGImage?
        if key != imageKey {
            cg = try Self.decode(req.image)
            prof.beginRun("\(profileName) promptSegment points=\(req.points.count) \(cg!.width)x\(cg!.height)")
        } else {
            prof.beginRun("\(profileName) promptSegment points=\(req.points.count) (cached image)")
        }
        let r = prof.region("segment", "forward") { () -> EdgeTAMPredictor.Prediction in
            if let cg { predictor.setImage(cg) }
            return predictor.predict(points: req.points, labels: labels, box: req.box)
        }
        imageKey = key
        prof.endRun(denominators: ["image": 1])
        try Task.checkCancellation()
        let soft = req.mode == softMode
        let png = try Self.encodePNG(soft ? EdgeTAMImage.matteCGImage(r.soft) : EdgeTAMImage.maskCGImage(r.mask))
        return PromptSegmentResponse(
            matte: Matte(format: .png, data: png, width: r.mask.dim(1), height: r.mask.dim(0),
                         kind: soft ? .softAlpha : .binary),
            score: r.score)
    }

    static func decode(_ image: Image) throws -> CGImage {
        guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw PromptSegmentError.decodeFailed }
        return cg
    }
    static func encodePNG(_ cg: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
        else { throw PromptSegmentError.encodeFailed }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw PromptSegmentError.encodeFailed }
        return data as Data
    }
}

enum PromptSegmentError: Error { case noPrompt, decodeFailed, encodeFailed }

/// Store-first weights resolution shared by the packages (AB-T-0171): explicit URL → the engine's flat
/// `models--<org>--<name>/` → hub snapshot → the swift-transformers layout v0.4.x downloaded into → hub download
/// (engine-less / pre-1.24 callers only; an engine ≥ 1.24 has already materialized the declared source).
enum StoreWeights {
    static func stored(repo: String, file: String, root: URL?) -> URL? {
        guard let root else { return nil }
        let store = ModelStore(root: root)
        let candidates = [store.directory(for: repo), store.snapshotDirectory(for: repo),
                          root.appending(path: "models/\(repo)", directoryHint: .isDirectory)]
        return candidates.compactMap { $0?.appending(path: file) }.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    static func resolve(repo: String, file: String, root: URL?, explicit: URL?) async throws -> URL {
        if let explicit {
            guard FileManager.default.fileExists(atPath: explicit.path) else { throw WeightsMissing(url: explicit) }
            return explicit
        }
        if let found = stored(repo: repo, file: file, root: root) { return found }
        let hub = root.map { HubApi(downloadBase: $0) } ?? HubApi()
        let dir = try await hub.snapshot(from: repo, matching: [file]) { @Sendable p in
            WeightDownloadProgress.report(fraction: p.fractionCompleted)
        }
        let url = dir.appendingPathComponent(file)
        guard FileManager.default.fileExists(atPath: url.path) else { throw WeightsMissing(url: url) }
        return url
    }

    struct WeightsMissing: Error { let url: URL }
}
