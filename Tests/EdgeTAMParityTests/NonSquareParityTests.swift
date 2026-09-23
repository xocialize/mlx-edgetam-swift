// NonSquareParityTests.swift — end-to-end parity THROUGH `setImage` on NON-square images (AB-T-0171).
// The pre-v0.5 gate fed PyTorch's already-preprocessed encoder input, so resize / normalisation / coordinate
// scaling / postprocess were only exercised by one truck click. This drives the real predictor path —
// CGImage → setImage → predict → source-res mask — against upstream SAM2ImagePredictor goldens
// (oracle/make_parity_golden.py → nonsquare_golden.json):
//   • gate  (1200×800, flat synthetic — the Forge Canvas Lab pointer-gate page, regenerated here): 5 clicks
//     + 2 boxes. Also pins the upstream behaviour the Lab reported: mark A (top-left) is in every mask.
//   • truck (1800×1200, natural): 2 clicks + 2 boxes; needs oracle/goldens/truck.png (run_oracle.py).
// Weights: $EDGETAM_WEIGHTS, else oracle/weights/edgetam_fp32.safetensors, else the shared-store fp16
// release; the test SKIPS when none is present (weights are never committed). CPU, fp32 compute.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import XCTest
@testable import EdgeTAM

final class NonSquareParityTests: XCTestCase {

    struct Golden: Decodable {
        struct Prompt: Decodable { let click: [Float]?; let box: [Float]?; let score: Float; let mask: [Int] }
        struct Img: Decodable { let width: Int; let height: Int; let prompts: [Prompt] }
        let images: [String: Img]
    }

    static let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()

    func weights() throws -> (path: String, fp16: Bool) {
        let candidates: [String?] = [ProcessInfo.processInfo.environment["EDGETAM_WEIGHTS"],
                          Self.root.appendingPathComponent("oracle/weights/edgetam_fp32.safetensors").path,
                          "/Volumes/Satechi/Models/models--mlx-community--EdgeTAM-fp16/model.safetensors"]
        guard let p = candidates.compactMap({ $0 }).first(where: { FileManager.default.fileExists(atPath: $0) })
        else { throw XCTSkip("no EdgeTAM weights (set EDGETAM_WEIGHTS)") }
        return (p, p.contains("fp16"))
    }

    func golden() throws -> Golden {
        let url = Self.root.appendingPathComponent("Tests/EdgeTAMParityTests/nonsquare_golden.json")
        return try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
    }

    /// The Lab's GateDocument.make: grey 128, five 12-px squares, sRGB.
    static func gatePage() -> CGImage {
        let marks: [((Int, Int), (UInt8, UInt8, UInt8))] = [((37, 41), (255, 0, 255)), ((1149, 58), (0, 255, 255)),
            ((601, 397), (255, 255, 0)), ((71, 757), (255, 0, 0)), ((1003, 731), (0, 0, 255))]
        let (w, h) = (1200, 800)
        var px = [UInt8](repeating: 128, count: w * h * 4)
        for i in stride(from: 3, to: px.count, by: 4) { px[i] = 255 }
        for ((x0, y0), c) in marks { for y in y0 ..< y0 + 12 { for x in x0 ..< x0 + 12 {
            let i = (y * w + x) * 4; px[i] = c.0; px[i + 1] = c.1; px[i + 2] = c.2 } } }
        let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)!
        return ctx.makeImage()!
    }

    static func decodeRLE(_ runs: [Int], count: Int) -> [Bool] {
        var out = [Bool](); out.reserveCapacity(count)
        for (i, r) in runs.enumerated() { out.append(contentsOf: repeatElement(i % 2 == 1, count: r)) }
        precondition(out.count == count, "RLE length \(out.count) ≠ \(count)")
        return out
    }

    /// Runs every golden prompt through the predictor; returns the Swift masks for extra checks.
    @discardableResult
    func check(_ name: String, _ cg: CGImage, file: StaticString = #filePath, line: UInt = #line) throws -> [[Bool]] {
        let (path, fp16) = try weights()
        let g = try XCTUnwrap(try golden().images[name])
        XCTAssertEqual([cg.width, cg.height], [g.width, g.height])
        // fp32 weights: exact port (measured Δscore ≤ 1e-4, mask IoU 1.0000 on every prompt). fp16 release
        // weights (fp32 compute): weight rounding alone moves a LOW-confidence click by up to 0.022 (truck
        // 1375,550: 0.4617 vs 0.4837, mask IoU 0.984); confident prompts stay ≤ 0.002 / ≥ 0.985.
        let (scoreTol, iouMin): (Float, Float) = fp16 ? (0.03, 0.97) : (2e-3, 0.995)
        var masks: [[Bool]] = []
        try Device.withDefaultDevice(.cpu) {
            let p = try EdgeTAMPredictor.fromPretrained(path, dtype: .float32)
            p.setImage(cg)
            for (i, gp) in g.prompts.enumerated() {
                let r = gp.click.map { p.predict(point: ($0[0], $0[1])) }
                    ?? p.predict(points: [], labels: [], box: gp.box!)          // .auto → single-mask for a box
                let swift = r.mask.asArray(Float.self).map { $0 > 0.5 }
                let gold = Self.decodeRLE(gp.mask, count: g.width * g.height)
                var inter = 0, uni = 0
                for k in 0 ..< swift.count where swift[k] || gold[k] { uni += 1; if swift[k] && gold[k] { inter += 1 } }
                let iou = uni == 0 ? 1 : Float(inter) / Float(uni)
                let what = gp.click.map { "click \($0)" } ?? "box \(gp.box!)"
                print(String(format: "[parity] %@ %@  score swift %.4f torch %.4f  mask IoU %.4f", name, what, r.score, gp.score, iou))
                XCTAssertEqual(r.score, gp.score, accuracy: scoreTol, "\(name) prompt \(i) \(what) score", file: file, line: line)
                XCTAssertGreaterThan(iou, iouMin, "\(name) prompt \(i) \(what) mask IoU", file: file, line: line)
                masks.append(swift)
            }
        }
        return masks
    }

    func testGatePageNonSquare() throws {
        let masks = try check("gate", Self.gatePage())
        // Upstream behaviour, pinned: every square's click mask also contains mark A (top-left, 37,41), which
        // PyTorch EdgeTAM reproduces (AB-T-0171) — a change here means the port drifted, not that it improved.
        for m in masks.prefix(5) {
            var a = 0
            for y in 41 ..< 53 { for x in 37 ..< 49 where m[y * 1200 + x] { a += 1 } }
            XCTAssertGreaterThan(a, 72, "mark A should be (mostly) inside every gate-click mask, as upstream")
        }
    }

    func testTruckNonSquare() throws {
        let url = Self.root.appendingPathComponent("oracle/goldens/truck.png")
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw XCTSkip("oracle/goldens/truck.png absent (run oracle/run_oracle.py)") }
        try check("truck", cg)
    }

    /// Antialiased resize = torch F.interpolate(antialias=True) on a known case (no weights needed).
    func testAntialiasWeightsRowsNormalised() {
        let w = EdgeTAMImage.kernelWeights(inSize: 1200, outSize: 1024, filter: .bilinear)
        let sums = w.sum(axis: 1).asArray(Float.self)
        XCTAssertEqual(sums.min()!, 1, accuracy: 1e-5); XCTAssertEqual(sums.max()!, 1, accuracy: 1e-5)
    }
}
