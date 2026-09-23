// MaterializationTests.swift — EdgeTAM weights live where the engine's store puts them (offline, no MLX).
// AB-T-0171: ≤ v0.4.x declared no WeightSourcing, so an engine ≥ 1.24 wrote only the install marker into
// `<root>/models--mlx-community--EdgeTAM-fp16/` while the package's own download landed in
// `<root>/models/mlx-community/EdgeTAM-fp16/` — a station could hold the marker with no weights beside it.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXEdgeTAM

final class MaterializationTests: XCTestCase {

    func testMATGate() {
        let report = MaterializationConformance.check(freshConfiguration: EdgeTAMConfiguration())
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWeightSourceDeclared() {
        let s = EdgeTAMConfiguration().weightSources
        XCTAssertEqual(s.count, 1)
        XCTAssertEqual(s[0].repo, "mlx-community/EdgeTAM-fp16")
        XCTAssertEqual(s[0].matching, ["model.safetensors"])
    }

    func testExplicitWeightsURLSuppressesMaterialization() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("edgetam-\(UUID()).safetensors")
        try Data([0]).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        XCTAssertTrue(EdgeTAMConfiguration(weightsURL: tmp).missingWeightSources(storeRoot: nil).isEmpty)
    }

    /// Flat (engine-materialized) wins; the v0.4.x swift-transformers layout is still found (no re-download);
    /// an empty store reads as missing to the engine's probe.
    func testStoreResolutionPrecedence() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("edgetam-store-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let fm = FileManager.default
        let cfg = EdgeTAMConfiguration(modelsRootDirectory: root)
        XCTAssertNil(EdgeTAMPackage.storedWeights(cfg))
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1)

        let legacy = root.appending(path: "models/mlx-community/EdgeTAM-fp16")
        try fm.createDirectory(at: legacy, withIntermediateDirectories: true)
        try Data([1]).write(to: legacy.appending(path: "model.safetensors"))
        XCTAssertEqual(EdgeTAMPackage.storedWeights(cfg)?.standardizedFileURL.path,
                       legacy.appending(path: "model.safetensors").standardizedFileURL.path)

        let flat = try XCTUnwrap(ModelStore(root: root).directory(for: cfg.repo))
        try fm.createDirectory(at: flat, withIntermediateDirectories: true)
        try Data([2]).write(to: flat.appending(path: "model.safetensors"))
        XCTAssertEqual(EdgeTAMPackage.storedWeights(cfg)?.standardizedFileURL.path,
                       flat.appending(path: "model.safetensors").standardizedFileURL.path)
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty, "flat layout satisfies the engine probe")
    }

    func testSoftMatteModeAdvertised() {
        let d = EdgeTAMPackage.manifest.surfaces.first { $0.capability == .promptSegment }
        XCTAssertEqual(d?.supportedModes, [EdgeTAMPackage.softMatte])
    }
}
