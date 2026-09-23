// SAM21PackageTests.swift — the SAM 2.1-S promptSegment package through the engine's offline gates (AB-T-0173).
// No MLX kernels run: MAT (weight sourcing + store precedence), CAN (entry checkpoint + declared cadence), and
// the manifest (licence, surface, split footprint).

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXEdgeTAM

final class SAM21PackageTests: XCTestCase {

    func testMATGate() {
        let report = MaterializationConformance.check(freshConfiguration: SAM21Configuration())
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWeightSourceDeclared() {
        let s = SAM21Configuration().weightSources
        XCTAssertEqual(s.map(\.repo), ["mlx-community/SAM2.1-hiera-small-fp16"])
        XCTAssertEqual(s.first?.matching, ["model.safetensors"])
    }

    func testFlatStoreLayoutWins() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("sam21-store-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = SAM21Configuration(modelsRootDirectory: root)
        XCTAssertNil(SAM21Package.storedWeights(cfg))
        let flat = try XCTUnwrap(ModelStore(root: root).directory(for: cfg.repo))
        try FileManager.default.createDirectory(at: flat, withIntermediateDirectories: true)
        try Data([1]).write(to: flat.appending(path: "model.safetensors"))
        XCTAssertEqual(SAM21Package.storedWeights(cfg)?.standardizedFileURL.path,
                       flat.appending(path: "model.safetensors").standardizedFileURL.path)
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
    }

    func testCANGatePreCancelledRun() async {
        let report = await CancellationConformance.checkRun(
            package: SAM21Package(configuration: SAM21Configuration()),
            request: PromptSegmentRequest(image: Image(format: .png, data: Data()), points: [[1, 1]], pointLabels: [1]))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        // Single forward per request: entry checkpoint + post-forward/pre-encode checkpoint (PromptSegmentSession).
        let report = CancellationConformance.checkCadence(
            manifest: SAM21Package.manifest, posture: .cadence([.init(phase: .postprocess, unit: .frame)]))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testManifest() {
        let m = SAM21Package.manifest
        XCTAssertEqual(m.license.weightLicense, .apache2)
        XCTAssertEqual(m.surfaces.map(\.capability), [.promptSegment])
        XCTAssertEqual(m.surfaces.first?.supportedModes, [SAM21Package.softMatte])
        XCTAssertEqual(SAM21Package.softMatte, EdgeTAMPackage.softMatte)
        let f = try? XCTUnwrap(m.requirements.footprints.first)
        XCTAssertEqual(f?.quant, .fp16)
        XCTAssertGreaterThan(f?.peakActivationBytes ?? 0, f?.residentBytes ?? 0)   // split, activation-dominated
    }
}
