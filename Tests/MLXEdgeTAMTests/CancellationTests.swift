// CancellationTests.swift — EdgeTAM through the engine's CAN gate (offline, no MLX kernels).
// CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before dispatch,
// prompts validation, or weights); CAN-3 is the document of record for the checkpoint cadence.
// EdgeTAM is NOT long-run implied (promptSegment/trackObject aren't long-run capabilities and
// peakActivationBytes 0.5 GB < 2 GB), so the sub-second exemption would technically pass — but
// trackObject is a real per-frame streaming loop over arbitrarily long clips, so the HONEST
// posture is the cadence the code actually has, declared instead of the exemption:
//   • trackObject: `try Task.checkCancellation()` once per propagated frame, inside the emit
//     closure of `EdgeTAMPackage.runTrack` (the closure fires after each frame's mask lands;
//     the CancellationError rethrows unchanged through the core's throwing `track`).
//   • promptSegment: single forward — entry checkpoint + post-forward/pre-PNG-encode
//     checkpoint in `EdgeTAMPackage.runSegment`.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXEdgeTAM

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRunSegment() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // dispatch or weights are touched, so this is offline-safe.
        let package = EdgeTAMPackage(configuration: EdgeTAMConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: PromptSegmentRequest(image: Image(format: .png, data: Data()),
                                          points: [[1, 1]], pointLabels: [1]))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANGatePreCancelledRunTrack() async {
        // Same entry checkpoint guards the trackObject dispatch arm.
        let package = EdgeTAMPackage(configuration: EdgeTAMConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: TrackObjectRequest(video: Video(format: .mp4, data: Data()),
                                        promptFrame: 0, points: [[1, 1]], pointLabels: [1]))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // Short-run envelope by the letter of the gate — but declare the real cadence anyway:
        // the trackObject per-frame loop is a genuine long-run surface on long clips.
        XCTAssertFalse(CancellationConformance.longRunImplied(by: EdgeTAMPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: EdgeTAMPackage.manifest,
            posture: .cadence([
                // trackObject: per propagated frame, in runTrack's emit closure (fires after
                // each frame's mask + memory-bank entry evals; streaming, flat working set).
                .init(phase: .generate, unit: .frame),
                // promptSegment: single forward; post-forward/pre-encode checkpoint per image.
                .init(phase: .postprocess, unit: .frame),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
