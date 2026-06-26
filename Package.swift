// swift-tools-version: 6.2
// mlx-edgetam-swift — EdgeTAM (on-device SAM 2) promptable segmentation on Swift/MLX. Image-mode first
// (RepViT-M1 encoder + FPN + SAM prompt encoder + mask decoder); video memory is Phase 2. From-scratch
// architecture port of facebookresearch/EdgeTAM (Apache-2.0). See SCOPING.md.
import PackageDescription

let mlxCore: [Target.Dependency] = [
    .product(name: "MLX", package: "mlx-swift"),
    .product(name: "MLXNN", package: "mlx-swift"),
    .product(name: "MLXFast", package: "mlx-swift"),
]

let package = Package(
    name: "mlx-edgetam-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "EdgeTAM", targets: ["EdgeTAM"]),                // MLX core (mlx-swift only)
        .library(name: "MLXEdgeTAM", targets: ["MLXEdgeTAM"]),          // engine-consumable ModelPackage
        .executable(name: "edgetam-smoke", targets: ["EdgeTAMSmoke"]),
        .executable(name: "edgetam-video-smoke", targets: ["EdgeTAMVideoSmoke"]),
        .executable(name: "edgetam-package-smoke", targets: ["EdgeTAMPackageSmoke"]),
        .executable(name: "edgetam-video-package-smoke", targets: ["EdgeTAMVideoPackageSmoke"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.3"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // MLXToolKit contract (promptSegment 1.10.0 + trackObject 1.11.0). Released as engine 0.11.0.
        .package(url: "https://github.com/xocialize/mlx-engine-swift.git", from: "0.11.0"),
        // FFmpeg-free native video decode (Video bytes → frames) for the trackObject runtime surface.
        // Stable version pin (v0.2.0 = the decode(input:) frames-in seam at 27e767f) so EdgeTAM's whole
        // graph is version-based and consumable by tag — a revision sub-dep breaks version consumers (SwiftPM).
        .package(url: "https://github.com/xocialize/frame-stream-native.git", from: "0.2.0"),
    ],
    targets: [
        .target(name: "EdgeTAM", dependencies: mlxCore, path: "Sources/EdgeTAM",
                swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(
            name: "MLXEdgeTAM",
            dependencies: [
                "EdgeTAM",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "FrameStreamNative", package: "frame-stream-native"),
            ],
            path: "Sources/MLXEdgeTAM"),
        .executableTarget(
            name: "EdgeTAMSmoke",
            dependencies: ["EdgeTAM", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/EdgeTAMSmoke", swiftSettings: [.swiftLanguageMode(.v5)]),
        .executableTarget(
            name: "EdgeTAMVideoSmoke",
            dependencies: ["EdgeTAM", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/EdgeTAMVideoSmoke", swiftSettings: [.swiftLanguageMode(.v5)]),
        .executableTarget(
            name: "EdgeTAMPackageSmoke",
            dependencies: ["MLXEdgeTAM",
                           .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                           .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/EdgeTAMPackageSmoke"),
        .executableTarget(
            name: "EdgeTAMVideoPackageSmoke",
            dependencies: ["MLXEdgeTAM", "EdgeTAM",
                           .product(name: "MLX", package: "mlx-swift"),
                           .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                           .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/EdgeTAMVideoPackageSmoke"),
    ]
)
