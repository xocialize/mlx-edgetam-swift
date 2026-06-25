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
        .executable(name: "edgetam-smoke", targets: ["Smoke"]),
        .executable(name: "edgetam-package-smoke", targets: ["PackageSmoke"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.3"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // MLXToolKit contract (promptSegment @ 1.10.0). Pinned to the revision that introduced it.
        .package(url: "https://github.com/xocialize/mlx-engine-swift.git", revision: "6aff0f4"),
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
            ],
            path: "Sources/MLXEdgeTAM"),
        .executableTarget(
            name: "Smoke",
            dependencies: ["EdgeTAM", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/Smoke", swiftSettings: [.swiftLanguageMode(.v5)]),
        .executableTarget(
            name: "PackageSmoke",
            dependencies: ["MLXEdgeTAM",
                           .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                           .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/PackageSmoke"),
    ]
)
