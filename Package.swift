// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "sebbu-fft",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
        .watchOS(.v26),
        .tvOS(.v26)
    ],
    products: [
        .library(
            name: "SebbuFFT",
            targets: ["SebbuFFT"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", from: "1.1.1")
    ],
    targets: [
        .target(
            name: "SebbuFFT",
            dependencies: [
                .product(name: "Numerics", package: "swift-numerics")
            ]
        ),
        .testTarget(
            name: "SebbuFFTTests",
            dependencies: ["SebbuFFT"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
