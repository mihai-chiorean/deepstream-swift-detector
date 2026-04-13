// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "DeepStreamDetectorSwift",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/wendylabsinc/tensorrt-swift", from: "0.0.1"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.6.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
        .package(url: "https://github.com/apple/swift-log", from: "1.6.0"),
        .package(url: "https://github.com/swift-server/async-http-client", from: "1.23.0"),
        .package(url: "https://github.com/apple/swift-container-plugin.git", from: "1.3.0"),
    ],
    targets: [
        // System library targets without pkgConfig: so SwiftPM doesn't attempt
        // host pkg-config resolution during cross-compilation.  Headers are
        // found via the -isystem flags baked into the wendyos SDK toolset.json;
        // linker flags come from the `link` directives in each module.modulemap.
        .systemLibrary(
            name: "CFFmpeg",
            providers: [
                .apt(["libavformat-dev", "libavcodec-dev", "libavutil-dev", "libswscale-dev"]),
            ]
        ),
        .systemLibrary(
            name: "CTurboJPEG",
            providers: [
                .apt(["libturbojpeg0-dev"]),
            ]
        ),
        .systemLibrary(
            name: "CGStreamer",
            providers: [
                .apt(["libgstreamer1.0-dev", "libgstreamer-plugins-base1.0-dev"]),
            ]
        ),
        .executableTarget(
            name: "Detector",
            dependencies: [
                .product(name: "TensorRT", package: "tensorrt-swift"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "AsyncHTTPClient", package: "async-http-client"),
                "CFFmpeg",
                "CTurboJPEG",
                "CGStreamer",
            ]
        ),
    ]
)
