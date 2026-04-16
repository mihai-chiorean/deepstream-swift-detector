// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "DeepStreamDetectorSwift",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.6.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
        .package(url: "https://github.com/apple/swift-log", from: "1.6.0"),
        .package(url: "https://github.com/swift-server/async-http-client", from: "1.23.0"),
        .package(url: "https://github.com/apple/swift-container-plugin.git", from: "1.3.0"),
    ],
    targets: [
        // CGStreamer — compiled C target (not a systemLibrary) so that
        // nvds_shim.c can include DeepStream headers and produce object code
        // linked into the Detector executable.
        //
        // Headers: The WendyOS SDK (6.2.3-RELEASE_wendyos_aarch64) ships
        // nvdsmeta.h and gstnvdsmeta.h in the sysroot at:
        //   wendyos-device.sdk/usr/include/
        // The toolset.json injects -isystem paths for gstreamer-1.0 and
        // glib-2.0; the sysroot root covers usr/include directly. No -I
        // overrides needed in Package.swift.
        //
        // Link stubs: libnvdsgst_meta and libnvds_meta are NOT in the SDK
        // sysroot (they live in /opt/nvidia/deepstream/deepstream-7.1/lib/
        // on the Jetson, bind-mounted by CDI at runtime). We declare them so
        // the binary carries the correct DT_NEEDED entries; --allow-shlib-
        // undefined (lld flag) prevents a link-time error since the stubs are
        // absent at cross-compile time.
        .target(
            name: "CGStreamer",
            exclude: ["vendor/README.md"],
            publicHeadersPath: ".",
            cSettings: [
                // DeepStream SDK headers vendored under Sources/CGStreamer/vendor/.
                // Sourced from Linux_for_Tegra/.../deepstream-7.1/sources/includes/.
                // The WendyOS SDK has a partial set (nvdsmeta.h, gstnvdsmeta.h)
                // but missing transitive headers (nvll_osd_struct.h etc.). The
                // partial SDK versions conflict with ours — use -isystem on the
                // vendor dir to take precedence over the SDK's -isystem.
                .unsafeFlags(["-isystem", "Sources/CGStreamer/vendor"]),
            ],
            linkerSettings: [
                .linkedLibrary("gstreamer-1.0"),
                .linkedLibrary("gstapp-1.0"),
                .linkedLibrary("gobject-2.0"),
                .linkedLibrary("glib-2.0"),
                .linkedLibrary("nvdsgst_meta"),
                .linkedLibrary("nvds_meta"),
                // lld: allow DS libs to be absent from the sysroot at
                // cross-compile time; they are CDI-injected at runtime.
                .unsafeFlags(["-Xlinker", "--allow-shlib-undefined"]),
            ]
        ),

        .executableTarget(
            name: "Detector",
            dependencies: [
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "AsyncHTTPClient", package: "async-http-client"),
                "CGStreamer",
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6),
            ]
        ),
    ]
)
