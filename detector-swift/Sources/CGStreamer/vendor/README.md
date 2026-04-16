# CGStreamer vendor/ — no longer used

DeepStream SDK headers (`nvdsmeta.h`, `gstnvdsmeta.h`, and related) are
provided directly by the WendyOS SDK sysroot at build time:

    wendyos-device.sdk/usr/include/nvdsmeta.h
    wendyos-device.sdk/usr/include/gstnvdsmeta.h

The SDK is `6.2.3-RELEASE_wendyos_aarch64` installed at:

    ~/.swiftpm/swift-sdks/6.2.3-RELEASE_wendyos_aarch64.artifactbundle/

`toolset.json` injects `-isystem` for `usr/include/gstreamer-1.0`,
`usr/include/glib-2.0`, and `usr/lib/glib-2.0/include`; the sysroot root
covers `usr/include` for the flat DS headers. No `-I` flags in Package.swift
are needed.

The `.gitkeep` file keeps this directory tracked so the path exists in the
working tree. Do not add headers here.
