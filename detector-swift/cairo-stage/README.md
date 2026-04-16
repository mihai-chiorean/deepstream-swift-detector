# cairo-stage/

Pre-staged aarch64 shared libraries required by `nvdsosd`, the DeepStream
GStreamer element that draws detection bounding boxes on the MJPEG branch.
`nvdsosd` dlopen-s `libcairo.so.2` (via Pango) at pipeline init time; without
it the element fails to load and bbox overlay is silently absent.

## Why staged files instead of apt-get

`apt-get install libcairo2` inside a `docker buildx` cross-build runs under
qemu-binfmt aarch64 emulation.  On this host that took 12+ minutes with apt
still churning.  Copying pre-built `.so` files from the Linux_for_Tegra rootfs
takes seconds and produces byte-for-byte identical libraries.

## What is here

`libcairo.so.2` and its full transitive dependency closure, resolved via
`readelf -d ... | grep NEEDED` BFS from the L4T aarch64 rootfs.  Excluded
from staging: `libc`, `libm`, `libz`, `ld-linux`, and other glibc components
already present in the `ubuntu:24.04` base image.

The full set: libcairo, libpixman-1, libfontconfig, libfreetype, libpng16,
libxcb, libxcb-render, libxcb-shm, libX11, libXext, libXrender, libXau,
libXdmcp, libbsd, libexpat, libmd, libuuid, libbrotlidec, libbrotlicommon,
libgraphite2 — each as `libfoo.so.N -> libfoo.so.N.V.P` (soname symlink +
versioned file).

## Source

`~/jetson/Linux_for_Tegra/rootfs/usr/lib/aarch64-linux-gnu/`

These are the exact library versions shipped with the L4T r36.x rootfs
(JetPack 6.x).  The versions here must match the L4T version on the Jetson
host, since CDI-injected DeepStream libs are compiled against those same
system libs.

## Regenerating

Run from the repo root (or `detector-swift/`):

    detector-swift/scripts/stage-cairo.sh

The script accepts an optional rootfs path:

    detector-swift/scripts/stage-cairo.sh /path/to/Linux_for_Tegra/rootfs

Re-run when:
- The Linux_for_Tegra installation is updated (L4T version bump).
- A new DeepStream plugin is added that dlopen-s a library not already in
  this set.  Add the new root soname to `SEEDS` in the script and re-run.
- Container boot shows `error while loading shared libraries: libfoo.so.N`.

## Git tracking

These files ARE tracked in git.  The goal is a fully reproducible build:
`docker buildx build` must produce an identical image from a clean checkout
without requiring the host to have `~/jetson/Linux_for_Tegra/` present.
The `.so` files are binary blobs (~3.5 MB total) but that is an acceptable
trade-off for build-time determinism.
