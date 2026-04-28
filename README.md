# DeepStream Swift Detector

A Swift port of an NVIDIA DeepStream object-detection pipeline, running on a Jetson Orin Nano 8 GB. Swift orchestrates a GStreamer graph — `rtspsrc → nvv4l2decoder → nvstreammux → nvinfer → nvtracker → fakesink` — reads inference metadata off a pad probe, and serves it as a typed `AsyncStream<DetectionFrame>` into an HTTP + WebSocket surface. Pixels stay on the GPU; Swift never touches an NVMM surface.

> **Blog post with the full story:** [*The hard part of Swift on Jetson was not Swift*](https://mihaichiorean.com/blog/hard-part-of-swift-on-jetson/) — what the language layer needed from the platform underneath it (CDI spec, Swift SDK sysroot), with measurements vs the Python sibling.
>
> **Python sibling + upstream context:** the Python version of this detector, plus broader WendyOS sample projects, live in [`wendylabsinc/samples`](https://github.com/wendylabsinc/samples) under `samples/deepstream-vision/`. This repo is a standalone extract focused on the Swift port, with full commit history preserved via `git subtree split`.

---

## What this actually is

- **`detector-swift/`** — the Swift detector. ~150 lines of Swift orchestrating GStreamer, plus a small C shim in `Sources/CGStreamer/nvds_shim.c` that walks DeepStream's `NvDsBatchMeta` linked lists and flattens them into POD structs Swift consumes.
- **`detector/`** — the Python sibling, kept for head-to-head comparison. Same TensorRT engine, same custom bbox parser, same tracker — different host process.
- **`monitor.html`** + **`monitor_proxy.py`** — a browser monitor that pulls video from `mediamtx` over WebRTC and overlays bounding boxes from the detector's WebSocket stream, synchronized via `requestVideoFrameCallback`'s `metadata.rtpTimestamp`.
- **`docs/`** — technical reference material: benchmarks (Run 1–3 head-to-head), CPU scaling analysis, tracker and transport research, mediamtx RTSP relay config, MJPEG-decoupling notes.
- **`gpu-stats/`**, **`vlm/`** — sidecar components (tegrastats exporter; Qwen3-VL inference server). Not required to run the detector; see upstream `wendylabsinc/samples` for their full context.

---

## Test environment

These caveats matter if you want to reproduce the results:

- **Hardware:** Jetson Orin Nano 8 GB developer kit.
- **JetPack:** **r36.4.4**. The DeepStream + GStreamer plugin chain is sensitive to JetPack minor versions; behavior on newer releases is not verified by this repo.
- **DeepStream:** 7.1 (the version that ships with JetPack r36.4.4 / L4T r36.4.4).
- **Host OS on the device:** [WendyOS](https://github.com/wendylabsinc/wendyos), a Yocto-based edge Linux distribution. WendyOS is in active development — its image builds, runtime libs, and CDI specs change between releases. If you're running on a different build than ours, expect to re-do some of the plumbing described below.
- **Inference model:** YOLO26n FP16 compiled to a TensorRT engine with a custom bbox parser (`libnvdsparsebbox_yolo26.so`). See `detector-swift/Dockerfile` for how the parser .so is built against DeepStream 7.1 headers.

### OS plumbing that wasn't turnkey

Getting Swift to drive DeepStream 7.1 on JetPack r36.4.4 required two pieces of plumbing on the OS side beyond what was turnkey in the original WendyOS build we started from:

- **CDI spec extensions.** Containerd on WendyOS uses a [CDI (Container Device Interface)](https://github.com/cncf-tags/container-device-interface) spec to bind-mount NVIDIA runtime libraries into containers at task start. The stock CDI spec did not bind-mount everything DeepStream 7.1 needs (`libnvv4l2.so`, `libv4l2_nvvideocodec.so`, `libtegrav4l2.so`, the full gst-plugins-bad chain, and several device nodes for NVDEC). We extended the CDI spec — and added a post-processing script to patch environment variables like `LIBV4L2_PLUGIN_DIR` that `nvidia-ctk` silently drops from CSV input — so that a minimal `ubuntu:24.04`-based container can inherit a working NVDEC + nvinfer + NvDCF pipeline at runtime. See `detector-swift/ARCHITECTURE.md` for the concrete list of bind-mounts.
- **A more complete Swift cross-compile SDK.** WendyOS ships a Swift SDK for aarch64, but DeepStream's C headers (`nvdsmeta.h`, `nvll_osd_struct.h`, etc.) and linker stubs for the DeepStream libs were missing. We extended the Yocto `deepstream-7.1` recipe to copy those headers into the SDK's `/usr/include/` and put linker-only stub `.so` files at `/usr/lib/`, then rebuilt the SDK image. The recipe change itself lives upstream in [`wendylabsinc/meta-wendyos-jetson`](https://github.com/wendylabsinc/meta-wendyos-jetson).

The point: if you're coming from a vanilla JetPack install or a different container runtime, the pipeline itself is standard DeepStream 7.1 — but expect to do equivalent plumbing to make everything reachable from inside your container and from a Swift cross-compile toolchain.

---

## Running it

### Prerequisites

- Jetson Orin Nano (or similar) running JetPack r36.4.4 with DeepStream 7.1 installed.
- A 1080p RTSP camera. The detector is currently configured for a single stream; multi-stream support is an open item (see the blog post's footer).
- Swift 6.2+ toolchain for cross-compilation on your build host (x86_64 Linux recommended).

### Build

The Dockerfile in `detector-swift/` builds an aarch64 container image from a pre-compiled Swift binary:

```bash
cd detector-swift
source ~/.local/share/swiftly/env.sh     # or however you have Swift on PATH
swift build --swift-sdk <your-aarch64-sdk> -c release
docker buildx build --platform linux/arm64 -t detector-swift:latest .
```

### Deploy

The `wendy.json` descriptors in `detector-swift/`, `detector/`, `gpu-stats/`, and `vlm/` are for the Wendy CLI (the deploy tool that ships with WendyOS). A typical deploy is:

```bash
WENDY_AGENT=<device-ip> wendy run -y --detach
```

If you're not on WendyOS, the container images are standard OCI and can be run via `ctr`, `nerdctl`, or `docker` directly — you'll need to supply the CDI spec (or mount the NVIDIA libs manually) and set `oom_score_adj` / memory cgroup caps by hand. See `OPERATIONS.md` for the adaptation notes.

### Configure

Edit `detector-swift/streams.json`:

```json
{
  "streams": [
    {
      "name": "camera1",
      "url": "rtsp://<your-camera>/stream",
      "enabled": true
    }
  ]
}
```

If you want to use the `mediamtx` RTSP relay (recommended for single-session cameras), point the detector at `rtsp://<host>:8554/<path>` and configure mediamtx per `docs/rtsp-relay.md`.

### Observe

- **Detector metrics** (Prometheus): `http://<device>:9090/metrics` — `deepstream_fps`, `deepstream_reconnects_total`, `deepstream_frames_processed_total`, latency histograms.
- **Detection stream** (WebSocket): `ws://<device>:9090/detections`.
- **Browser monitor:** `monitor_proxy.py` serves `monitor.html` with a same-origin proxy to the detector and the mediamtx WebRTC endpoint. Run it from a LAN host that can reach both the device and the browser.

---

## What's documented in `docs/`

The blog post covers the *what* and *why*. The `docs/` directory covers the *how* and the supporting measurements:

- **`docs/benchmark-python-vs-swift.md`** — head-to-head Run 1 / Run 2 / Run 3 report. Run 2 is SUPERSEDED; the Caveats section is required reading before citing any number.
- **`docs/cpu-scaling-research.md`** — where CPU goes element-by-element, plus the unmeasured multi-stream experiment design (GIL-bound pyds probe vs Swift C-shim).
- **`docs/tracker-and-transport-research.md`** — NvDCF tuning knobs, class filter location (it's on `nvinfer`, not `nvtracker`, in DS 7.1), NvSORT comparison, and the SEI-vs-DataChannel-vs-WebSocket-rtpTimestamp analysis.
- **`docs/mjpeg-contention-measurement.md`**, **`docs/mjpeg-decoupling-design.md`** — the Stage 1 → Stage 2 decision trail.
- **`docs/rtsp-relay.md`** — mediamtx as RTSP relay. Necessary if your camera is single-session (most prosumer IP cameras are).
- **`docs/benchmark-data/`** — raw CSVs from the measurement runs.
- **`docs/dockerfile-nvparser-stage-example.txt`** — fragment for building `libnvdsparsebbox_yolo26.so` against DS 7.1.

---

## Status + caveats

- **Current operational baseline:** ~21 FPS end-to-end, 26.6 % of one CPU core, stable RSS, `nvmap` flat at 421,520 kB, 26 h uninterrupted at the time of the blog post.
- **Not verified on:** JetPack releases newer than r36.4.4; GPUs other than Orin Nano 8 GB; WendyOS builds other than the one this work was done on.
- **Known to break on:** camera WiFi links with >500 ms RTT (the RTSP pull drops EOS and the detector recovers via its bus-watch reconnect loop; this was a real bug we fixed mid-flight).
- **Open items** (see the blog post footer): multi-stream scaling curve, NvSORT tracker swap, a proper stress-harness reliability audit.

## License + attribution

Originally developed inside [`wendylabsinc/samples`](https://github.com/wendylabsinc/samples). This standalone extract preserves the full commit history via `git subtree split`. The `pre-cleanup-20260420` tag on this repo points to the last commit before author-private docs were removed from the history; useful only as a recovery anchor, not as a reference for reading the code.
