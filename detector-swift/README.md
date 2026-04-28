# detector-swift

The Swift detector. Drives a DeepStream pipeline on a Jetson Orin Nano 8 GB,
serves detections over WebSocket and Prometheus metrics over HTTP.

The pipeline is `rtspsrc → rtph264depay → h264parse → nvv4l2decoder → nvstreammux → nvinfer → nvtracker → fakesink`.
Swift owns the orchestration (process, threading, HTTP/WS surface, reconnect
loop, broadcaster). NVIDIA owns everything inside `nvv4l2decoder` →
`nvtracker` (NVDEC, TensorRT, NvDCF). Swift never touches an NVMM surface —
the pad probe on `nvtracker.src` reads `NvDsBatchMeta` via a small C shim
(`Sources/CGStreamer/nvds_shim.c`) that flattens detection objects into POD
structs.

The blog post — [Swift doing NVIDIA's job](https://mihaichiorean.com/blog/swift-doing-nvidias-job/) —
covers the why and the head-to-head against the Python sibling at `../detector/`.

## Pre-reqs

- Jetson Orin Nano 8 GB.
- JetPack r36.4.4 with DeepStream 7.1 (the version that ships with L4T r36.4.4).
- WendyOS or an equivalent JetPack-based distribution. WendyOS 0.13.1 has a
  kernel regression that breaks NVDEC — pin to 0.10.5 or wait for the fix
  (regression doc lives in the upstream `wendylabsinc/samples` repo as
  `docs/wendyos-0.13.1-nvdec-regression.md`).
- A 1080p RTSP camera. Configure it in `streams.json`.
- Cross-compile host: x86_64 Linux with `swiftly` and Swift 6.2.3 installed.

## Build

Cross-compile the Swift binary against the WendyOS device SDK:

```bash
cd detector-swift
swift build --swift-sdk 6.2.3-RELEASE_wendyos-device_aarch64 -c release --product Detector
```

Pre-build the TensorRT engine on the device (the first start without a cached
plan takes 1-3 min while nvinfer builds it from `yolo26n.onnx`):

```bash
ssh root@<device> 'trtexec --onnx=/app/yolo26n.onnx \
    --saveEngine=/app/yolo26n_b2_fp16.engine \
    --fp16 --shapes=images:1x3x640x640'
```

`nvinfer_config.txt` references `model-engine-file=/app/yolo26n_b2_fp16.engine`
with `onnx-file=/app/yolo26n.onnx` as the fallback path. If the engine isn't
present, nvinfer will build it from the ONNX on first start; subsequent starts
load the cached plan.

Build the container image:

```bash
docker buildx build --platform linux/arm64 -t detector-swift:latest .
```

## Deploy

`wendy.json`'s `baseImage` defaults to a placeholder. Set it to a registry the
Wendy CLI can pull from — your local registry (`<host>:5000/swift-detector-base:latest`)
or any aarch64 Ubuntu base.

From `detector-swift/`:

```bash
WENDY_AGENT=<device-ip> wendy run -y --detach
```

Then apply the cgroup cap (mandatory — `oom_score_adj` defaults to -998
otherwise, which puts networking on the OOM kill list before the detector):

```bash
ssh root@<device> 'CONTAINER=detector-swift /usr/local/bin/detector-cap'
```

Note: `start.sh` at the repo root deploys the **Python** sibling at
`../detector/`, not this Swift detector. That's intentional given the
samples-clone framing of the top-level README. To deploy the Swift detector,
use the `wendy run` command above from this directory.

## Smoke test

Metrics — should return ~20 (FPS):

```bash
curl -s http://<device>:9090/metrics | grep '^deepstream_fps '
```

Detection stream:

```bash
websocat ws://<device>:9090/detections
```

Browser monitor: run `monitor_proxy.py` from the repo root on a LAN host
that can reach both the device and your browser. The proxy serves
`monitor.html` with overlays driven by the WebSocket and video pulled via
WebRTC from the mediamtx relay.

## Troubleshooting

**"Cannot allocate buffers" or `avdec_h264` fallback in the GStreamer logs.**
The CDI spec is missing `LIBV4L2_PLUGIN_DIR` — `nvidia-ctk` drops env-type
entries from CSV input. The fix is a post-processing script that patches the
generated YAML with `sed` after `nvidia-ctk` runs. See `ARCHITECTURE.md`
"OS plumbing" section.

**Tracker IDs flicker or switch between frames on parked objects.**
NvDCF is the default tracker variant (set in `Dockerfile` ENV). Two other
variants are available: NvSORT and NvDeepSORT. NvDeepSORT requires a
pre-built TRT engine for the appearance embedder (TAO model decryption is a
separate step and not done by the container). Edit the `Dockerfile` ENV and
swap `tracker_config.yml` to switch.

**First container start takes 1-3 min.** That's nvinfer building the TRT
engine from `yolo26n.onnx` — expected with the `onnx-file=` fallback. The
plan is cached on the device after first build; subsequent starts skip
straight to PLAYING. Pre-build with `trtexec` (Build section above) to skip
the first-start wait.

## Where the docs are

- `detector-swift/ARCHITECTURE.md` — pipeline shape, NvDsBatchMeta walk,
  CDI bind-mounts, the war stories.
- `detector-swift/PORT_PLAN.md` — staged port from Python; Stage 1 → Stage 2
  decision trail. (Note: this file is only present in working trees that
  predate the public extract; the public repo has it gitignored. Read
  ARCHITECTURE.md for the same material.)
- `../OPERATIONS.md` — topology, port table, device access, deploy recipes.
- `../docs/benchmark-python-vs-swift.md` — head-to-head Run 1 / Run 2 / Run 3.
- Blog: [Swift doing NVIDIA's job](https://mihaichiorean.com/blog/swift-doing-nvidias-job/).

## Known issues

- **Pad-probe lifecycle UAF risk.** `teardownPipeline()` calls
  `gst_pad_remove_probe` then `Unmanaged.release()` on the retained
  `DetectionStream` box. `gst_pad_remove_probe` does not block in-flight
  probe callbacks, so a release that races with a streaming-thread callback
  is a use-after-free in theory. Has not been observed in practice (the
  detector goes through reconnect cycles on camera flap and stays up). Fix
  candidates: two-phase teardown with an active-callback drain, or set the
  pipeline to NULL state before remove+release. Not shipped because the
  reorder hasn't been A/B tested at runtime and a wrong fix here would be
  worse than the latent bug. See `REVIEW-GPU.md` Finding 9 for the full
  analysis.
