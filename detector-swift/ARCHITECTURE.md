# Swift on the Edge: YOLO26 + Qwen3-VL on a Jetson Orin Nano

*DeepStream-native Swift detector: nvinfer + nvtracker in-pipeline, hardware NVDEC decode, zero NVMM buffer mapping. Stage 2 (2026-04-15): MJPEG branch removed; video served by mediamtx WebRTC.*

---

## The Pitch

We replaced a **1,438-line Python detector** (built on NVIDIA's DeepStream SDK + GStreamer + PyGObject + NumPy + OpenCV + Flask) with a **Swift 6.2 implementation** that:

- Runs **YOLO26n** (NMS-free end-to-one detector) through DeepStream's `nvinfer` plugin
- Tracks objects with **NvDCF** (`nvtracker`, `libnvds_nvmultiobjecttracker.so`)
- Decodes H.264 via **NVDEC hardware** (`nvv4l2decoder`) — never touches CPU-side pixels
- Publishes detections via a **WebSocket `/detections` endpoint** — JSON per frame with `ptsNs` for PTS-accurate overlay sync
- Serves video via **mediamtx WebRTC** — the detector pipeline terminates at `fakesink`, no JPEG encoding
- Sends best-frame crops to a **Qwen3-VL-2B** sidecar via llama.cpp for natural-language descriptions (currently disabled, infrastructure in place)
- **Cross-compiles natively** from x86 to aarch64 using the `6.2.3-RELEASE_wendyos-device_aarch64` Swift SDK

Numbers on a Jetson Orin Nano 8GB (release build, Stage 2, 2026-04-15):

| Metric | Value |
|---|---|
| End-to-end FPS | **~20** (camera-limited, 1080p RTSP) |
| CPU usage | **~55%** |
| NVMM memory growth | **0 MB/s** (was 45–80 MB/s with the old appsink+map design) |
| nvmap iovmm steady-state | **~206 MB** (Stage 2 baseline; was ~224 MB pre-Stage-2 with MJPEG branch) |
| Preprocess latency (steady) | **~26.5 ms** (nvstreammux + nvinfer) |
| Inference latency (steady) | **~22.8 ms** (nvinfer only) |
| Build time (cross-compile) | **~0.7 s** incremental, ~14 s cold |

---

## What It Replaces

The Python original (`detector/detector.py`) depends on:

- **DeepStream 7.1 SDK** — GStreamer pipelines, `pyds` bindings, `nvinfer` plugin
- **GObject Introspection** (`gi`) — Python-to-GStreamer bridge
- **NumPy + OpenCV** — preprocessing, rendering, JPEG encoding
- **Flask** — HTTP server for MJPEG stream + metrics
- **prometheus-client** — metrics library

That's a ~2 GB container image with the full DeepStream runtime, GStreamer plugin graph, and Python interpreter.

---

## What We Built

| File | Responsibility |
|------|----------------|
| `Detector.swift` | Entry point, CLI args, per-stream detection loop |
| `GStreamerFrameReader.swift` | DeepStream pipeline actor; owns pure detection pipeline, installs pad probe on nvtracker src |
| `HTTPServer.swift` | Hummingbird 2 HTTP server: /detections (WebSocket), /stream (301→WHEP), /metrics, /health |
| `Metrics.swift` | Prometheus registry backed by `Mutex<RegistryState>` (not actor — too hot for async hops) |
| `DetectionBroadcaster.swift` | Fans out per-frame detection JSON to N WebSocket clients |
| `TrackFinalizer.swift` | Queues finalized tracks, crops best frame, submits to VLM (currently disabled) |
| `VLMClient.swift` | async HTTP client for llama.cpp sidecar (OpenAI format) |
| `CGStreamer/nvds_shim.c` | C shim: reads `NvDsBatchMeta` from a GstBuffer via a pad probe, flattens into `WendyDetection[]` |
| `CGStreamer/shim.h` | C helpers for GObject/GStreamer macros that do not import to Swift |
| `CNvdsParser/nvdsparsebbox_yolo26.cpp` | Custom `nvinfer` bbox parser for YOLO26 `[1, 300, 6]` output |

No `tensorrt-swift` bindings. No Kalman filter. No FFmpeg. No turbojpeg. No Swift-side JPEG encoding or bbox rendering. No `MJPEGSidecar` — Stage 2 removed it.

**TensorRT is still the inference runtime.** `yolo26n_b2_fp16.engine` is a TensorRT FP16 engine, loaded and executed by the `nvinfer` GStreamer element (DeepStream's TRT wrapper). What we removed is the path where Swift code called `enqueueV3` directly via the `tensorrt-swift` Swift bindings; inference now runs inside the GStreamer pipeline, driven by nvinfer, with the same engine.

---

## Architecture

```
 ┌────────────────────────────────────────────────────────────────────────────┐
 │  Jetson Orin Nano 8GB (WendyOS / L4T r36.4.4 / JetPack 6.2.1)            │
 │                                                                            │
 │  ┌─── CDI injection (nvidia-ctk → /etc/cdi/nvidia.yaml) ────────────────┐ │
 │  │  DeepStream 7.1 libs, GStreamer NVMM plugins, CUDA, cuDNN,           │ │
 │  │  libnvv4l2.so Tegra V4L2 userspace shim                              │ │
 │  └───────────────────────────────────────────────────────────────────────┘ │
 │                                                                            │
 │  ┌──────────────────────────────────────────────────────┐                 │
 │  │  mediamtx (systemd, :8554 RTSP relay, :8889 WebRTC)  │                 │
 │  │  rtsp://camera → relay → /relayed → WHEP on :8889    │                 │
 │  └──────────────────────────────────────────────────────┘                 │
 │                                                                            │
 │  ┌─────────────────────────────────────────────────────────────────────┐  │
 │  │  detector (Swift, :9090)                                            │  │
 │  │                                                                     │  │
 │  │  rtspsrc location=<url> latency=200 protocols=tcp                   │  │
 │  │    ! rtph264depay ! h264parse                                       │  │
 │  │    ! nvv4l2decoder          ← NVDEC hardware (NVMM output)          │  │
 │  │    ! nvstreammux batch-size=1 1920x1080                             │  │
 │  │    ! nvinfer config=/app/nvinfer_config.txt  ← YOLO26n FP16 engine  │  │
 │  │    ! nvtracker name=wendy_tracker (NvDCF)                           │  │
 │  │    ! fakesink                                                       │  │
 │  │         ↑                                                           │  │
 │  │   pad probe on nvtracker.src                                        │  │
 │  │   reads NvDsBatchMeta via nvds_shim.c                               │  │
 │  │         ↓                                                           │  │
 │  │   AsyncStream<DetectionFrame>                                       │  │
 │  │     → Detector.swift loop                                           │  │
 │  │     → DetectionBroadcaster (actor)                                  │  │
 │  │     → Metrics.swift                                                 │  │
 │  │                                                                     │  │
 │  │  Hummingbird HTTP :9090                                             │  │
 │  │    /detections — WebSocket (JSON per frame, ptsNs)                  │  │
 │  │    /stream     — 301 redirect to /webrtc/relayed/whep               │  │
 │  │    /metrics    — Prometheus text format                              │  │
 │  │    /health     — JSON                                               │  │
 │  └─────────────────────────────────────────────────────────────────────┘  │
 │                                                                            │
 │  ┌──────────────────┐    ┌────────────────────────────────────────────┐   │
 │  │ llama-vlm :8090  │    │ monitor_proxy.py (dev host)                │   │
 │  │ Qwen3-VL-2B Q4   │    │ /detector/* → 10.42.0.2:9090              │   │
 │  │ llama.cpp server │    │ /webrtc/*   → 10.42.0.2:8889              │   │
 │  │ (currently       │    │ /detector/detections → WS splice           │   │
 │  │  disabled)       │    └────────────────────────────────────────────┘   │
 │  └──────────────────┘                                                     │
 └────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: The Pipeline — Pure Detection (Stage 2)

The full pipeline string (built in `GStreamerFrameReader.buildPipelineString()`):

```
rtspsrc location=<url> latency=200 protocols=tcp
  ! rtph264depay ! h264parse
  ! nvv4l2decoder
  ! m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080
  ! nvinfer config-file-path=/app/nvinfer_config.txt
  ! nvtracker name=wendy_tracker
      ll-config-file=/app/tracker_config.yml
      ll-lib-file=.../libnvds_nvmultiobjecttracker.so
  ! fakesink
```

No tee. No valve. No MJPEG branch. `nvjpegenc`, `nvdsosd`, `nvvideoconvert`, and `appsink` are not in the graph. Video is delivered by `mediamtx` — the RTSP relay it was already running — via WebRTC. The detector only publishes detection metadata.

**Why this works:** The Stage 1 design (tee + MJPEG sidecar) worked correctly but carried measurable cost: ~5 ms of extra preprocess latency (31.5 ms vs 26.2 ms baseline) and ~43 MB of extra nvmap when the MJPEG branch was active. Once `mediamtx` was confirmed to relay the camera stream via WebRTC with reachable ICE candidates, removing the MJPEG branch from the detector was a one-hour change that permanently reclaimed that budget. The cheapest encoder is no encoder.

**Detection branch**: a pad probe on `nvtracker`'s `src` pad intercepts each buffer before it reaches `fakesink`. The C shim (`nvds_shim.c`) reads `NvDsBatchMeta` from the buffer as `GstMeta` — no `gst_buffer_map`, no NVMM pinning.

### Why This Fixes the NVMM Leak

The original design used a single `appsink` and called `gst_buffer_map` on the raw decoded frame buffer. That buffer lives in the NVMM pool — pinned GPU memory managed by `NvBufSurface`. Mapping it pins a pool allocation and requires an explicit unmap. Under the old code the unmap was missing or racing with the buffer reclaim, causing 45–80 MB/s of NVMM pool growth.

The Stage 1 redesign moved the tee after nvtracker and used `NvDsBatchMeta` for detections (no buffer map). The Stage 2 redesign removes the tee entirely. Neither design calls `gst_buffer_map` on any NVMM buffer.

---

## Part 2: YOLO26 — The NMS-Free One-to-One Head

YOLO26 (released Jan 2026) uses a **one-to-one detection head trained with bipartite matching**. The model output is `[1, 300, 6]` — 300 detection slots, each `[x1, y1, x2, y2, confidence, class_id]`. There is no NMS, no sigmoid, no argmax, no DFL.

### Custom BBox Parser

DeepStream's stock parsers assume the YOLOv3–v11 layout (anchor-based or anchor-free with per-class score vectors). YOLO26's output structure requires a custom parser:

```
Sources/CNvdsParser/nvdsparsebbox_yolo26.cpp
```

The parser implements `NvDsInferParseYolo26` (the `nvinfer` custom parser callback). It iterates the 300 slots, filters by `detectionParams.perClassThreshold[class_id]`, and appends to `objectList`. Coordinates are in letterboxed-input pixel space (0..640); `nvinfer` scales back to source-frame pixels when `maintain-aspect-ratio=1` is set.

Cross-compiled on the host:

```bash
clang++ --target=aarch64-linux-gnu \
  --sysroot=$WENDY_SDK \
  -fuse-ld=lld \
  -std=c++14 -Wall -shared -fPIC -O2 \
  -I/opt/nvidia/deepstream/deepstream/sources/includes \
  nvdsparsebbox_yolo26.cpp \
  -o libnvdsparsebbox_yolo26.so
```

`nvinfer_config.txt` sets `cluster-mode=4` to skip clustering entirely — YOLO26's one-to-one head already emits final detections, clustering would corrupt the output.

### One-to-One Head — How It Works

YOLO11 and earlier used a one-to-many assignment: each ground-truth box could match multiple anchor predictions; NMS removed duplicates at inference. YOLO26 uses a **Hungarian-algorithm loss** at training time: each GT matches exactly one prediction. Duplicates are suppressed during training, not at inference. The result is a clean `[1, 300, 6]` tensor with no post-processing required.

Other YOLO26 changes relevant to this deployment:

| Change | Why it matters |
|---|---|
| **DFL removed** | DFL represented bbox coords as probability distributions; removing it simplifies the graph and eliminates the softmax + integral at inference |
| **cluster-mode=4** | nvinfer skips all clustering passes; required for the 1:1 head |
| **ONNX export is simpler** | Fewer ops, more reliable TensorRT conversion |

---

## Part 3: Detection Data Flow — C Shim to AsyncStream

The entire detection path avoids Swift-visible NVMM pointers:

```
GStreamer streaming thread
  → probe_callback() in nvds_shim.c
  → gst_buffer_get_nvds_batch_meta(buf)   // reads GstMeta, no gst_buffer_map
  → wendy_nvds_flatten()                  // flattens into WendyDetection[] on the stack
  → WendyDetectionCallback (function ptr) // calls Swift @_cdecl entry point
  → wendyDetectionProbeEntry()            // copies into [Detection], calls DetectionStream.ingest()
  → AsyncStream<DetectionFrame>.Continuation.yield()
```

`DetectionStream` is a heap-allocated Swift class retained via `Unmanaged.passRetained` for the lifetime of the pipeline. `AsyncStream.Continuation.yield` is `Sendable` and safe to call from any thread — no actor hop required for the callback.

Each `DetectionFrame` carries `timing.ptsNs` — the buffer's GStreamer PTS extracted via `GST_BUFFER_PTS(buf)` in `nvds_shim.c`. The WebSocket broadcaster includes this in the JSON payload as `ptsNs`. The browser uses it to match detection frames against the WebRTC video timeline via `requestVideoFrameCallback`.

The `frameLatencyNs` value is the **inter-frame interval** — time between consecutive probe callbacks. At 20 fps this reads ~50 ms steady-state; spikes reveal pipeline stalls or decode jitter. It is not end-to-end camera-to-detection latency.

### Prometheus Metrics

| Metric | Type | What it measures |
|---|---|---|
| `deepstream_fps` | gauge | Detected frames per second |
| `deepstream_active_tracks` | gauge | Live tracker IDs per frame |
| `deepstream_detections_total` | counter | Cumulative detection count |
| `deepstream_frames_processed_total` | counter | Cumulative frame count |
| `deepstream_total_latency_ms` | histogram | Inter-frame interval (probe-to-probe, not E2E) |
| `deepstream_inference_latency_ms` | histogram | nvinfer processing time (component latency meta) |
| `deepstream_preprocess_latency_ms` | histogram | nvstreammux + nvinfer accumulated |
| `deepstream_decode_latency_ms` | histogram | nvv4l2decoder |
| `deepstream_postprocess_latency_ms` | histogram | nvtracker NvDCF |

---

## Part 4: Hardware Decode

### nvv4l2decoder (NVDEC via Tegra V4L2 userspace shim)

On JetPack 6 / L4T r36.x, `nvv4l2decoder` is **not** a kernel V4L2 M2M driver. It is a userspace V4L2 shim:

```
GStreamer nvv4l2decoder
  → opens /dev/v4l2-nvdec (a char device backed by /dev/null — the fd is a hook, not a real device)
  → libv4l2.so intercepts every V4L2 ioctl
  → libv4l2_nvvideocodec.so (the actual NVIDIA V4L2 plugin)
  → libtegrav4l2.so → NvMMLite → NVDEC hardware
```

The plugin is found via `LIBV4L2_PLUGIN_DIR`, injected into every container by the CDI spec. Without this env var, `libv4l2` falls through to kernel ioctls on `/dev/null`, which silently fails — `nvv4l2decoder` reports "cannot allocate buffers" or similar.

NVDEC output stays in **NVMM** (GPU-accessible DMA-BUF memory). The decoded frames never touch system RAM on their way to `nvinfer`. This is why CPU usage and RSS are low — the CPU never moves pixel data.

---

## Part 5: Stage 2 — Decoupled Video via WebRTC

Stage 2 removed the MJPEG branch from the detector and replaced it with WebRTC delivery from `mediamtx`.

### Why it works

`mediamtx` was already in the stack as an RTSP relay (the camera allows only one RTSP session; mediamtx fans it out). Enabling its built-in WebRTC publication is a config change:

```yaml
webrtc: yes
webrtcAddress: :8889
webrtcLocalUDPAddress: :8189
webrtcAdditionalHosts: [10.42.0.2, 192.168.68.70]
```

`monitor_proxy.py` reverse-proxies `/webrtc/*` to `10.42.0.2:8889/*`. The browser POSTs an SDP offer to `/webrtc/relayed/whep` (WHEP — WebRTC-HTTP Egress Protocol), receives a 201 with the answer SDP, and sets up an `RTCPeerConnection`. No STUN required — ICE candidates in the SDP from mediamtx are reachable directly.

### PTS-accurate overlay sync

Each detection frame carries `ptsNs` (GStreamer buffer PTS in nanoseconds). The browser uses `requestVideoFrameCallback` to get `metadata.mediaTime` for each presented video frame, converts it to nanoseconds, and matches against a 60-frame detection ring buffer. Over the first 8 frames the browser calibrates a `ptsOffsetNs` (the constant delta between `mediaTime * 1e9` and `ptsNs`). After calibration the overlay draws the detection whose `ptsNs` is closest to the estimated playback PTS. Firefox falls back to `requestAnimationFrame` with the latest detection.

### What was removed

- `MJPEGSidecar.swift` — deleted
- `tee`, `valve`, `videodrop`, `nvvideoconvert`, `nvdsosd`, `nvjpegenc`, `appsink` from the pipeline string
- `ENV MJPEG_DISABLED=1` from the Dockerfile (the MJPEG branch is gone, not disabled)
- `COPY libgstvideodrop-stub.so` from the Dockerfile
- `COPY cairo-stage/` from the Dockerfile (`nvdsosd` is no longer in the pipeline; `libcairo` is no longer needed)

### Measured gains

| Metric | Before (Stage 1 MJPEG-on) | After (Stage 2) |
|---|---|---|
| Preprocess latency (steady) | ~31.5 ms | **~26.5 ms** |
| Inference latency (steady) | ~25 ms | **~22.8 ms** |
| nvmap iovmm | ~224 MB (MJPEG-on) | **~206 MB** (MJPEG branch removed) |
| VmRSS | ~620 MB | **~595 MB** |

---

## Part 6: Swift SDK and Cross-Compile

The aarch64 cross-compile uses:

```
~/.swiftpm/swift-sdks/6.2.3-RELEASE_wendyos-device_aarch64.artifactbundle/
```

Invoke:

```bash
swift build --swift-sdk 6.2.3-RELEASE_wendyos-device_aarch64 -c release
```

This runs the Swift compiler natively on x86_64 and emits aarch64 object files directly — no QEMU, no Docker emulation.

### Adding DeepStream Headers to the SDK

The upstream SDK and `meta-wendyos-jetson` Yocto base did not include DeepStream headers or linker stubs at the FHS paths (`/usr/include/`, `/usr/lib/`) that clang needs for cross-compile. The fix lives in two Yocto files:

**`meta-wendyos-jetson/recipes-devtools/deepstream/deepstream-7.1_%.bbappend`**

This bbappend extends the `deepstream-7.1-dev` package's `do_install` to:
1. Copy all `*.h` and `*.hpp` from `$D/opt/nvidia/deepstream/deepstream-7.1/sources/includes/` into `$D/usr/include/`, skipping headers already owned by `tegra-mmapi-dev` (`nvbufsurface.h`, `nvbufsurftransform.h`) to avoid RPM file conflicts.
2. Copy two `.so` linker stubs — `libnvdsgst_meta.so` and `libnvds_meta.so` — from the DS lib dir into `$D/usr/lib/`. These are not versioned symlinks (that would satisfy Yocto's `dev-elf` QA check); they are direct copies used as linker stubs. `INSANE_SKIP:${PN}-dev += "dev-elf"` silences the QA check. `PRIVATE_LIBS:${PN}-dev` suppresses duplicate shlib provider registration — the real DSOs still live under `/opt/nvidia/deepstream/deepstream-7.1/lib/` at runtime and come in via CDI.
3. `FILES:${PN}-dev` is extended to own the new `/usr/include/` copies and the `/usr/lib/` stubs.

**`meta-wendyos-jetson/recipes-core/images/wendyos-sdk-image.bb`**

The SDK image recipe includes `deepstream-7.1-dev` (not `deepstream-7.1` main, which has an `libv4l2.so.0` RPM transaction conflict with `tegra-libraries-multimedia-v4l`). Including only the `-dev` package pulls in the headers and linker stubs but not the runtime DSOs, which avoids the conflict. The SDK image also includes `cuda-libraries-dev`, `cudnn-dev`, `tensorrt-core-dev`, `tegra-mmapi-dev`, `ffmpeg-dev`, and `libjpeg-turbo-dev`.

After the bbappend lands, regenerate the SDK tarball:

```bash
bitbake wendyos-sdk-image
# produces the .artifactbundle.zip used by swift sdk install
```

### Cross-Linker Configuration

The `toolset.json` inside the SDK bundle passes `--target=aarch64-unknown-linux-gnu` to clang (C compiler) and `-Xclang-linker --target=aarch64-unknown-linux-gnu` through swiftc (link step). Without the linker-driver flag, clang defaults to the host triple and `ld.lld` rejects the aarch64 object files with `is incompatible with elf64-x86-64`.

The custom bbox parser `.so` is cross-compiled separately (see Part 2) and `COPY`'d into the container image.

---

## Part 7: Runtime CDI Injection

`nvidia-ctk` generates `/etc/cdi/nvidia.yaml` from CSV files in `/etc/nvidia-container-runtime/host-files-for-container.d/`. Every container with the `"type": "gpu"` entitlement gets bind-mounts of:

- DeepStream 7.1 libs and GStreamer NVMM plugins (`l4t-deepstream.csv`)
- CUDA, cuDNN, CUDA driver libs (`drivers.csv`, `l4t.csv`)
- GStreamer core libs, GLib deps (`l4t-gstreamer.csv`)
- `libnvv4l2.so` Tegra V4L2 userspace plugin chain (`libv4l2_nvvideocodec.so`, `libtegrav4l2.so`)
- Device nodes (`/dev/v4l2-nvdec`, `/dev/nvidia0`, `/dev/nvhost-*`)
- Environment variables: `LIBV4L2_PLUGIN_DIR`, `GST_PLUGIN_PATH`

The `LIBV4L2_PLUGIN_DIR` env var is injected via a `fix-cdi-gstreamer-paths.sh` post-processing script because `nvidia-ctk 1.16.2` ignores the `env` type in CSV files. The script patches the generated YAML with `sed` after `nvidia-ctk` runs.

The detector container itself is:
- Base: `ubuntu:24.04`
- COPY: pre-built `Detector` binary + Swift runtime libs
- COPY: `libnvdsparsebbox_yolo26.so` (custom YOLO26 bbox parser)
- COPY: `nvinfer_config.txt`, `tracker_config.yml`, `streams.json`, `labels.txt`

No NVIDIA SDK in the image. No cairo. No videodrop stub. All NVIDIA libs arrive at runtime via CDI.

---

## Part 8: Concurrency Architecture

Two isolation domains in Stage 2 (MJPEGSidecar is gone):

### `GStreamerFrameReader` (actor)

Pipeline state (element pointers, probe ID, `DetectionStream`) is actor-isolated. The GStreamer streaming thread calls the `@_cdecl` probe entry point, which is a free function — not on any actor. It reads the `DetectionStream` via `Unmanaged.takeUnretainedValue()` and calls `continuation.yield()`, which is `Sendable`. No actor hop in the hot path.

### `DetectionBroadcaster` (actor)

Owns the subscriber map for WebSocket `/detections` clients. Each subscriber gets its own `AsyncStream<String>` continuation. The broadcaster's `distribute(_:)` method serializes fan-out under actor isolation without locks. Slow clients drop stale frames via per-client `bufferingNewest(4)`.

### `MetricsRegistry` (Mutex, not actor)

Called on the hot path at 20+ FPS per stream. A `Mutex<RegistryState>` (from `swift-synchronization`) avoids the async hop that an actor requires. The `/metrics` endpoint snapshots the registry in one lock acquisition, then formats outside the lock.

---

## Part 9: The VLM Second Stage (Currently Disabled)

The infrastructure for VLM descriptions is present but the call is disabled in `Detector.swift` (the `TrackFinalizer` is instantiated but not wired into the detection loop).

When enabled: when the NvDCF tracker drops a confirmed track (object leaves scene), `TrackFinalizer` crops the best-confidence frame and POSTs it to a local llama.cpp sidecar running Qwen3-VL-2B Q4_K_M:

```
docker run --runtime nvidia --name llama-vlm --restart unless-stopped \
  -p 8090:8090 -v /root/vlm-models:/models \
  dustynv/llama_cpp:b5283-r36.4-cu128-24.04 \
  llama-server \
    --model /models/Qwen3-VL-2B-Instruct-Q4_K_M.gguf \
    --mmproj /models/mmproj-F16.gguf \
    --host 0.0.0.0 --port 8090 \
    --n-gpu-layers 99 --ctx-size 4096
```

Memory footprint: ~2.2 GB (LM weights + mmproj + KV cache). Coexists with the YOLO inference load because `nvinfer` runs within DeepStream's CUDA context while llama.cpp has its own — both share the Orin Nano's unified LPDDR5.

**Why llama.cpp, not TRT-LLM or vLLM:** TRT-LLM has no verified Nano support and requires ~16 GB RAM to cross-build a 2B engine. vLLM's PagedAttention pre-allocates 6.4 GB on unified memory and is fragile alongside a running CUDA inference workload. llama.cpp uses ~2.2 GB, lazily allocates, and has explicit Qwen3-VL support.

---

## Part 10: Dependencies

```
hummingbird 2.6+       — HTTP server (Hummingbird 2, Sendable-clean)
swift-argument-parser  — CLI
swift-log              — Structured logging
async-http-client      — VLM sidecar calls (currently disabled path)

libgstreamer-1.0       — Video pipeline (system library, CDI-injected at runtime)
DeepStream 7.1 libs    — nvinfer, nvtracker (CDI-injected; nvdsosd/nvjpegenc no longer used)
```

Removed from the previous design: `tensorrt-swift`, `libturbojpeg`, `libavformat/avcodec` (FFmpeg), `swift-container-plugin` (now using a Dockerfile again).

---

## Part 11: Running It

### Step 1: Install the Swift SDK

```bash
swift sdk install \
  https://..../6.2.3-RELEASE_wendyos-device_aarch64.artifactbundle.zip \
  --checksum <sha256>
```

### Step 2: Configure your camera

```json
// detector-swift/streams.json
{
  "streams": [
    {
      "name": "camera1",
      "url": "rtsp://user:pass@192.168.1.100:554/stream1",
      "enabled": true
    }
  ]
}
```

### Step 3: Build and deploy

```bash
cd detector-swift
swift build --swift-sdk 6.2.3-RELEASE_wendyos-device_aarch64 -c release

# Cross-compile the bbox parser
clang++ --target=aarch64-linux-gnu --sysroot=$WENDY_SDK -fuse-ld=lld \
  -std=c++14 -shared -fPIC -O2 \
  Sources/CNvdsParser/nvdsparsebbox_yolo26.cpp \
  -o libnvdsparsebbox_yolo26.so

WENDY_AGENT=10.42.0.2 wendy run -y --detach
```

### Step 4: View

```bash
# Prometheus metrics
curl http://<device>:9090/metrics | grep deepstream_fps

# Detection WebSocket (raw)
websocat ws://<device>:9090/detections

# Full dashboard (via monitor_proxy.py on dev host)
open http://localhost:8001/
# Video: WebRTC via /webrtc/relayed/whep (WHEP)
# Detections: WebSocket via /detector/detections
```

---

## Head-to-Head: Python + DeepStream vs. Swift + DeepStream

Both detectors run on the same hardware (Jetson Orin Nano 8GB, JetPack 6.2.1) against the same 1080p/20fps RTSP camera (via `mediamtx` relay), doing object detection with tracking and serving Prometheus metrics over HTTP. Both now run the same `yolo26n_b2_fp16.engine` with the same `libnvdsparsebbox_yolo26.so` parser — model parity is new in this run.

**Preliminary benchmark.** Numbers below come from the 2026-04-16 run and the Stage 2 measurement on 2026-04-15. Swift steady-state numbers are from ~4-minute windows. Python exited ~90 s into sampling for reasons not yet determined, so the concurrent comparison is an open debug item. See "Benchmark caveats" below.

### The Numbers

| | Python + DeepStream | Swift + DeepStream (Stage 2) |
|---|---|---|
| **FPS** | ~20 (camera-limited; source-capped) | ~20 (camera-limited; source-capped) |
| **Inference** | `nvinfer` (shared) | `nvinfer` (shared) |
| **CPU usage** | 80–100% | **~55%** |
| **Memory (VmRSS)** | ~615 MB (90 s before exit — no steady state) | **~595 MB steady** |
| **Inference latency (pad-probe histogram)** | n/a (no steady state) | **~22.8 ms** |
| **Preprocess latency** | n/a | **~26.5 ms** (nvstreammux + nvinfer) |
| **Postprocess latency** | n/a | ~14 ms (nvtracker NvDCF) |
| **Nvmap iovmm** | ~900 MB pool (during concurrent window) | **~206 MB steady-state** (both direct and relay paths; MJPEG branch removed) |
| **Decode** | Software (`uridecodebin` fallback) | **NVDEC hardware** (`nvv4l2decoder`) |
| **Model** | YOLO26n FP16 (same engine as Swift) | YOLO26n FP16 |
| **Tracking** | NvDCF (DeepStream) | NvDCF (DeepStream) |
| **Video delivery** | MJPEG via Flask | **WebRTC via mediamtx** |
| **VLM second stage** | None | Qwen3-VL-2B via llama.cpp (disabled) |
| **Build time** | Minutes (Docker on device) | **~0.7s** incremental cross-compile |
| **Lines of code** | ~1,438 (Python) | ~1,600 (Swift + C shim; Stage 2 deleted ~400 lines) |

**Critical note on Swift latency numbers.** The 22.8 ms / 26.5 ms figures are **pad-probe histogram averages that include `nvstreammux` queueing time**, not pure compute. Real `nvinfer` compute for YOLO26n FP16 on Orin Nano is in the ~8–12 ms range (verifiable via Nsight Systems); the histogram delta includes mux wait. Reporting the histogram as "inference latency" is convenient but misleading — it's pipeline-wall-clock for the buffer passing the probe, not GPU time.

**Nvmap relay number.** Stage 2 nvmap baseline is **~206 MB steady-state**. The ~23 MB drop from the pre-Stage-2 224 MB reflects removing the MJPEG branch (nvjpegenc pool + nvdsosd surfaces). The relay path itself adds no measurable nvmap cost — a direct-camera vs. mediamtx-relay A/B confirmed both paths reach the same ~206 MB pool once the decoder finishes warmup. Earlier "915 MB pool during concurrent window" numbers were the two-pipeline sum, not a relay tax.

**Concurrent window: inconclusive.** Under load Swift's histogram latencies did not inflate, which at first looked like headroom. More likely, Python died during TRT warmup and was never exercising the GPU at steady state. Re-run after Python stability is resolved before drawing load-sharing conclusions.

### Benchmark caveats

- **Bring-up contamination.** Swift was restarted mid-session (resets its histograms); the container built during the first samples; `mediamtx` was added to the path mid-session. The "window" is not a single uniform condition.
- **No thermal or clock control.** `jetson_clocks` was not invoked and thermal state was not monitored. Orin Nano DVFS can swing compute headroom noticeably between a cold and warm device.
- **Sampling cadence is coarse.** 15 s polling against ~25 ms inference spikes.
- **Small-N statistics.** Fine for order-of-magnitude, not fine for asserting differences smaller than sample variance.

These numbers are a baseline for "does Swift hold steady-state on this hardware." They are not a head-to-head verdict until Python stability is resolved.

### Where Swift Wins

**CPU efficiency.** The Python version runs at 80–100% CPU on this hardware; the Swift version at ~55%. The likely mechanism is that Python's GIL serializes GStreamer callbacks and HTTP serving onto one core, while Swift uses structured concurrency — the probe callback fires on the GStreamer streaming thread (not the cooperative pool), and HTTP serving uses Hummingbird's NIO event loops.

**Hardware decode.** The Python version's `uridecodebin` silently fell back to software decode (`avdec_h264`). The Swift version explicitly uses `nvv4l2decoder` with the corrected `LIBV4L2_PLUGIN_DIR` chain, offloading H.264 decode to NVDEC.

**NVMM correctness.** The old appsink+map design leaked NVMM at 45–80 MB/s. The current design has zero `gst_buffer_map` on NVMM buffers — detection flows through `NvDsBatchMeta`.

**Build cycle.** Cross-compile on x86 in 0.7 seconds (incremental). The Python version builds inside Docker on the device.

### Where Python Wins

**Lines of code.** The Python version is shorter because DeepStream handles more — the pipeline itself is the program. The Swift version must explicitly manage the pipeline, probe installation, async stream bridging, and HTTP serving.

**Ecosystem.** DeepStream is NVIDIA's supported production framework. The Swift-on-Jetson path is uncharted — every CDI issue, GStreamer version mismatch, and libv4l2 plugin chain problem had to be debugged from first principles.

---

## War Stories

1. **NVMM leak (45–80 MB/s).** The original design used `appsink` at the end of the full nvinfer+nvtracker pipeline and called `gst_buffer_map` on the decoded frame buffer. NVMM buffers are pool-allocated; mapping pins them. The unmap was racing with the pipeline's buffer reclaim. Fix: use the pad probe + `NvDsBatchMeta` for detections; in Stage 2, remove the MJPEG branch entirely.

2. **`/dev/v4l2-nvdec` is `/dev/null`.** On L4T r36.x the entire NVDEC stack is in userspace (libv4l2 plugin chain). The device node is only there so `open()` returns a valid fd. Without `LIBV4L2_PLUGIN_DIR` pointing at `libv4l2_nvvideocodec.so`, libv4l2 passes ioctls through to the kernel fd (which is `/dev/null`) and `nvv4l2decoder` silently fails with "cannot allocate buffers". Fix: inject `LIBV4L2_PLUGIN_DIR` via CDI.

3. **DS headers missing from the Swift SDK sysroot.** The upstream SDK had no DeepStream headers or linker stubs at FHS paths. Fix: bbappend in `meta-wendyos-jetson` to install them as part of `deepstream-7.1-dev`, then rebuild the SDK image via `bitbake wendyos-sdk-image`.

4. **`cairo-stage/` was required by nvdsosd.** `nvdsosd` links `libcairo.so` at build time. If the library is absent from the container at startup, `nvdsosd` element creation fails even though the CUDA OSD code path never calls cairo. Fix in Stage 1: stage `cairo-stage/` into the container image. Fix in Stage 2: remove `nvdsosd` from the pipeline entirely; `cairo-stage/` is no longer needed.

5. **`cluster-mode=4` is mandatory.** Leaving `cluster-mode` at the default (0 = OpenCV groupRectangles) with a 1:1 YOLO26 head produces garbled detections — the clustering code is not designed for pre-de-duplicated inputs. Fix: `cluster-mode=4` in `nvinfer_config.txt`.

6. **gst-plugin-scanner ABI mismatch.** The Ubuntu 24.04 `gst-plugin-scanner` binary is ABI-incompatible with the L4T GStreamer 1.22 core injected by CDI. Setting `GST_PLUGIN_SCANNER=""` disables the external scanner and lets GStreamer fall back to in-process scanning. `gst_registry_scan_path` is called programmatically in `GStreamerFrameReader.setupPluginEnvironment()`.

7. **videodrop ABI collision (Stage 1, now gone).** The Ubuntu gst-plugins-bad `videodrop` element failed to load on the CDI runtime due to `g_once_init_leave_pointer: undefined symbol` — a GLib ABI mismatch between Ubuntu 24.04 and the CDI-injected JetPack GLib. Stage 1 worked around this with a stub `.so`. Stage 2 removed the need entirely by deleting the MJPEG branch.
