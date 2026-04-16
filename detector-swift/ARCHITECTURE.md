# Swift on the Edge: YOLO26 + Qwen3-VL on a Jetson Orin Nano

*DeepStream-native Swift detector: nvinfer + nvtracker in-pipeline, hardware NVDEC decode, zero NVMM buffer mapping, in-pipeline OSD overlays.*

---

## The Pitch

We replaced a **1,438-line Python detector** (built on NVIDIA's DeepStream SDK + GStreamer + PyGObject + NumPy + OpenCV + Flask) with a **Swift 6.2 implementation** that:

- Runs **YOLO26n** (NMS-free end-to-one detector) through DeepStream's `nvinfer` plugin
- Tracks objects with **NvDCF** (`nvtracker`, `libnvds_nvmultiobjecttracker.so`)
- Decodes H.264 via **NVDEC hardware** (`nvv4l2decoder`) — never touches CPU-side pixels
- Draws bounding boxes **in the pipeline** via `nvdsosd` — no Swift-side rendering
- Encodes MJPEG frames via **nvjpegenc hardware** — the only `gst_buffer_map` in the whole pipeline is on system-memory JPEG output, which is cheap
- Sends best-frame crops to a **Qwen3-VL-2B** sidecar via llama.cpp for natural-language descriptions (currently disabled, infrastructure in place)
- **Cross-compiles natively** from x86 to aarch64 using the `6.2.3-RELEASE_wendyos-device_aarch64` Swift SDK

Numbers on a Jetson Orin Nano 8GB (release build):

| Metric | Value |
|---|---|
| End-to-end FPS | **~20** (camera-limited, 1080p RTSP) |
| CPU usage | **~55%** |
| NVMM memory growth | **0 MB/s** (was 45–80 MB/s with the old appsink+map design) |
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
| `GStreamerFrameReader.swift` | DeepStream pipeline actor; owns the tee pipeline, installs pad probe on nvtracker src |
| `MJPEGSidecar.swift` | Pulls JPEG frames from appsink (system-memory only), broadcasts to HTTP clients |
| `HTTPServer.swift` | Hummingbird 2 HTTP server: /stream (MJPEG), /metrics (Prometheus), /health |
| `Metrics.swift` | Prometheus registry backed by `Mutex<RegistryState>` (not actor — too hot for async hops) |
| `TrackFinalizer.swift` | Queues finalized tracks, crops best frame, submits to VLM (currently disabled) |
| `VLMClient.swift` | async HTTP client for llama.cpp sidecar (OpenAI format) |
| `CGStreamer/nvds_shim.c` | C shim: reads `NvDsBatchMeta` from a GstBuffer via a pad probe, flattens into `WendyDetection[]` |
| `CGStreamer/shim.h` | C helpers for GObject/GStreamer macros that do not import to Swift |
| `CNvdsParser/nvdsparsebbox_yolo26.cpp` | Custom `nvinfer` bbox parser for YOLO26 `[1, 300, 6]` output |

No `tensorrt-swift` bindings. No Kalman filter. No FFmpeg. No turbojpeg. No Swift-side JPEG encoding or bbox rendering.

**TensorRT is still the inference runtime.** `yolo26n_b2_fp16.engine` is a TensorRT FP16 engine, loaded and executed by the `nvinfer` GStreamer element (DeepStream's TRT wrapper). What we removed is the path where Swift code called `enqueueV3` directly via the `tensorrt-swift` Swift bindings; inference now runs inside the GStreamer pipeline, driven by nvinfer, with the same engine.

---

## Architecture

```
 ┌────────────────────────────────────────────────────────────────────────────┐
 │  Jetson Orin Nano 8GB (WendyOS / L4T r36.4.4 / JetPack 6.2.1)            │
 │                                                                            │
 │  ┌─── CDI injection (nvidia-ctk → /etc/cdi/nvidia.yaml) ────────────────┐ │
 │  │  DeepStream 7.1 libs, GStreamer NVMM plugins, CUDA, cuDNN,           │ │
 │  │  libnvv4l2.so Tegra V4L2 userspace shim, nvjpeg, nvdsosd            │ │
 │  └───────────────────────────────────────────────────────────────────────┘ │
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
 │  │    ! tee name=t                                                     │  │
 │  │         │                                                           │  │
 │  │    ┌────┴─────────────────────────────────────┐                    │  │
 │  │    │ detection branch          MJPEG branch    │                    │  │
 │  │    │                                           │                    │  │
 │  │  t. ! queue                  t. ! queue        │                    │  │
 │  │    ! fakesink                   max-size-buffers=2 leaky=downstream │  │
 │  │    ↑                           ! nvvideoconvert                    │  │
 │  │  pad probe on                  ! nvdsosd        ← draws bboxes     │  │
 │  │  nvtracker.src                 ! nvvideoconvert                    │  │
 │  │  reads NvDsBatchMeta           ! nvjpegenc quality=85              │  │
 │  │  via nvds_shim.c               ! appsink name=mjpeg_sink           │  │
 │  │    ↓                             max-buffers=1 drop=true           │  │
 │  │  AsyncStream<DetectionFrame>    ↓                                  │  │
 │  │    → Detector.swift loop       MJPEGSidecar (actor)                │  │
 │  │    → Metrics.swift              → AsyncStream<[UInt8]>             │  │
 │  │                                 per HTTP client                    │  │
 │  │                                                                     │  │
 │  │  Hummingbird HTTP :9090                                             │  │
 │  │    /stream   — MJPEG (multipart/x-mixed-replace)                   │  │
 │  │    /metrics  — Prometheus text format                               │  │
 │  │    /health   — JSON                                                 │  │
 │  └─────────────────────────────────────────────────────────────────────┘  │
 │                                                                            │
 │  ┌──────────────────┐    ┌────────────────────────────────────────────┐   │
 │  │ llama-vlm :8090  │    │ monitor_proxy.py (dev host)                │   │
 │  │ Qwen3-VL-2B Q4   │    │ reverse-proxy → device :9090               │   │
 │  │ llama.cpp server │    │ same-origin access for monitor.html        │   │
 │  │ (currently       │    └────────────────────────────────────────────┘   │
 │  │  disabled)       │                                                     │
 │  └──────────────────┘                                                     │
 └────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: The Pipeline — Why the Tee Is After nvtracker

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
  ! tee name=t
  t. ! queue ! fakesink
  t. ! queue max-size-buffers=2 leaky=downstream
     ! nvvideoconvert
     ! nvdsosd
     ! nvvideoconvert
     ! nvjpegenc quality=85
     ! appsink name=mjpeg_sink emit-signals=false sync=false
                max-buffers=1 drop=true
```

The tee placement is deliberate. `NvDsBatchMeta` — the GStreamer metadata structure that carries per-object bounding boxes, class IDs, confidence scores, and tracker IDs — is attached to each buffer by `nvinfer` and enriched by `nvtracker`. Both downstream branches therefore carry the full metadata.

- **Detection branch**: `fakesink` discards the buffer. A pad probe on `nvtracker`'s `src` pad intercepts each buffer *before* it reaches the sink. The C shim (`nvds_shim.c`) reads `NvDsBatchMeta` from the buffer as `GstMeta` — no `gst_buffer_map`, no NVMM pinning.
- **MJPEG branch**: `nvdsosd` reads `NvDsObjectMeta.rect_params` and draws colored bounding boxes directly onto the NVMM surface in-place. `nvjpegenc` then hardware-encodes the annotated frame to a system-memory JPEG. The `gst_buffer_map` in `MJPEGSidecar` operates on this system-memory JPEG buffer, which is cheap (malloc'd pages, not NVMM pool).

### Why This Fixes the NVMM Leak

The previous design used a single `appsink` and called `gst_buffer_map` on the raw decoded frame buffer. That buffer lives in the NVMM pool — pinned GPU memory managed by `NvBufSurface`. Mapping it pins a pool allocation and requires an explicit unmap. Under the old code the unmap was missing or racing with the buffer reclaim, causing 45–80 MB/s of NVMM pool growth.

The new design never calls `gst_buffer_map` on any NVMM buffer. Detection data flows through `GstMeta` (zero-copy metadata on the buffer). JPEG output arrives in system memory, where mapping is safe and idempotent.

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

The latency value (`frameLatencyNs`) reported in each `DetectionFrame` is the **inter-frame interval** — time between consecutive probe callbacks, computed using `gst_util_get_timestamp()` diffs in the C shim. At 20 fps this reads ~50 ms steady-state; spikes reveal pipeline stalls or decode jitter.

**Important caveat on `deepstream_total_latency_ms`:** this metric measures the probe callback interval, not end-to-end camera-to-detection latency. `rtspsrc` tags live buffers with RTP-derived PTS that is typically offset from the pipeline `base_time` by several seconds (the stream may have started before the pipeline). Using `buffer.PTS - base_time` would give nonsense. The inter-frame interval is a reliable proxy for pipeline health but is not end-to-end latency.

### Prometheus Metrics

| Metric | Type | What it measures |
|---|---|---|
| `deepstream_fps` | gauge | Detected frames per second |
| `deepstream_active_tracks` | gauge | Live tracker IDs per frame |
| `deepstream_detections_total` | counter | Cumulative detection count |
| `deepstream_frames_processed_total` | counter | Cumulative frame count |
| `deepstream_total_latency_ms` | histogram | Inter-frame interval (probe-to-probe, not E2E) |
| `deepstream_inference_latency_ms` | histogram | nvinfer processing time |
| `deepstream_gpu_memory_mb` | gauge | NVMM pool allocation (canary for leaks) |

---

## Part 4: In-Pipeline Overlays via nvdsosd

`nvdsosd` reads `NvDsObjectMeta.rect_params` (left, top, width, height in source-frame pixels) and `border_color` / `bg_color` from `NvDsObjectMeta`, then draws bounding boxes directly onto the NVMM surface using CUDA kernels. No Swift-side rendering, no cairo, no turbojpeg, no font blitting.

The MJPEG branch after `nvdsosd` is what the user sees in the browser: red boxes around detected objects, drawn before JPEG encode. `nvdsosd`'s default colors are used; per-class coloring would require setting `border_color` in a custom `NvDsBatchMeta` mutation, which is future work.

`cairo-stage/` in the container image (`aarch64 libcairo + transitive deps`, staged from `~/jetson/Linux_for_Tegra/rootfs/`) is required because `nvdsosd` links against `libcairo.so` even when only the CUDA OSD path is used. The library must be present on the dynamic linker path at startup.

---

## Part 5: Hardware Decode and Encode

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

NVDEC output stays in **NVMM** (GPU-accessible DMA-BUF memory). The decoded frames never touch system RAM on their way to `nvinfer` or `nvdsosd`. This is why CPU usage and RSS are low — the CPU never moves pixel data.

### nvjpegenc (Hardware JPEG)

`nvjpegenc` takes the annotated NVMM surface from `nvdsosd` (after the second `nvvideoconvert`) and encodes it to a JPEG at `quality=85`. The output buffer is in **system memory** (malloc'd pages). `MJPEGSidecar` calls `gst_buffer_map` on this buffer — cheap, no NVMM pool impact.

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
- COPY: `cairo-stage/` (aarch64 libcairo + transitive deps staged from `~/jetson/Linux_for_Tegra/rootfs/` to satisfy `nvdsosd`'s link-time dependency)
- COPY: `nvinfer_config.txt`, `tracker_config.yml`, `streams.json`, `labels.txt`

No NVIDIA SDK in the image. All NVIDIA libs arrive at runtime via CDI.

---

## Part 8: Concurrency Architecture

Three isolation domains, each chosen for a specific reason:

### `GStreamerFrameReader` (actor)

Pipeline state (element pointers, probe ID, `DetectionStream`) is actor-isolated. The GStreamer streaming thread calls the `@_cdecl` probe entry point, which is a free function — not on any actor. It reads the `DetectionStream` via `Unmanaged.takeUnretainedValue()` and calls `continuation.yield()`, which is `Sendable`. No actor hop in the hot path.

### `MJPEGSidecar` (actor)

Owns the pull loop task and the subscriber map. The pull loop runs in a detached `Task` that calls `wendy_gst_pull_sample` (the only `gst_buffer_map` in the codebase). Subscribers receive frames via `AsyncStream<[UInt8]>`. Actor isolation serializes subscribe/unsubscribe/broadcast without locks.

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
DeepStream 7.1 libs    — nvinfer, nvtracker, nvdsosd, nvjpegenc (CDI-injected)
```

Removed from the previous design: `tensorrt-swift`, `libturbojpeg`, `libavformat/avcodec` (FFmpeg), `swift-container-plugin` (now using a Dockerfile again — see the `Dockerfile` at the repo root).

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

# Build and push the container image (Dockerfile at repo root)
docker buildx build --platform=linux/arm64 \
  -t <device>:5000/detector:latest --push .
```

### Step 4: Start the container (on device)

```bash
ctr run --runtime io.containerd.runc.v2 \
  --net-host \
  --device-config /etc/cdi/nvidia.yaml \
  <device>:5000/detector:latest detector
```

### Step 5: View

```bash
# Live MJPEG with bounding boxes drawn by nvdsosd
open http://<device>:9090/stream

# Prometheus metrics
curl http://<device>:9090/metrics | grep deepstream_fps

# Full dashboard (via monitor_proxy.py on dev host)
open http://localhost:8080/monitor.html
```

---

## Head-to-Head: Python + DeepStream vs. Swift + DeepStream

Both detectors run on the same hardware (Jetson Orin Nano 8GB, JetPack 6.2.1) against the same 1080p/20fps RTSP camera, doing object detection with tracking and serving Prometheus metrics over HTTP.

**Note: quantitative numbers below are to be updated by a dedicated benchmarking task. The qualitative observations are current.**

### The Numbers (to be updated)

| | Python + DeepStream | Swift + DeepStream |
|---|---|---|
| **FPS** | ~20 (camera-limited) | ~20 (camera-limited) |
| **Inference** | `nvinfer` (shared) | `nvinfer` (shared) |
| **CPU usage** | 80–100% | **~55%** |
| **Memory (RSS)** | ~1.5 GB | **to be benchmarked** |
| **Decode** | Software (`uridecodebin` fallback) | **NVDEC hardware** (`nvv4l2decoder`) |
| **Model** | YOLO11n | YOLO26n FP16 |
| **Tracking** | NvDCF (DeepStream) | NvDCF (DeepStream) |
| **VLM second stage** | None | Qwen3-VL-2B via llama.cpp (disabled) |
| **Build time** | Minutes (Docker on device) | **~0.7s** incremental cross-compile |
| **Lines of code** | ~1,438 (Python) | ~2,000 (Swift + C shim) |

### Where Swift Wins

**CPU efficiency.** The Python version burns 80–100% CPU because the GIL serializes GStreamer callbacks and HTTP serving onto one core. The Swift version uses structured concurrency — the probe callback fires on the GStreamer streaming thread (not the cooperative pool), MJPEG pulls run in a detached actor task, and HTTP serving uses Hummingbird's NIO event loops. No contention, no GC pauses.

**Hardware decode.** The Python version's `uridecodebin` silently fell back to software decode (`avdec_h264`). The Swift version explicitly uses `nvv4l2decoder` with the corrected `LIBV4L2_PLUGIN_DIR` chain, offloading H.264 decode to NVDEC.

**NVMM correctness.** The old appsink+map design leaked NVMM at 45–80 MB/s. The new design has zero `gst_buffer_map` on NVMM buffers — detection flows through `NvDsBatchMeta`, MJPEG output arrives in system memory.

**Build cycle.** Cross-compile on x86 in 0.7 seconds (incremental). The Python version builds inside Docker on the device.

### Where Python Wins

**Lines of code.** The Python version is shorter because DeepStream handles more — the pipeline itself is the program. The Swift version must explicitly manage the pipeline, probe installation, async stream bridging, and HTTP serving.

**Ecosystem.** DeepStream is NVIDIA's supported production framework. The Swift-on-Jetson path is uncharted — every CDI issue, GStreamer version mismatch, and libv4l2 plugin chain problem had to be debugged from first principles.

---

## War Stories

1. **NVMM leak (45–80 MB/s).** The original design used `appsink` at the end of the full nvinfer+nvtracker pipeline and called `gst_buffer_map` on the decoded frame buffer. NVMM buffers are pool-allocated; mapping pins them. The unmap was racing with the pipeline's buffer reclaim. Fix: move the tee after nvtracker, use the pad probe + `NvDsBatchMeta` for detections, and map only the system-memory JPEG output.

2. **`/dev/v4l2-nvdec` is `/dev/null`.** On L4T r36.x the entire NVDEC stack is in userspace (libv4l2 plugin chain). The device node is only there so `open()` returns a valid fd. Without `LIBV4L2_PLUGIN_DIR` pointing at `libv4l2_nvvideocodec.so`, libv4l2 passes ioctls through to the kernel fd (which is `/dev/null`) and `nvv4l2decoder` silently fails with "cannot allocate buffers". Fix: inject `LIBV4L2_PLUGIN_DIR` via CDI.

3. **DS headers missing from the Swift SDK sysroot.** The upstream SDK had no DeepStream headers or linker stubs at FHS paths. Fix: bbappend in `meta-wendyos-jetson` to install them as part of `deepstream-7.1-dev`, then rebuild the SDK image via `bitbake wendyos-sdk-image`.

4. **`nvdsosd` requires libcairo even on the CUDA path.** `nvdsosd` links `libcairo.so` at build time. If the library is absent from the container at startup, `nvdsosd` element creation fails even though the CUDA OSD code path never calls cairo. Fix: stage `cairo-stage/` into the container image.

5. **`cluster-mode=4` is mandatory.** Leaving `cluster-mode` at the default (0 = OpenCV groupRectangles) with a 1:1 YOLO26 head produces garbled detections — the clustering code is not designed for pre-de-duplicated inputs. Fix: `cluster-mode=4` in `nvinfer_config.txt`.

6. **gst-plugin-scanner ABI mismatch.** The Ubuntu 24.04 `gst-plugin-scanner` binary is ABI-incompatible with the L4T GStreamer 1.22 core injected by CDI. Setting `GST_PLUGIN_SCANNER=""` disables the external scanner and lets GStreamer fall back to in-process scanning. `gst_registry_scan_path` is called programmatically in `GStreamerFrameReader.setupPluginEnvironment()`.
