# Swift on the Edge: YOLO26 + Qwen3-VL on a Jetson Orin Nano

*Replacing Python + DeepStream with pure Swift 6.2, then going deeper: native cross-compile, hardware decode via GStreamer, and on-device VLM descriptions.*

---

## The Pitch

We replaced a **1,438-line Python detector** (built on NVIDIA's DeepStream SDK + GStreamer + PyGObject + NumPy + OpenCV + Flask) with a **~4,250-line pure Swift 6.2 implementation** that:

- Runs **YOLO26n** (the NMS-free end-to-end detector) directly against TensorRT 10.3
- Pulls RTSP frames through **GStreamer via C interop** (no subprocess, no FFmpeg glue)
- Tracks objects with a **SORT-style Kalman tracker**
- Sends best-frame crops to a **Qwen3-VL-2B** sidecar via llama.cpp for natural-language descriptions
- **Cross-compiles natively** from x86 to aarch64 in 85 seconds (vs. 25 minutes through Docker + QEMU)
- **Deploys without Docker** — the whole pipeline uses the Swift Container Plugin path

Live numbers on a Jetson Orin Nano 8GB (release build, hardware NVDEC decode):

| Metric | Value |
|---|---|
| End-to-end FPS | **~20** (camera-limited, 1080p RTSP) |
| YOLO26 inference | **29.2 ms** per frame (FP16 TensorRT 10.3) |
| CPU usage | **54.5%** (vs 80-100% for Python + DeepStream) |
| Memory (RSS) | **686 MB** (vs ~1.5 GB for Python) |
| Build time (cross-compile) | **0.7 s** incremental, 14s cold |
| Build time (old Docker + QEMU path) | ~1500 s |

A representative run produced descriptions like:

> "This is a red 2004 Toyota Corolla, a compact hatchback with a sleek design and a well-known Toyota emblem on the front grille."

All generated on-device. Nothing leaves the Jetson.

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

## What We Built (12 Swift files + 1 C shim)

| File | Lines | Responsibility |
|------|------:|----------------|
| `Detector.swift` | 285 | Entry point, CLI args, per-stream detection loop |
| `DetectorEngine.swift` | 358 | TensorRT engine lifecycle, single/batch inference, engine-to-disk caching |
| `GStreamerFrameReader.swift` | 397 | **NEW** — in-process GStreamer pipeline, nvv4l2decoder → RGB → appsink |
| `RTSPFrameReader.swift` | 319 | FFmpeg subprocess fallback (kept for non-Jetson platforms) |
| `YOLOPreprocessor.swift` | 194 | Letterbox resize, bilinear interpolation, HWC to CHW |
| `YOLOPostprocessor.swift` | **144** | YOLO26 output decode — **no NMS**, just confidence filter (was 286 lines) |
| `Tracker.swift` | 629 | SORT-style IoU tracker, 8-state Kalman filter, finalized-track emission |
| `TrackFinalizer.swift` | 171 | **NEW** — queues finalized tracks, crops best frame, submits to VLM |
| `VLMClient.swift` | 223 | **NEW** — async HTTP client for the llama.cpp sidecar (OpenAI format) |
| `FrameRenderer.swift` | 567 | Bbox drawing, 5×7 bitmap font, turbojpeg encoding |
| `HTTPServer.swift` | 370 | Hummingbird 2 server: MJPEG stream, /metrics, /health, VLM endpoints |
| `Metrics.swift` | 466 | Prometheus registry with Mutex, histograms, counters |
| `CGStreamer/shim.h` | 133 | **NEW** — C helpers for GStreamer macros that don't import to Swift |

**Total: 4,256 lines of Swift + C. Still zero C++ wrapper code.**

---

## Architecture (Top to Bottom)

```
┌──────────────────────────────────────────────────────────────────────┐
│  Jetson Orin Nano 8GB (WendyOS 0.10.5 / L4T r36.4.4 / JetPack 6.2.1)│
│                                                                      │
│  ┌─────────────────────────────────┐   ┌─────────────────────────┐  │
│  │  detector (Swift, :9090)         │   │ llama-vlm (:8090)       │  │
│  │                                  │   │                         │  │
│  │  RTSP Camera (TP-Link, 1080p20)  │   │ llama.cpp b8748+qwen3vl │  │
│  │           │                      │   │ Qwen3-VL-2B Q4_K_M      │  │
│  │           ▼                      │   │ mmproj F16              │  │
│  │  GStreamerFrameReader            │   │ ~2.2 GB GPU             │  │
│  │    rtspsrc ! rtph264depay !      │   │                         │  │
│  │    h264parse ! nvv4l2decoder !   │   │ POST /v1/chat/completions│  │
│  │    nvvideoconvert ! RGB !        │◄──┤ (image_url + text)      │  │
│  │    appsink                       │   └─────────────────────────┘  │
│  │           │                      │                                │
│  │           ▼                      │                                │
│  │  YOLOPreprocessor                │                                │
│  │    letterbox → RGB → CHW Float   │                                │
│  │           │                      │                                │
│  │           ▼                      │                                │
│  │  DetectorEngine (actor)          │                                │
│  │    TensorRT 10.3 FP16            │                                │
│  │    YOLO26n (NMS-free)            │                                │
│  │    output [1,300,6] = xyxy + conf + class                         │
│  │           │                      │                                │
│  │           ▼                      │                                │
│  │  YOLOPostprocessor               │                                │
│  │    confidence filter only        │                                │
│  │           │                      │                                │
│  │           ▼                      │                                │
│  │  IOUTracker                      │                                │
│  │    Kalman constant-velocity      │                                │
│  │    greedy IoU association        │                                │
│  │    best-frame tracking           │                                │
│  │           │                      │                                │
│  │       ┌───┴───┐                  │                                │
│  │       │       │                  │                                │
│  │       ▼       ▼                  │                                │
│  │  FrameRenderer  TrackFinalizer ──┤                                │
│  │   bbox+font      best crop       │                                │
│  │   turbojpeg      JPEG → VLM      │                                │
│  │       │                          │                                │
│  │       ▼                          │                                │
│  │  Hummingbird HTTP :9090          │                                │
│  │    /stream   — MJPEG             │                                │
│  │    /metrics  — Prometheus        │                                │
│  │    /health   — JSON              │                                │
│  │    /api/vlm_descriptions         │                                │
│  │    /api/vlm_status               │                                │
│  └─────────────────────────────────┘                                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: YOLO26 — When "Upgrade" Means "Delete Code"

The original Swift detector ran YOLO11n with the classic post-processing dance:

- Decode `[1, 84, 8400]` transposed output tensor
- Apply sigmoid to class logits
- Argmax over 80 COCO classes per anchor
- Greedy per-class NMS at IoU 0.45
- TopK cap at 300 detections

That was ~286 lines of `YOLOPostprocessor.swift`.

YOLO26 (released Jan 2026) ships an **NMS-free one-to-one detection head**. The model output is `[1, 300, 6]` — 300 detection slots, each `[x1, y1, x2, y2, confidence, class_id]`. Duplicates are resolved inside the model via bipartite-matching loss during training. DFL is gone too.

The Swift postprocessor became:

```swift
for slot in 0 ..< maxDetections {
    let offset = slot * 6
    let confidence = base[offset + 4]
    guard confidence >= confidenceThreshold else { continue }
    // x1,y1,x2,y2 → xywh
    let x1 = base[offset + 0]
    let y1 = base[offset + 1]
    let x2 = base[offset + 2]
    let y2 = base[offset + 3]
    let classId = Int(base[offset + 5])
    detections.append(Detection(x: x1, y: y1,
                                width: x2 - x1, height: y2 - y1,
                                classId: classId,
                                confidence: confidence,
                                label: labels[classId]))
}
```

**144 lines total. Half of what it was.** No NMS. No sigmoid. No argmax. Just a confidence filter and a coordinate format conversion.

Export is a one-liner:

```bash
yolo export model=yolo26n.pt format=onnx
```

The cleaner ONNX graph (no DFL, no NMS ops) also makes TensorRT conversion more reliable.

**Lesson:** keep an eye on model releases even when you don't think you need them. A better model can let you delete infrastructure code.

### Deep Dive: How the One-to-One Head Actually Works

YOLO11 and earlier used a one-to-many assignment: during training, one ground-truth box can be assigned to multiple anchor predictions, and the model learns to predict many candidate boxes that overlap the same object. NMS at inference cleans up the duplicates.

YOLO26 uses a **one-to-one head trained with bipartite matching**, directly inspired by DETR (the Detection Transformer). The loss function runs the Hungarian algorithm to compute an optimal assignment between predictions and ground truths — each GT matches exactly one prediction. The model learns to suppress its own duplicates because duplicating is penalized by the matching loss.

Key differences from DETR:
- DETR uses learned object queries and cross-attention; slow to train, hundreds of epochs
- YOLO26 retains the CNN backbone + anchor grid, just changes the training assignment
- Convergence is much faster (comparable to YOLOv8/11 training schedules)
- Inference latency profile similar to YOLOv8/11 — no transformer overhead

Other changes in YOLO26 vs YOLO11:

| Change | Why it matters |
|---|---|
| **DFL (Distribution Focal Loss) removed** | DFL represents bbox coordinates as probability distributions over discrete bins. Accurate but adds a softmax + integral at inference. Removing it simplifies the graph and speeds up CPU paths. |
| **Small-target-aware label assignment (STAL)** | Weights small-object loss more heavily during training. Improves COCO mAP on small objects by ~1.5 points. |
| **Progressive loss balancing (ProgLoss)** | Reweights classification vs. bbox vs. objectness losses through training. Better convergence. |
| **MuSGD optimizer** | Ultralytics' custom SGD variant. Minor training stability improvement. |
| **43% faster CPU inference** | Claimed by Ultralytics. We can't verify directly (we run on GPU) but the simpler graph matches the story. |

### Deep Dive: YOLO26 ONNX to TensorRT Conversion

The export command with defaults gives you `[batch, 300, 6]` end-to-end output. Conversion to TensorRT:

```bash
trtexec \
    --onnx=yolo26n.onnx \
    --saveEngine=yolo26n_b2_fp16.engine \
    --fp16 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:2x3x640x640 \
    --maxShapes=images:2x3x640x640 \
    --memPoolSize=workspace:2048M
```

Notes:

- **`--fp16`** enables FP16 kernels where supported. On Orin Nano, most CNN layers have FP16 kernels, accuracy loss is <0.3 mAP vs FP32
- **Dynamic batch** — min/opt/max shapes let the engine handle batch 1 and batch 2 in one engine. TensorRT picks the right kernels per shape at inference time
- **`--memPoolSize=workspace:2048M`** — the workspace is scratch memory for tactic selection and intermediate activations. Default is often fine, but on Orin Nano's limited memory you may want to cap it explicitly
- **SM version coupling** — the built engine is tied to SM 8.7 (Orin) and TensorRT 10.3. Deploying to a different GPU or TRT version requires rebuilding

In our project the first start builds the engine from ONNX at runtime (~5-8 minutes on Nano) and caches it to disk for subsequent starts. See `DetectorEngine.swift` around the `loadedEngine.save(to:)` call.

### Deep Dive: Why the Nano Variant

YOLO26 variants:

| Variant | Params | FLOPs (640²) | COCO mAP | FP16 latency on Orin Nano (est.) |
|---|---|---|---|---|
| n (nano) | 2.6M | 1.5 GFLOPs | 37.5% | ~28 ms ← we use this |
| s (small) | 9.4M | 6.5 GFLOPs | 44.9% | ~100 ms |
| m (medium) | 20.1M | 22.1 GFLOPs | 50.4% | ~250 ms |
| l (large) | 25.3M | 56.4 GFLOPs | 52.5% | ~500 ms |
| x (xlarge) | 68.2M | 197.1 GFLOPs | 54.7% | ~1800 ms |

Orin Nano's integrated Ampere GPU does ~5 TFLOPS FP16. For a 20 FPS target (50 ms budget per frame), only the nano and small variants are candidates. We chose nano to leave headroom for the tracker, rendering, JPEG encoding, HTTP serving, and occasional VLM calls — all sharing the same SoC.

If you need higher accuracy, options include:

- Upgrade to Orin NX (16 GB, ~10 TFLOPS FP16) or AGX Orin (32/64 GB, ~40+ TFLOPS FP16) — `small` or `medium` become feasible
- Use INT8 quantization with a calibration dataset — 1.5-2× speedup from FP16, ~0.5-1 mAP accuracy loss
- Temporal ensembling — cache detections across frames for occluded objects; cheaper than running a bigger model
- Region-of-interest cropping — run the detector only on parts of the frame where motion was detected (frame differencing, background subtraction)

---

## Part 2: The VLM Second Stage

The detector finds cars. But "car" isn't useful — a human looking at a surveillance feed wants to know *which* car.

### The architecture

When the tracker prunes a confirmed track (the object has left the scene), we crop the best frame (highest-confidence bbox) and POST it to a local VLM sidecar:

```swift
// TrackFinalizer.swift
func submit(track: FinalizedTrack, frame: Frame, labels: [String]) {
    guard let jpegCrop = cropAndEncode(frame: frame, bbox: track.bestBBox) else { return }
    let label = labels[track.classId]
    let prompt = promptForClass(label)  // class-specific prompts

    let request = PendingDescription(
        trackId: track.id,
        label: label,
        jpegData: jpegCrop,
        prompt: prompt
    )

    // Bounded queue with backpressure — drop oldest if VLM can't keep up.
    if pending.count >= maxPending { pending.removeFirst() }
    pending.append(request)

    if processingTask == nil {
        processingTask = Task { await processQueue() }
    }
}
```

The VLM client is ~220 lines of Swift talking to llama.cpp's OpenAI-compatible `/v1/chat/completions` endpoint via `async-http-client`. Images are base64'd into a `image_url` content part. Class-specific prompts bias the model toward useful details:

```swift
private func promptForClass(_ label: String) -> String {
    switch label {
    case "person":
        return "Describe this person briefly: what they are wearing, "
             + "what they are doing, and any notable features. One sentence."
    case "car", "truck", "bus":
        return "Describe this vehicle briefly: color, type, make if "
             + "identifiable, and any notable features. One sentence."
    default:
        return "Describe what you see in this image briefly. One sentence."
    }
}
```

### Why llama.cpp, not TensorRT-LLM or vLLM

On an Orin Nano 8GB, the choice was clear after some research:

| Runtime | Verdict |
|---|---|
| **TensorRT-LLM** | Officially: no Nano support. The v0.12-jetson branch can't cross-build (needs ~16GB RAM just to compile), and the Jetson branch has no Qwen3-VL support. |
| **vLLM** | Runs, but PagedAttention consumes 6.4 GB on unified memory. OOM risk with any concurrent CUDA work. Needs `--enforce-eager` to stay alive, which kills throughput. |
| **llama.cpp** | ~2.2 GB for Qwen3-VL-2B Q4_K_M. Coexists with the YOLO TensorRT engine (~200 MB) with plenty of headroom. ~23.6 tok/s decode. |

We ship it as a Docker sidecar:

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

**Gotcha:** The `dustynv` image's bundled `llama-cpp-python 0.3.8` was too old for Qwen3-VL. We had to rebuild `llama-server` from source (b8748, `-DCMAKE_CUDA_ARCHITECTURES=87` for Orin's SM 8.7) and bind-mount it into the container.

### GPU arbitration

TensorRT kernels aren't preemptible. The llama.cpp sidecar has its own CUDA context. Both share unified LPDDR5. We avoid collisions by:

1. Running llama.cpp as a separate process (own CUDA context, own memory pool)
2. Bounding the VLM request queue in `TrackFinalizer` (10 max, drop oldest)
3. VLM requests are event-triggered (on track finalize), not per-frame

YOLO runs at 19.4 FPS without interference. The VLM processes ~0.5 queries/sec in the background. At this cadence, decode happens during natural gaps in the YOLO workload and we never see a frame drop.

### Deep Dive: Qwen3-VL-2B Architecture

Qwen3-VL is a multimodal LLM from Alibaba. The 2B variant has:

- **Vision encoder (ViT)** — ~300M params, 16x16 patch encoder trained on image-text pairs
- **MMProj (multimodal projector)** — a small MLP that projects ViT embeddings into the text token embedding space. Stored separately in GGUF as `mmproj-F16.gguf`
- **Language model** — ~1.7B param decoder-only Transformer with grouped-query attention
- **Context length** — 32k tokens at training time, usable up to 8k+ with Q4_K_M on an Orin Nano

Q4_K_M is one of llama.cpp's k-quant schemes: 4 bits per weight for most weights, 6 bits for the most important rows (detected by activation magnitude). Size reduction ~4× vs FP16 with measured perplexity loss under 2%.

**Runtime path through llama.cpp:**

```
HTTP POST /v1/chat/completions
  → JSON parse: extract image URL, prompt
  → base64 decode to JPEG bytes
  → libclip.so: decode JPEG → RGB → resize → ViT forward pass → patch embeddings
  → MMProj: project patch embeddings → language token embeddings
  → Prefill: language model processes the image tokens + text prompt
  → Decode loop: generate response tokens one at a time
  → Stream or batch return as OpenAI-format response
```

On Orin Nano we measured:

- Image encode + prefill: ~190 ms for a 48×48 test image (larger images take longer but not dramatically so — ViT has fixed patch count once resized)
- Decode: ~42 ms per token (~23.6 tok/s)
- Typical car description: 20-30 tokens → ~1-1.3 s total

Memory: ~1.0 GB for the LM weights + ~0.8 GB for the mmproj + ~0.4 GB for KV cache at 4k context = ~2.2 GB total.

### Deep Dive: Why Not TensorRT-LLM or vLLM (the honest version)

**TensorRT-LLM on Jetson** is real but painful:

1. Official NVIDIA support for Jetson is only for AGX Orin (32/64 GB). The `v0.12.0-jetson` branch exists for other Jetsons but NVIDIA explicitly disclaims support
2. Engine build requires roughly 4× the model size in RAM. For a 2B FP16 model that's ~16 GB — doesn't fit on an 8 GB Nano. You must cross-build on a workstation and copy the engine
3. The v0.12-jetson branch does NOT support Qwen2-VL or Qwen3-VL. Newer branches (v0.17+) add Qwen2.5-VL via a PyTorch path but have no verified Jetson container
4. Framework overhead: the TRT-LLM container image is 18.5 GB. At runtime, NCCL + cuDNN + framework buffers consume 4-5 GB before the model even loads
5. Reported throughput on AGX Orin 64 GB caps around 20 tok/s on 7B models before thermal throttling. Orin Nano would be worse

**vLLM on Jetson** is officially supported for JetPack 6 but has problems on unified memory:

1. PagedAttention aggressively allocates KV cache blocks using `--gpu-memory-utilization 0.9` by default. On Jetson, "GPU memory" is the same LPDDR5 as system RAM. vLLM can consume 6.4 GB vs llama.cpp's 4.2 GB for the same model
2. Open bug ([vllm #13131](https://github.com/vllm-project/vllm/issues/13131)) around unstable memory accounting on unified-memory systems (Jetson, GH200)
3. You have to set `--gpu-memory-utilization 0.65` and `--enforce-eager` (disables CUDA graphs) to stay alive — significantly cuts throughput
4. Coexistence with a running TensorRT YOLO engine is fragile; vLLM pre-allocates its pool at startup, leaving little headroom

**llama.cpp** sidesteps all this. It lazily allocates CUDA contexts, uses memory-mapped weights, and has explicit Qwen3-VL support in recent builds. The downside is that it's not the most feature-rich — no continuous batching, no speculative decoding, no FP8 — but we don't need those features for event-triggered single-user queries.

### Deep Dive: The `dustynv` Image Gotcha

We tried three base images before landing on one that worked:

1. **`dustynv/llama_cpp:r36.4.0`** — pulled successfully but bundled an old llama-cpp-python that refused to load the Qwen3-VL model (the `qwen3vl` architecture name wasn't recognized)
2. **`dustynv/llama_cpp:0.3.8-r36.4.0-cu128-24.04`** — same llama-cpp-python version, same issue
3. **`dustynv/llama_cpp:b5283-r36.4-cu128-24.04`** — had the right CUDA + container foundation, but the bundled `libllama.so` was from the llama.cpp snapshot at commit `b5283`, which was old enough that the Python `Llama` wrapper rejected Qwen3-VL models even though the C++ core actually supported it

The fix was to use image #3 as the base but compile `llama-server` from source inside the container:

```bash
docker exec -it llama-vlm bash
cd /opt && git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout b8748 # recent tag with stable Qwen3-VL support
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build --target llama-server -j4
cp build/bin/llama-server /root/llama-server   # copy out to host via bind mount
```

Then restart the container with the host's `llama-server` bind-mounted at `/usr/local/bin/llama-server`. `CMAKE_CUDA_ARCHITECTURES=87` targets Orin's compute capability 8.7 (Ampere-based, not Ada). Worth knowing if you build custom CUDA code for Orin.

### Deep Dive: Prompt Engineering for the VLM

The default llama.cpp Qwen3-VL chat template generates verbose, often hedging responses. A generic "describe this image" prompt returns things like:

> "Based on what I can see in this image, there appears to be a vehicle that looks like it could be a sedan or possibly an SUV, though the angle makes it somewhat difficult to determine with certainty..."

That's unusable for a structured dashboard. We tighten it two ways:

1. **Class-specific prompts.** The tracker knows the YOLO class. A car prompt asks specifically for color, type, make if identifiable. A person prompt asks about clothing and activity. See `TrackFinalizer.swift` → `promptForClass`
2. **One-sentence constraint.** The phrase "Respond in one sentence." reduces hedging and stops the model from speculating about scene context

Result: descriptions like

> "A white, four-door SUV is parked on a wet street, with a green mailbox and a yellow crosswalk line visible nearby."
> "This is a red 2004 Toyota Corolla, a compact hatchback with a sleek design and a well-known Toyota emblem on the front grille."

Are the makes and years accurate? **Sometimes.** The VLM is confident but not always correct. For a production use case you'd want to cross-reference against an ALPR (license plate reader) result or at least not trust single identifications. For the blog post, the confident-looking descriptions are good enough to make the point.

---

## Part 3: Native Cross-Compile — From 25 Min to 85 Seconds

The first version of this project built through Docker BuildKit with `--platform=linux/arm64`. BuildKit uses QEMU user-mode emulation to run the Swift compiler under emulated aarch64. Every `swift build` step took **~1500 seconds (25 minutes)**. Every fix cycle was 25 minutes.

### The wendyos Swift SDK

The wendy team ships a Swift SDK artifact bundle (`6.2.3-RELEASE_wendyos_aarch64`) that contains an aarch64 Linux sysroot. Swift's cross-compile support lets you use it via:

```bash
swift build --swift-sdk 6.2.3-RELEASE_wendyos_aarch64 -c release
```

This runs the Swift compiler natively on x86_64 (fast) and emits aarch64 object files directly — no QEMU, no emulation. The sysroot provides system headers and `.so` symlinks for linking.

Expected: fast builds. Actual: several hours of debugging. Here's the hall of fame.

### Gotcha 1: The SDK was missing half the sysroot

The shipped SDK had FFmpeg and TurboJPEG dev headers, but **no GStreamer, no DeepStream, no CUDA toolkit**. The TensorRT Swift bindings hardcode `-I/usr/local/cuda/include` in their `cxxSettings.unsafeFlags` — that's an absolute path on the build host, which doesn't exist on our x86 builder.

The fix was to overlay the missing pieces from the pre-built Jetson sysroot at `/home/mihai/jetson/Linux_for_Tegra/wendyos-sysroot-jetson-20251126/`:

```bash
SDK=~/.swiftpm/swift-sdks/6.2.3-RELEASE_wendyos_aarch64.artifactbundle/6.2.3-RELEASE_wendyos_aarch64/aarch64-unknown-linux-gnu/debian-bookworm.sdk
L4T=/home/mihai/jetson/Linux_for_Tegra/wendyos-sysroot-jetson-20251126
ROOTFS=/home/mihai/jetson/Linux_for_Tegra/rootfs

# GStreamer + GLib headers & libs
rsync -a --ignore-existing $L4T/usr/include/gstreamer-1.0/ $SDK/usr/include/gstreamer-1.0/
rsync -a --ignore-existing $L4T/usr/include/glib-2.0/      $SDK/usr/include/glib-2.0/
rsync -a --ignore-existing $L4T/usr/lib/aarch64-linux-gnu/libgst*.so* $SDK/usr/lib/aarch64-linux-gnu/
rsync -a --ignore-existing $L4T/usr/lib/aarch64-linux-gnu/libglib*.so* $SDK/usr/lib/aarch64-linux-gnu/
rsync -a --ignore-existing $L4T/usr/lib/aarch64-linux-gnu/libgobject*.so* $SDK/usr/lib/aarch64-linux-gnu/

# CUDA 12.6 toolkit (2.1 GB of headers + stubs)
rsync -a --ignore-existing $L4T/usr/local/cuda-12.6/targets/aarch64-linux/include/ \
    $SDK/usr/local/cuda-12.6/targets/aarch64-linux/include/
ln -sfn cuda-12.6 $SDK/usr/local/cuda
ln -sfn targets/aarch64-linux/include $SDK/usr/local/cuda-12.6/include

# DeepStream / NvBufSurface headers from the L4T rootfs
rsync -a --ignore-existing $ROOTFS/opt/nvidia/deepstream/deepstream-7.1/sources/includes/ \
    $SDK/opt/nvidia/deepstream/deepstream-7.1/sources/includes/
```

### Gotcha 2: The cross-linker wasn't cross-aware

With all headers in place, Swift compiled 471 out of 473 build tasks successfully. Then the link step exploded with:

```
ld.lld: error: .../Scrt1.o is incompatible with elf64-x86-64
```

Swift was cross-compiling aarch64 object files (correct) but invoking `clang` as the linker driver **without telling it the target architecture**. Clang defaulted to the host (x86_64) and passed `-m elf_x86_64` to `ld.lld`.

The fix lives in `toolset.json` in the SDK bundle:

```json
{
  "cCompiler": {
    "extraCLIOptions": [
      "--target=aarch64-unknown-linux-gnu",
      "-isystem", "<SDK>/usr/local/cuda/include",
      "-isystem", "<SDK>/usr/include/gstreamer-1.0",
      "-isystem", "<SDK>/usr/include/glib-2.0",
      "-isystem", "<SDK>/usr/lib/aarch64-linux-gnu/glib-2.0/include",
      "-isystem", "<SDK>/usr/include/aarch64-linux-gnu"
    ]
  },
  "cxxCompiler": {
    "extraCLIOptions": [
      "-lstdc++",
      "--target=aarch64-unknown-linux-gnu",
      "-isystem", "<SDK>/usr/local/cuda/include",
      ...
    ]
  },
  "swiftCompiler": {
    "extraCLIOptions": [
      "-Xlinker", "-R/usr/lib/swift/linux/",
      "-use-ld=lld",
      "-Xclang-linker", "--target=aarch64-unknown-linux-gnu"
    ]
  }
}
```

Key insights:

1. `cCompiler.extraCLIOptions` goes to clang when used as a C compiler — needs `--target=` to cross-compile
2. `swiftCompiler.extraCLIOptions` goes to swiftc, which forwards `-Xclang-linker` args to clang when invoking the linker — needs `--target=` again because swift's link step is a separate clang invocation
3. **Don't add `--target=` to `linker.extraCLIOptions`** — those go to `ld.lld` directly, which doesn't take that flag (it chooses the target based on the input objects, or we're supposed to tell clang-as-linker-driver to pass `-m aarch64linux`).
4. System headers for GStreamer/GLib/CUDA baked in via `-isystem` — we tried `pkgConfig:` in the Swift targets first, but SwiftPM's pkg-config resolution is flaky under cross-compile and not worth the fight. Bake the paths and move on.

### Gotcha 3: `PKG_CONFIG_LIBDIR` vs `PKG_CONFIG_PATH`

When we still had `pkgConfig:` in `Package.swift`, SwiftPM would get host x86 library paths back from the build host's pkg-config, poisoning the linker command with `-L/usr/lib/x86_64-linux-gnu`. The fix:

```bash
export PKG_CONFIG_LIBDIR="$SDK/usr/lib/aarch64-linux-gnu/pkgconfig:$SDK/usr/lib/pkgconfig:$SDK/usr/share/pkgconfig"
export PKG_CONFIG_SYSROOT_DIR="$SDK"
```

`PKG_CONFIG_LIBDIR` (not `_PATH`) **replaces** the default search list — `_PATH` only appends. `_SYSROOT_DIR` prepends the sysroot to all returned paths.

This worked reliably from the shell but was inconsistent from inside SwiftPM. We eventually gave up on `pkgConfig:` entirely and baked paths into `toolset.json`.

### Result

```
Build complete! (85.06s)
```

**17.6× faster.** Every iteration after that was 3-5 seconds for incremental builds.

### Deep Dive: What's Actually in a Swift SDK Artifact Bundle

Swift's cross-compile support uses a format called an **artifact bundle**. Structure:

```
6.2.3-RELEASE_wendyos_aarch64.artifactbundle/
├── info.json                           (metadata: versions, supported triples)
└── 6.2.3-RELEASE_wendyos_aarch64/      (the artifact)
    └── aarch64-unknown-linux-gnu/      (target triple)
        ├── swift-sdk.json              (SDK config: which toolset, where's the sysroot)
        ├── toolset.json                (C/C++/Swift compiler & linker extra args)
        └── debian-bookworm.sdk/        (the actual sysroot)
            ├── usr/include/            (headers — glibc, Linux, libraries)
            ├── usr/lib/                (libraries, .so symlinks)
            ├── usr/lib/swift/          (Swift runtime .so files)
            ├── usr/local/cuda/         (CUDA toolkit — we added this)
            └── ...
```

- `info.json` lists the artifact ID (`6.2.3-RELEASE_wendyos_aarch64`) and supported host triples (x86_64 Linux in our case)
- `swift-sdk.json` tells swift-driver where to find the sysroot (`debian-bookworm.sdk`) and how to locate the Swift runtime within it (`debian-bookworm.sdk/usr/lib/swift`)
- `toolset.json` is the extensibility hook — you can inject extra flags for each compiler

Swift's cross-compile support was originally designed for musl-static targets (one sysroot, no external deps). We're using it for glibc-dynamic with heavy NVIDIA dependencies — the toolset.json extension points were essential. This is arguably an abuse of the system but it works.

### Deep Dive: toolset.json Schema and Why Each Flag Is There

```json
{
  "schemaVersion": "1.0",
  "cCompiler": {
    "extraCLIOptions": [
      "--target=aarch64-unknown-linux-gnu",
      "-isystem", "<SDK>/usr/local/cuda/include",
      "-isystem", "<SDK>/usr/include/gstreamer-1.0",
      "-isystem", "<SDK>/usr/include/glib-2.0",
      "-isystem", "<SDK>/usr/lib/aarch64-linux-gnu/glib-2.0/include",
      "-isystem", "<SDK>/usr/include/aarch64-linux-gnu"
    ]
  },
  "cxxCompiler": { /* same flags + `-lstdc++` */ },
  "swiftCompiler": {
    "extraCLIOptions": [
      "-Xlinker", "-R/usr/lib/swift/linux/",
      "-use-ld=lld",
      "-Xclang-linker", "--target=aarch64-unknown-linux-gnu"
    ]
  }
}
```

Each flag has a specific job:

| Flag | Who sees it | What it does |
|---|---|---|
| `--target=aarch64-unknown-linux-gnu` (in cCompiler) | clang when called to compile C code | Sets the target triple. Without this, clang defaults to the host (x86_64) and emits x86 code |
| `-isystem <path>` (in cCompiler) | clang's header search | Adds system include paths for headers that aren't discovered via pkg-config |
| `-Xlinker -R/usr/lib/swift/linux/` | swiftc | Sets RPATH so the runtime linker finds Swift libraries at the install location on the device |
| `-use-ld=lld` | swiftc | Use LLVM's lld linker instead of GNU ld or gold. lld is multi-arch aware |
| `-Xclang-linker --target=...` | swiftc → clang (when clang is invoked as a linker driver) | When swift calls clang to invoke lld, this tells clang what target to emit link commands for. Separate from the cCompiler target because it's a second clang invocation |

Notably absent: anything in a `linker` section. We tried that first and `ld.lld` rejected `--target=` as an unknown argument. The linker config is not the right place for target-triple info because swift-driver invokes clang as the linker driver, and clang-as-linker takes its own `-Xclang-linker` args.

### Deep Dive: Why Absolute Paths in toolset.json

The `-isystem` paths in toolset.json are absolute (baked to `$HOME/.swiftpm/.../debian-bookworm.sdk/usr/include/...`). This is ugly for a portable SDK — every developer would need to re-generate toolset.json for their machine.

Better approach for production:

- Use relative paths with a special token. Swift's SDK format supports `<SDK_ROOT>` in some contexts; we'd need to check the schema version
- Ship a script that generates toolset.json based on a template at SDK install time
- Or use a wrapper script around `swift build` that sets env vars like `CPATH` to point at the SDK paths

For this blog post we chose to document the problem and leave the absolute paths. The wendy team's upstream solution for `meta-wendyos-jetson` should be to bake the paths into the Yocto-built SDK with a relative reference.

### Deep Dive: SwiftPM's pkgConfig Is Not Cross-Compile Safe

The `pkgConfig:` argument on a `systemLibrary` target tells SwiftPM to invoke pkg-config during dependency resolution and add the returned flags to the target's build command.

On a host-native build, this works fine: pkg-config reads host `.pc` files and returns paths like `/usr/include/gstreamer-1.0` and `-L/usr/lib/x86_64-linux-gnu -lgstreamer-1.0`.

On a cross-build, we want pkg-config to query the target sysroot's `.pc` files. The pkg-config env vars for this are:

- `PKG_CONFIG_LIBDIR` — replace the default search list (not append; _PATH appends)
- `PKG_CONFIG_SYSROOT_DIR` — prepend to all paths in the resolved `.pc` files
- `PKG_CONFIG_PATH` — additional directories (appends, rarely what you want for cross)

We set `PKG_CONFIG_LIBDIR` and `PKG_CONFIG_SYSROOT_DIR` correctly, and pkg-config on the command line returned the right SDK-relative paths. But SwiftPM's behavior was inconsistent — sometimes it used our env, sometimes it didn't. Specifically: after `rm -rf .swiftpm` (clearing SwiftPM's package cache) the next resolve would produce a manifest with **empty** include paths for the system library targets.

We traced this enough to know SwiftPM runs pkg-config internally during manifest resolution and caches results at `.swiftpm/cache/...`. The cache lookup may pre-empt the env vars. It's also possible that Swift's own implementation of pkg-config parsing is different from the `pkg-config(1)` binary and doesn't fully honor `PKG_CONFIG_LIBDIR`. We didn't dig deep enough to be sure.

The fix was to give up on `pkgConfig:` entirely and bake the paths into toolset.json. Now the system library targets in `Package.swift` are minimal:

```swift
.systemLibrary(
    name: "CGStreamer",
    providers: [
        .apt(["libgstreamer1.0-dev", "libgstreamer-plugins-base1.0-dev"]),
    ]
),
```

No `pkgConfig:`. The header search paths come from the SDK's toolset.json, the linker flags come from the module.modulemap's `link` directives:

```
// Sources/CGStreamer/module.modulemap
module CGStreamer [system] {
    header "shim.h"
    link "gstreamer-1.0"
    link "gstapp-1.0"
    link "gobject-2.0"
    link "glib-2.0"
    export *
}
```

This is more static and less flexible but **actually reliable**.

### Deep Dive: How the Swift Driver Invokes Clang

A cross-compile `swift build` does roughly:

1. **swift-package-manager** parses the manifest (`Package.swift`), resolves dependencies, and generates a build plan
2. For each Swift target: invokes **swiftc** (the Swift compiler frontend driver)
3. swiftc emits LLVM IR → aarch64 object files via the bundled LLVM backend. This step natively knows the target from `-target aarch64-unknown-linux-gnu` — no issue here
4. For each C/C++ system library target: invokes **clang** with the sysroot set and the compiler flags from toolset.json. Clang emits aarch64 object files. This step needs `--target=aarch64-unknown-linux-gnu` explicitly in `cCompiler.extraCLIOptions`, otherwise clang defaults to host x86_64
5. For the final link step: invokes **swiftc** again, which internally invokes **clang-as-linker-driver** (because of `-use-ld=lld`). Clang-as-linker needs `--target=` via `-Xclang-linker`. Clang then invokes `ld.lld` with the correct `-m aarch64linux` emulation

Every one of those `--target=` placements took an iteration to get right. The error from step 5 was the worst because it looked like a sysroot bug (`.o is incompatible with elf64-x86-64`) when it was actually a missing flag.

### The binary verifies as a real aarch64 ELF:

```bash
$ file .build/aarch64-unknown-linux-gnu/release/Detector
ELF 64-bit LSB pie executable, ARM aarch64, version 1 (SYSV),
dynamically linked, interpreter /lib/ld-linux-aarch64.so.1, ...

$ readelf -d .../Detector | grep NEEDED
 libturbojpeg.so.0
 libavformat.so.58
 libavcodec.so.58
 libavutil.so.56
 libswscale.so.5
 libgstreamer-1.0.so.0      ← GStreamer
 libgstapp-1.0.so.0
 libgobject-2.0.so.0
 libglib-2.0.so.0
 libnvinfer.so.10           ← TensorRT 10
 libnvinfer_plugin.so.10
 libcuda.so.1               ← CUDA driver
 libswiftCore.so
 ...
```

---

## Part 4: Delete the Dockerfile

`wendy run` has two code paths (see `wendy-agent/Sources/Wendy/cli/commands/BuildCommand.swift`):

1. **If a `Dockerfile` exists** → BuildKit + QEMU (slow, what we started with)
2. **If it's a Swift Package with no Dockerfile** → `swift package build-container-image` via the Swift Container Plugin

The second path does native cross-compile against the wendyos SDK, then packages the resulting binary into a minimal container image with `swift:6.2.3-slim` as the base — no Docker build, no QEMU.

So: **delete the Dockerfile**, add `swift-container-plugin` to `Package.swift`, and `wendy run` just works:

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/wendylabsinc/tensorrt-swift", from: "0.0.1"),
    .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.6.0"),
    .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    .package(url: "https://github.com/apple/swift-log", from: "1.6.0"),
    .package(url: "https://github.com/swift-server/async-http-client", from: "1.23.0"),
    .package(url: "https://github.com/apple/swift-container-plugin.git", from: "1.3.0"),
],
```

The `wendy run` output goes from 25 minutes of BuildKit logs to "native compile + layer push" in under 2 minutes.

### Deep Dive: What swift-container-plugin Actually Does

The [swift-container-plugin](https://github.com/apple/swift-container-plugin) is an SPM (Swift Package Manager) command plugin that adds a `build-container-image` subcommand. When invoked:

1. **Runs `swift build`** — cross-compile your executable against whatever SDK is selected. The output is an aarch64 ELF at `.build/aarch64-unknown-linux-gnu/release/<product>`
2. **Constructs an OCI image** using a base layer — by default `swift:X.Y-slim` from Docker Hub, but you can override with `--from`. The base provides the Swift runtime libraries (`libswiftCore.so`, etc.)
3. **Adds your binary as a new layer** — tarred up with the right file permissions and owners (root:root by default)
4. **Adds resource layers** if `--resources=host:container` flags are passed
5. **Sets OCI metadata** — `CMD`, `ENTRYPOINT`, `ENV`, `EXPOSE`, labels from args
6. **Pushes via the OCI distribution protocol** — HTTPS by default, HTTP with `--allow-insecure-http=destination`

The critical trick: **no Docker daemon involved.** swift-container-plugin implements the OCI image spec and registry protocol directly in Swift. It's independent of the Docker build machinery. That's why we don't need QEMU, Docker buildx, or even a running Docker daemon on the build host.

### Deep Dive: OCI Image Layering and Cache Behavior

The produced image has roughly these layers (varies with plugin version):

1. **Base image layers** from `swift:6.2.3-jammy` — Ubuntu 22.04 + Swift 6.2 runtime. Cached after first pull
2. **Resource layers** — one per `--resources` flag. Small files like `labels.txt`, `streams.json`, `yolo26n.onnx` (~10 MB)
3. **Binary layer** — the compiled `Detector` executable + debug info (~80 MB)
4. **Metadata layer** — OCI config JSON with env vars, cmd, labels

Wendy's build flow pushes to the Jetson's local insecure registry at `<device>:5000`. Containerd on the device pulls from that registry. The first deploy uploads all layers (~300 MB). Subsequent deploys with code changes only push the binary layer + metadata; the base and resources layers are cached.

### Deep Dive: Why `swift:6.2.3-jammy` as the Base

Three criteria:

1. **Distribution match** — our cross-compile SDK's sysroot is `debian-bookworm.sdk` but the L4T libraries it overlays (FFmpeg, turbojpeg) are Ubuntu 22.04 (jammy) ABI. Targeting the jammy runtime means the `libavformat.so.58` we linked at build time exists at runtime
2. **Architecture** — the `swift:*-jammy` tags are multi-arch. `docker pull` on an aarch64 machine gets the aarch64 variant automatically
3. **Swift runtime** — includes `libswiftCore.so`, `libFoundation.so`, `libDispatch.so`, etc. — the dynamic dependencies our binary needs but that aren't in a minimal Ubuntu base

Alternative we rejected: `FROM scratch` with all Swift runtime libs manually COPYed. Simpler image, more work to maintain. The `swift:*-jammy` base gives us a known-good runtime environment with minimal hassle.

### Deep Dive: Wendy's Build-Path Decision Tree

Reading `wendy-agent/Sources/Wendy/cli/commands/BuildCommand.swift`:

```swift
let isSwiftPackage = FileManager.default.fileExists(atPath: "Package.swift")

for item in directory where isDockerfile(item) {
    try await withBuiltDockerfileApp(/* ... */)
    return  // Dockerfile wins, exit early
}

if isSwiftPackage {
    // Native cross-compile path via swift-container-plugin
    try await withBuiltSwiftApp(/* ... */)
} else {
    // Python project without Dockerfile — offer to generate one
    try await generatePythonDockerfileAndBuild()
}
```

The decision is **Dockerfile first, Swift second**. Having a Dockerfile forces the Docker path even if a Swift Package is present. We initially had both, which is why our builds went the slow route. Deleting the Dockerfile was the actual unlock.

### Deep Dive: The Actual `wendy run` Sequence

Once the Dockerfile is gone:

1. Wendy detects `Package.swift` → Swift path
2. Verifies `swift-container-plugin` is a dependency of the package (required)
3. Ensures the swiftly toolchain and the wendyos SDK are installed; offers to install if missing
4. Invokes:
   ```
   swift package --swift-sdk 6.2.3-RELEASE_wendyos_aarch64 \
     --allow-network-connections=all \
     build-container-image \
     --from=swift:6.2.3-slim \
     --allow-insecure-http=destination \
     --product=Detector \
     --repository=<device>:5000/detector \
     --architecture=arm64 \
     --resources=labels.txt:/app/labels.txt \
     --resources=streams.json:/app/streams.json \
     --resources=yolo26n.onnx:/app/yolo26n.onnx \
     --cmd /app/Detector \
     --env LD_LIBRARY_PATH=... \
     --env GST_PLUGIN_PATH=...
   ```
5. Waits for the push to complete (<2 min usually)
6. Sends the wendy-agent on the device a gRPC call telling it to start/restart the container with the right entitlements (gpu, host network)
7. wendy-agent uses containerd to pull the new image tag and launch a new container

The build-time env vars (PKG_CONFIG_*, PATH pointing at swiftly) still need to be set in the shell where you run `wendy run` because swift-package-manager spawns its subprocess from that environment.

---

## Part 5: GStreamer via C Interop

The original Swift detector shelled out to `ffmpeg` for RTSP decoding. That works but:

- Software H.264 decode burns a full CPU core per stream
- Pipe boundary: raw RGB24 frames (~6 MB at 1080p) get copied between processes every frame
- No access to NVIDIA's NVDEC hardware decoder

On Jetson, the right decoder is `nvv4l2decoder` (a GStreamer element that wraps NVDEC via a userspace V4L2 shim). To drive it from Swift, we wrap GStreamer's C API directly.

### The system library target

```swift
// Package.swift
.systemLibrary(
    name: "CGStreamer",
    providers: [
        .apt(["libgstreamer1.0-dev", "libgstreamer-plugins-base1.0-dev"]),
    ]
),
```

```
// Sources/CGStreamer/module.modulemap
module CGStreamer [system] {
    header "shim.h"
    link "gstreamer-1.0"
    link "gstapp-1.0"
    link "gobject-2.0"
    link "glib-2.0"
    export *
}
```

### The C shim

GStreamer uses GObject, which means type casts go through macros like `GST_BIN(element)` that expand to `G_TYPE_CHECK_INSTANCE_CAST`. Swift's C importer doesn't handle those macros. A tiny C shim in `shim.h` bridges the gap:

```c
static inline GstBin *wendy_gst_bin_cast(GstElement *e) {
    return GST_BIN(e);
}

static inline GstAppSink *wendy_gst_app_sink_cast(GstElement *e) {
    return GST_APP_SINK(e);
}

// Pull a sample, map the buffer, and return the raw bytes through out-params.
// Caller must release the handle with wendy_gst_release_sample().
static inline int wendy_gst_pull_sample(
    GstAppSink *sink, void **out_sample, void **out_data, size_t *out_size)
{
    GstSample *sample = gst_app_sink_pull_sample(sink);
    if (!sample) return 0;
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    typedef struct { GstSample *sample; GstMapInfo map; } SampleHandle;
    SampleHandle *h = g_malloc(sizeof(SampleHandle));
    h->sample = sample;
    if (!gst_buffer_map(buffer, &h->map, GST_MAP_READ)) { ... return 0; }
    *out_sample = h;
    *out_data   = h->map.data;
    *out_size   = h->map.size;
    return 1;
}
```

### The Swift pipeline

```swift
// GStreamerFrameReader.swift (actor)
actor GStreamerFrameReader {
    private var pipeline: UnsafeMutablePointer<GstElement>?
    private var appsink:  UnsafeMutablePointer<GstAppSink>?

    func start() throws {
        gst_init(nil, nil)

        let pipelineString = """
        rtspsrc location=\(url) latency=200 protocols=tcp ! \
        rtph264depay ! h264parse ! \
        nvv4l2decoder ! \
        nvvideoconvert ! \
        video/x-raw,format=RGB,width=\(width),height=\(height) ! \
        appsink name=wendy_sink emit-signals=false sync=false max-buffers=2 drop=true
        """

        var error: UnsafeMutablePointer<GError>?
        guard let parsed = gst_parse_launch(pipelineString, &error) else {
            throw GStreamerError.parseFailed(...)
        }
        self.pipeline = parsed

        let bin = wendy_gst_bin_cast(parsed)
        guard let sinkElement = gst_bin_get_by_name(bin, "wendy_sink") else {
            throw GStreamerError.elementNotFound("wendy_sink")
        }
        self.appsink = wendy_gst_app_sink_cast(sinkElement)

        gst_element_set_state(parsed, GST_STATE_PLAYING)
    }

    func nextFrame() async throws -> Frame? {
        guard let sink = appsink else { throw GStreamerError.notStarted }

        // gst_app_sink_pull_sample is blocking — run on a dedicated dispatch
        // queue so we don't pin the cooperative thread pool.
        let sinkPtr = SendableGstPointer(sink)
        let bytes: [UInt8]? = await withCheckedContinuation { continuation in
            pullQueue.async {
                var sampleHandle: UnsafeMutableRawPointer?
                var dataPtr: UnsafeMutableRawPointer?
                var size: Int = 0
                let ok = wendy_gst_pull_sample(sinkPtr.pointer,
                                               &sampleHandle, &dataPtr, &size)
                if ok == 0 { continuation.resume(returning: nil); return }
                defer { wendy_gst_release_sample(sampleHandle) }
                let buf = UnsafeBufferPointer<UInt8>(
                    start: dataPtr!.bindMemory(to: UInt8.self, capacity: size),
                    count: size)
                continuation.resume(returning: Array(buf))
            }
        }
        guard let bytes else { return nil }
        return Frame(data: bytes, width: width, height: height)
    }
}
```

No GLib main loop, no signal callbacks, no `g_signal_connect` trampolines. Pull samples synchronously on a dedicated dispatch queue and hand the bytes back to the actor via a continuation. Structured concurrency all the way down.

### Deep Dive: GStreamer Pipeline Lifecycle

A GStreamer pipeline goes through explicit state transitions:

1. **NULL** — the pipeline is parsed but no resources are allocated. `gst_parse_launch` leaves us here
2. **READY** — resources allocated, elements linked, pads negotiated with static caps. No data flowing yet
3. **PAUSED** — pipeline is primed. For live sources, this is where `rtspsrc` connects to the camera and negotiates RTP caps. For file sources, the pipeline pre-rolls the first buffer
4. **PLAYING** — data is actively flowing through the pipeline

`gst_element_set_state(pipeline, GST_STATE_PLAYING)` is an **async** state transition. The call returns immediately with one of:

- `GST_STATE_CHANGE_SUCCESS` — the transition completed synchronously (rare for PLAYING)
- `GST_STATE_CHANGE_ASYNC` — in progress, check the bus for completion messages (typical)
- `GST_STATE_CHANGE_FAILURE` — something went wrong immediately (e.g., missing element)
- `GST_STATE_CHANGE_NO_PREROLL` — async for live sources, data will arrive whenever the camera is ready

We handle only the `FAILURE` case explicitly. For `ASYNC` we rely on the first `gst_app_sink_pull_sample` call to block until frames start flowing; if the RTSP connection fails, pull returns NULL and we report an error.

A production version would also attach a bus watch (`gst_bus_add_watch`) to catch error and warning messages from the pipeline bus, which carry more detail than the state-change return code. That requires a GLib main loop or explicit bus polling — either way, more code than we needed for the MVP.

### Deep Dive: Caps Negotiation and the RGB Output

The pipeline's `nvvideoconvert ! video/x-raw,format=RGB,width=1920,height=1080 ! appsink` clause isn't just cosmetic. That **caps filter** forces nvvideoconvert's output pad to negotiate to exactly RGB 1920×1080. Without the caps filter, nvvideoconvert might produce I420 or NV12 (its preferred output for upstream NVIDIA consumers), and appsink would pass that through unchanged — our Swift code would receive YUV instead of RGB and render the frame wrong.

Things to know about caps:

- `RGB` in GStreamer is **packed 24-bit RGB**, one byte per channel, stride = width × 3
- `RGBA` adds an alpha byte, stride = width × 4
- `NV12` is YUV420 semi-planar, Y plane + interleaved UV plane, stride = width for Y and width/2 for UV
- `I420` is YUV420 planar, Y + U + V in separate planes

We want RGB because our preprocessor expects packed RGB24 (matching what FFmpeg was producing). Using YUV would require a preprocessor rewrite.

### Deep Dive: GstBuffer, GstSample, GstMapInfo — Memory Contract

When appsink hands us a frame, the ownership chain is:

1. `gst_app_sink_pull_sample(sink)` returns a `GstSample *` — we own this reference and must `gst_sample_unref` it when done
2. `gst_sample_get_buffer(sample)` returns a borrowed `GstBuffer *` — do not unref separately; it's owned by the sample
3. `gst_buffer_map(buffer, &map, GST_MAP_READ)` fills `map.data` / `map.size` with pointers to the actual pixel bytes — borrowed, backed by the buffer
4. When done reading: `gst_buffer_unmap(buffer, &map)` then `gst_sample_unref(sample)` in that order

Our C shim wraps all this into a `SampleHandle` struct that Swift treats as an opaque `void*`. `wendy_gst_pull_sample` allocates the handle and sets up the mapping; `wendy_gst_release_sample` tears it down. Swift never touches the GObject ref count.

Why the indirection? Because `GstMapInfo` is a non-opaque C struct that Swift imports as a value type — and Swift structs moved across a `withCheckedContinuation` boundary don't share backing memory, which would corrupt the map state. The shim keeps the `GstMapInfo` in C-allocated memory and hands Swift a pointer to it.

### Deep Dive: Why We Don't Use Signals (`g_signal_connect`)

The "canonical" way to read from a GStreamer appsink is via the `new-sample` signal:

```c
g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), user_data);
```

The callback fires from the streaming thread whenever a sample is ready, enabling a pure event-driven design. We chose **synchronous pull** instead for several reasons:

1. **`g_signal_connect` is a macro** that expands to `g_signal_connect_data` with a cast to `GCallback`. Swift can't import the macro; you'd call `g_signal_connect_data` directly and cast the callback via `unsafeBitCast(callback, to: GCallback.self)`. The `@convention(c)` callback can't capture Swift context, so you'd pass `self` via the `gpointer data` parameter and cast back inside the callback. Doable but ugly
2. **GLib main loop** — in event-driven mode, many GStreamer operations assume a main loop is running on some thread. You'd need `g_main_loop_run()` on a dedicated pthread. That's one more lifecycle to manage
3. **Backpressure** — with callbacks, frames arrive at whatever rate the pipeline produces them. If your consumer is slow, you either drop frames manually or the pipeline queue backs up. With synchronous pull, backpressure is natural: you pull only when ready
4. **Structured concurrency fit** — pulling on a dispatch queue with a `withCheckedContinuation` fits Swift's async/await model cleanly. A signal callback would have to bridge out-of-band with a `CheckedContinuation` array or similar

The synchronous pull approach has one downside: it blocks one thread per stream. For dozens of streams on one device, the callback approach scales better. For our 1-2 streams per Jetson use case, synchronous pull is simpler and sufficient.

### Deep Dive: GStreamer Element Factory Lookup and Plugin Registry

When `gst_parse_launch` parses `rtspsrc ! rtph264depay ! ...`, it looks up each element name in the **plugin registry**. The registry is a database of all GStreamer plugins and elements known to the runtime. It's populated by scanning directories listed in:

- `GST_PLUGIN_PATH_1_0` / `GST_PLUGIN_PATH` environment variables
- Hard-coded default paths compiled into `libgstreamer-1.0.so`
- Paths added programmatically via `gst_registry_scan_path`

First-run scanning writes a cached registry to `~/.cache/gstreamer-1.0/registry.<arch>.bin`. Subsequent runs load from cache (fast) unless plugins have changed.

On our Jetson container:

- `GST_PLUGIN_PATH` is set by the CDI spec to include DeepStream paths
- Our Swift code calls `gst_registry_scan_path` programmatically on two extra paths (Bookworm plugins, NVIDIA passthrough)
- We disable the external `gst-plugin-scanner` binary (`GST_PLUGIN_SCANNER=""`) because the Ubuntu scanner binary is ABI-incompatible with the L4T GStreamer 1.22 core that CDI injects

The "pluginsFound=0" we saw during debugging was a red herring — `gst_registry_scan_path` returns the number of **new** plugins added, not the total. If the plugins are already in the cached registry, it returns 0 even though they're loaded.

---

## Part 6: The `LIBV4L2_PLUGIN_DIR` Discovery (a.k.a. "why is /dev/v4l2-nvdec a symlink to /dev/null?")

This one deserves its own section because it's the kind of rabbit hole that only makes sense once you're at the bottom of it.

**The symptom:** after wiring up GStreamer, the `nvv4l2decoder` element failed to initialize inside the container. We fell back to `avdec_h264` (software decode via FFmpeg's GStreamer plugin) — same CPU cost as the original subprocess. The whole point of the GStreamer refactor was to reach NVDEC, and we weren't reaching it.

**First investigation:** `/dev/v4l2-nvdec` was being mounted into the container, but `ls -la` inside the container showed... major 1, minor 3. That's `/dev/null`.

The knee-jerk conclusion was "WendyOS isn't exposing the device properly" or "the CDI spec is broken". So we asked: *does the real device node exist on the host?*

```bash
$ ssh root@jetson 'ls -la /dev/v4l2-nvdec; stat /dev/v4l2-nvdec'
crw-rw-rw- 1 root root 1, 3 Apr 10 14:30 /dev/v4l2-nvdec
```

Major 1, minor 3. **On the host too.** That's `/dev/null`, everywhere. Not a container mount bug.

**Second investigation:** is the NVDEC driver even loaded?

```bash
$ lsmod | grep -i nvdec
(nothing)
$ ls /sys/class/video4linux/
(empty)
$ grep -i nvdec /proc/devices
(nothing)
```

No `/dev/video*` node. No V4L2M2M device in `/proc/devices`. The platform device `15480000.nvdec` registers under `tegra_drm`, not `videodev`. At this point the obvious conclusion is: **the kernel doesn't expose NVDEC as a V4L2 device on this L4T branch.** We're done, hardware decode is impossible without a kernel/BSP change.

Except... the existing DeepStream pipelines on this same device do use `nvv4l2decoder`. And they work. How?

**The actual mechanism (it took embarrassingly long to find):**

```bash
$ cat /etc/udev/rules.d/99-tegra-devices.rules
# /dev/v4l2-nvdec — for upstream linux on Tegra platforms
ACTION=="add", SUBSYSTEM=="drm", KERNEL=="card0", \
    RUN+="/bin/mknod -m 666 /dev/v4l2-nvdec c 1 3"
```

The `/dev/null` device node is **intentional**. On JetPack 6 / L4T r36.x, `nvv4l2decoder` is not a kernel V4L2 driver — it's a **userspace V4L2 shim**:

```
GStreamer nvv4l2decoder
    ↓ opens /dev/v4l2-nvdec  (libv4l2 passes fstat — it's a char device, that's all it needs)
libv4l2.so  (intercepts every V4L2 ioctl before it reaches the kernel)
    ↓
libv4l2_nvvideocodec.so  ← the actual V4L2 plugin
    ↓
libtegrav4l2.so
    ↓
NvMMLite
    ↓
NVDEC hardware
```

The `/dev/null` node exists only so that `open()` returns a valid fd that `libv4l2` can hook. All the "V4L2 ioctls" are implemented in the userspace plugin; the kernel never sees them.

**Why was it failing in our container?** `libv4l2` looks for its plugins at a default path: `/usr/lib/aarch64-linux-gnu/libv4l/plugins/nv/`. The WendyOS CDI spec bind-mounts the NVIDIA `libv4l2_nvvideocodec.so` into the container, but at a different path (`/usr/share/nvidia-container-passthrough/usr/lib/aarch64-linux-gnu/libv4l/plugins/nv/`). Without `LIBV4L2_PLUGIN_DIR` set, libv4l2 doesn't find the plugin, silently falls through to kernel ioctls, hits `/dev/null`, and fails — which makes `nvv4l2decoder` look broken.

**The fix** is one line in the CDI spec:

```yaml
# /etc/cdi/nvidia.yaml
containerEdits:
  env:
    - LIBV4L2_PLUGIN_DIR=/usr/share/nvidia-container-passthrough/usr/lib/aarch64-linux-gnu/libv4l/plugins/nv
```

Injected via a sed patch in the WendyOS `fix-cdi-gstreamer-paths.sh` post-processing script. After `systemctl restart edgeos-cdi-generate.service`, fresh containers get the env var and `nvv4l2decoder` works.

**Takeaway:** when a device node looks wrong on Jetson, check the udev rules and the libv4l2 plugin dir before assuming kernel/BSP changes. The entire V4L2M2M decode API on L4T r36 is implemented in userspace. The `/dev/null` trick is not a bug.

### Deep Dive: libv4l2 Plugin Architecture

`libv4l2` (also called libv4lconvert) is a userspace library that wraps the Linux V4L2 kernel API. It was originally created to paper over differences between V4L2 driver implementations and to convert exotic pixel formats transparently. Modern distros ship it as `libv4l2.so.0`.

Beyond format conversion, libv4l2 has a **plugin mechanism**:

- At library init, libv4l2 scans `$LIBV4L2_PLUGIN_DIR` (default `/usr/lib/x86_64-linux-gnu/libv4l/plugins/` or `/usr/lib/aarch64-linux-gnu/libv4l/plugins/`)
- For each `.so` file found, it calls the plugin's `plugin_init` entry point
- Plugins register callback functions that intercept `open`, `close`, `ioctl`, `mmap`, `read`, `write` on specific device nodes

When an application calls `v4l2_open("/dev/v4l2-nvdec", ...)`:

1. libv4l2 internally calls `open` on the device node (gets a real fd, even if the node is /dev/null)
2. It iterates through registered plugins asking "do you own this fd?"
3. NVIDIA's `libv4l2_nvvideocodec.so` plugin says yes based on the filename or a probe ioctl
4. From then on, every `v4l2_ioctl` call on that fd is routed to the plugin's ioctl handler
5. The plugin decodes the ioctl intent (e.g., `VIDIOC_QBUF` = queue buffer for decode) and calls into `libtegrav4l2.so` which talks to NvMMLite which talks to the NVDEC hardware

The real kernel `/dev/null` fd is never used — every operation is intercepted before reaching the kernel. The fd is there just so `open` returns something valid.

GStreamer's `nvv4l2decoder` plugin:

- Loads `libgstnvvideo4linux2.so` (GStreamer plugin)
- Internally calls `v4l2_open("/dev/v4l2-nvdec")` via libv4l2
- libv4l2 finds `libv4l2_nvvideocodec.so` (if `LIBV4L2_PLUGIN_DIR` points at it), loads it, and all subsequent ioctls go through the userspace path
- If the plugin isn't found, libv4l2 passes ioctls through to the kernel `/dev/null`, which silently fails — and `nvv4l2decoder` reports "cannot allocate buffers" or similar cryptic errors

### Deep Dive: NvMMLite, libtegrav4l2, and the Tegra Multimedia API

The layers below libv4l2_nvvideocodec:

**libtegrav4l2.so** — translates V4L2 intent into NvMMLite calls. It's closed-source, lives in `/usr/lib/aarch64-linux-gnu/tegra/libtegrav4l2.so`. Not documented publicly but matches typical V4L2 M2M semantics.

**NvMMLite** — NVIDIA's multimedia abstraction for Tegra. Wraps individual hardware blocks:

- **NVDEC** — H.264/H.265/VP9/AV1 decode
- **NVENC** — H.264/H.265 encode
- **VIC** (Video Image Compositor) — color conversion, scaling, composition
- **JPEG** — hardware JPEG encode/decode
- **NvBufSurface** — DMA-BUF-backed buffer management across these blocks

**NvBufSurface** is critical for zero-copy: buffers allocated as NvBufSurface can be shared between NVDEC, VIC, and CUDA contexts without copies through system memory. GStreamer's `nvvideoconvert` uses NvBufSurface internally; if the downstream element (like `nvinfer` in DeepStream) also understands NvBufSurface, the whole pipeline runs on GPU memory.

Our pipeline uses `video/x-raw,format=RGB` which forces a copy to system memory at the boundary between `nvvideoconvert` and `appsink` — we give up zero-copy to get plain RGB bytes into Swift. A future optimization would be to consume NvBufSurface directly via a Swift wrapper, but that requires:

1. Wrapping `nvbufsurface.h` as a Swift C module (headers are in the SDK, we already did that)
2. Changing the pipeline to use `video/x-raw(memory:NVMM),format=RGBA`
3. Calling `NvBufSurfaceMap` to get CUDA-accessible pointers
4. Passing those pointers directly to TensorRT's `enqueueV3` or `enqueueV2` without a CPU copy

That's a blog post in itself — probably the next one.

### Deep Dive: Why NVIDIA Doesn't Ship a Kernel V4L2 M2M Driver

Upstream Linux has several in-kernel V4L2 M2M drivers (Samsung MFC, Broadcom VPU, Amlogic, etc.). They follow a standard pattern: kernel driver exposes `/dev/videoN` with ioctls for queue/dequeue buffers, userspace just uses plain V4L2. No plugin hackery.

NVIDIA's choice to put the stack in userspace has tradeoffs:

- **Pro**: faster to evolve — no kernel releases or upstream patches. NVIDIA ships plugin updates as library updates
- **Pro**: avoids GPL licensing concerns on the codec logic (kernel V4L2 drivers need to be GPL, userspace libraries can be proprietary)
- **Pro**: tighter integration with other NVIDIA userspace components (cuDNN, TRT, etc.) that already expect NvBufSurface memory layouts
- **Con**: container deployments need special attention (CDI spec, LIBV4L2_PLUGIN_DIR, passthrough paths)
- **Con**: debugging is harder — you can't use standard V4L2 tools like `v4l2-ctl` without going through the plugin path
- **Con**: not upstream Linux, so distros that don't ship NVIDIA's libv4l2 plugin can't use the hardware decoder

For the blog post the interesting part is: **this is an API you think you know, but the implementation is completely different from what you expect.** V4L2 as a contract, NvMMLite as the implementation.

### Deep Dive: The WendyOS CDI Spec Layout

WendyOS uses `nvidia-ctk` to generate `/etc/cdi/nvidia.yaml` from input CSV files in `/etc/nvidia-container-runtime/host-files-for-container.d/`. The layout (after our patches):

| File | Purpose | Lines |
|---|---|---|
| `drivers.csv` | CUDA driver libs, NVDEC/NVENC plugins | ~80 |
| `devices.csv` | Device nodes (/dev/nvidia0, /dev/nvhost-*, /dev/v4l2-nvdec) | ~40 |
| `l4t.csv` | L4T-specific libs (commented-out `v4l2-nvdec` entry with warning) | ~300 |
| `l4t-deepstream.csv` | DeepStream 7.1 libs and plugins | ~480 |
| `l4t-gstreamer.csv` (added today) | GStreamer core libs, GLib deps, plugins | ~50 |
| `fix-cdi-gstreamer-paths.sh` | Post-processing: fix plugin paths, inject env vars | N/A (script) |

The CSV format is flat: `<type>, <path>` where type is `dev`, `lib`, `sym`, `dir`, or `env`. nvidia-ctk reads the CSVs, follows symlinks, resolves transitive library dependencies via `ldd`, and writes the final CDI spec as YAML.

The `fix-cdi-gstreamer-paths.sh` script runs **after** nvidia-ctk. It uses `sed` to patch the generated `/etc/cdi/nvidia.yaml` with things nvidia-ctk can't express, like environment variables (`LIBV4L2_PLUGIN_DIR`, `GST_PLUGIN_PATH`). The `env` type in CSV files is ignored by nvidia-ctk 1.16.2, so we do it in the post-processor instead.

The `edgeos-cdi-generate.service` systemd unit runs nvidia-ctk and the post-processor in sequence. Restart it to regenerate the spec after editing CSVs or the script.

---

## Part 7: Concurrency — Still the Swift 6.2 Story

Three isolation domains, each chosen for a reason:

### `DetectorEngine` (actor)

TensorRT's `ExecutionContext` is not thread-safe. Actor isolation guarantees serial access without manual locking. Multiple streams queue naturally.

```swift
actor DetectorEngine {
    let context: ExecutionContext
    let postprocessor: YOLOPostprocessor

    func detect(frame: Frame) async throws -> [Detection] {
        let (inputData, letterbox) = frame.data.withUnsafeBufferPointer { ptr in
            YOLOPreprocessor.preprocess(ptr, width: frame.width, height: frame.height)
        }
        var outputBuffer = [Float](repeating: 0.0, count: 300 * 6)
        try await context.enqueueF32(
            inputName: "images", input: inputData,
            outputName: "output0", output: &outputBuffer)
        var detections = outputBuffer.withUnsafeBufferPointer { ptr in
            postprocessor.process(output: ptr, batchSize: 1)
        }
        YOLOPreprocessor.remapBoxes(&detections, letterbox: letterbox)
        return detections
    }
}
```

### `MetricsRegistry` (Mutex, not actor)

Called on the hot path at 20+ FPS. A `Mutex<RegistryState>` avoids the async hop that an actor would require. The `/metrics` endpoint snapshots everything in one lock acquisition, then formats outside the lock.

### `GStreamerFrameReader` (actor, with dispatch escape hatch)

Pipeline state is actor-isolated. But `gst_app_sink_pull_sample` blocks, and blocking the Swift cooperative thread pool deadlocks under multiple streams. So we bounce the blocking call onto a dedicated `DispatchQueue` via `withCheckedContinuation`. The `SendableGstPointer` wrapper (an `@unchecked Sendable` struct around a raw pointer) lets us cross the closure boundary without a Swift 6 Sendable compile error, while the actor guarantees only one task pulls at a time.

### `DetectorState` (actor)

MJPEG frame buffer with per-client `AsyncStream` continuations. Actor isolation ensures client connect/disconnect/frame-broadcast are serialised. The MJPEG endpoint uses `withThrowingTaskGroup` inside a Hummingbird `ResponseBody` closure — frame forwarder + keepalive clock + writer loop, all cleaned up automatically on client disconnect.

### Deep Dive: Swift 6 Actors vs. Mutexes in Practice

Swift 6 actors are conceptually simple: each actor has a single logical thread of execution (the "actor's executor"), and all access to actor state must go through that executor via `await`. In practice there are several subtleties that matter for hot-path code.

**Actor hop cost.** An `await someActor.method()` incurs a cooperative thread-pool context switch if you're calling from outside the actor's isolation domain. Swift's scheduler is fast — on the order of 100-500 ns for a no-op hop on modern hardware — but at 20 FPS with multiple hops per frame, the cost is ~1-5 µs of pure scheduling overhead per frame. Negligible against 28 ms of inference, but worth knowing.

**Reentrancy.** Actors are reentrant by default. When an actor method `await`s something, the executor is free to run other queued calls to the same actor. This is usually what you want (better throughput) but it means *you can't assume invariants hold across an `await`*. If you mutate state, `await` something, then read state, the state may have changed.

In our code, the critical hot-path actor methods (`DetectorEngine.detect`, `IOUTracker.update` as a value-type mutation, etc.) don't `await` anything internally. They're straight-line synchronous code inside an `async` method, which means the actor hop happens at the call boundary and no reentrancy can occur during execution.

**Mutex vs actor for the metrics registry.** We use `Mutex<RegistryState>` (from `swift-synchronization`) for metrics instead of an actor. Reasoning:

- Metric increments happen at 20+ Hz per stream, from the detection loop which is already in an actor's isolation domain
- Making MetricsRegistry an actor would require an `await` at every increment — another actor hop
- A Mutex is a plain spinlock + short critical section (~50 ns). No scheduling, no async overhead
- The tradeoff: you can't call mutex-protected code from inside an actor without blocking the actor's executor thread, but our metric writes are fire-and-forget (record the value and move on), not long-running

**The `nonisolated` escape hatch.** The `MetricKey`, `GaugeMetric`, etc. handle types are `nonisolated` because they can be read/written from any concurrency domain (including the Hummingbird HTTP server when rendering Prometheus output). Their backing storage is the `Mutex<RegistryState>` which enforces correctness.

### Deep Dive: The `SendableGstPointer` Trick

The pattern:

```swift
private struct SendableGstPointer<T>: @unchecked Sendable {
    let pointer: UnsafeMutablePointer<T>
    init(_ pointer: UnsafeMutablePointer<T>) {
        self.pointer = pointer
    }
}
```

`UnsafeMutablePointer<T>` is not `Sendable` by default because the pointee isn't guaranteed to be thread-safe. But in our case:

1. The pointer points to a `GstAppSink` that lives for the duration of the actor
2. Only one thread at a time pulls from the sink (enforced by the actor's single-execution constraint)
3. The pointer itself is a POD value (64 bits), so crossing a closure boundary is a plain copy

`@unchecked Sendable` tells the compiler "trust me, this is safe to send across isolation domains," skipping the Sendable check. We use it sparingly — only for GStreamer C pointers that we know are actor-owned.

**Why not `nonisolated(unsafe)` on the stored property?** Because `nonisolated(unsafe)` means "this property can be accessed from any isolation without the compiler's safety checks" — it's a storage attribute. We want the opposite: the pointer is **actor-isolated** when stored, we just need to cross one boundary (into the dispatch queue) while temporarily pretending it's Sendable. `@unchecked Sendable` on the wrapper struct is the right fit.

### Deep Dive: Bridging Blocking C APIs into Async Swift

The pattern for any blocking C function:

```swift
let result: T = await withCheckedContinuation { continuation in
    dedicatedQueue.async {
        let value = blocking_c_function()
        continuation.resume(returning: value)
    }
}
```

Things to get right:

1. **Use a dedicated DispatchQueue, not `.global()`.** Swift's cooperative thread pool has a fixed number of threads (usually `numberOfCores`). If you block all of them, you deadlock the whole async runtime. A dedicated queue has its own thread and won't contend with the cooperative pool
2. **Don't `resume` the continuation more than once.** If the C function can fail or be cancelled, guard the resume call with an atomic flag
3. **Handle cancellation.** `withCheckedContinuation` doesn't participate in `Task.isCancelled` automatically. For long-running blocks, check cancellation before and after the blocking call, and throw `CancellationError` if needed
4. **Sendable across the boundary.** Anything captured by the dispatch closure must be `Sendable` (or `@unchecked Sendable`). See the `SendableGstPointer` pattern above

Our GStreamer frame pull is a clean example:

```swift
func nextFrame() async throws -> Frame? {
    guard let sink = appsink else { throw GStreamerError.notStarted }
    let sinkPtr = SendableGstPointer(sink)
    let frameWidth = width
    let frameHeight = height

    let bytes: [UInt8]? = await withCheckedContinuation { continuation in
        pullQueue.async {
            var sampleHandle: UnsafeMutableRawPointer?
            var dataPtr: UnsafeMutableRawPointer?
            var size: Int = 0
            let ok = wendy_gst_pull_sample(sinkPtr.pointer,
                                           &sampleHandle, &dataPtr, &size)
            if ok == 0 { continuation.resume(returning: nil); return }
            defer { wendy_gst_release_sample(sampleHandle) }
            let buffer = UnsafeBufferPointer<UInt8>(
                start: dataPtr!.bindMemory(to: UInt8.self, capacity: size),
                count: size)
            continuation.resume(returning: Array(buffer))
        }
    }
    guard let bytes else { return nil }
    return Frame(data: bytes, width: frameWidth, height: frameHeight)
}
```

Note: `frameWidth`/`frameHeight` are captured as local constants before the closure to avoid an actor hop inside the closure (Swift won't let you read actor properties from inside a nonisolated closure).

### Deep Dive: Why Not a Frame Queue?

The detection loop processes frames synchronously: pull a frame, run detect+track+render+metrics, loop. There's no queue between frame capture and inference.

Alternative: an `AsyncStream<Frame>` or a bounded buffer between `GStreamerFrameReader` and the inference loop. Benefits:

- Decouples capture rate from processing rate — if processing slows down temporarily, frames buffer
- Could run capture on its own Task, slightly improving throughput

Why we didn't:

- `appsink` is already a bounded buffer with `max-buffers=2 drop=true`. Two frames is enough to cover jitter
- A Swift-side queue adds complexity (bounded capacity, drop semantics, memory pressure) for marginal benefit
- The detection loop is the rate limit — if it can't keep up, adding a queue doesn't help, just delays frame-drop decisions
- Frame-drop at the GStreamer boundary is preferable to frame-drop in the middle of the pipeline (less work wasted)

---

## Part 8: Dependencies

```
tensorrt-swift        — Swift bindings for TensorRT 10 (wendylabsinc)
hummingbird 2.6+      — HTTP server
swift-argument-parser  — CLI
swift-log              — Structured logging
async-http-client      — VLM sidecar calls
swift-container-plugin — Native containerization (no Docker)

libturbojpeg           — JPEG encoding (system library)
libgstreamer-1.0       — Video pipeline (system library)
libavformat/avcodec    — FFmpeg fallback decoder (system library)
```

---

## Part 9: Running It

### Step 1: Install the wendyos Swift SDK

```bash
swift sdk install https://github.com/wendylabsinc/wendy-swift-tools/releases/download/0.4.0/6.2.3-RELEASE_wendyos_aarch64.artifactbundle.zip \
  --checksum ef8fa5a2eda766e3b1df791dc175bbf87f570b9cc6f95ada1fe7643a327e087e
```

Then overlay the missing pieces (GStreamer, CUDA, DeepStream) from a Jetson sysroot if you don't have them in the shipped SDK. See `scripts/extend-sdk.sh`.

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

### Step 3: Export the YOLO26 model

```bash
pip install ultralytics
yolo export model=yolo26n.pt format=onnx
mv yolo26n.onnx detector-swift/
```

### Step 4: Deploy

```bash
cd detector-swift
export WENDY_AGENT=your-jetson.local
wendy run --detach --restart-unless-stopped
```

The first launch takes ~5-8 minutes because the TensorRT engine is built from the ONNX file on-device. The engine is then cached to disk, so subsequent starts are immediate.

### Step 5: Launch the VLM sidecar

```bash
ssh root@your-jetson.local "docker run -d --runtime nvidia \
  --name llama-vlm --restart unless-stopped -p 8090:8090 \
  -v /root/vlm-models:/models \
  dustynv/llama_cpp:b5283-r36.4-cu128-24.04 \
  /usr/local/bin/llama-server \
    --model /models/Qwen3-VL-2B-Instruct-Q4_K_M.gguf \
    --mmproj /models/mmproj-F16.gguf \
    --host 0.0.0.0 --port 8090 \
    --n-gpu-layers 99 --ctx-size 4096"
```

### Step 6: View

```bash
# Live MJPEG with bounding boxes
open http://your-jetson.local:9090/stream

# Recent VLM descriptions (JSON)
curl http://your-jetson.local:9090/api/vlm_descriptions | jq .

# Prometheus metrics
curl http://your-jetson.local:9090/metrics | grep deepstream_fps

# Full dashboard
open monitor.html
```

---

## Why Swift for This

| Concern | Python + DeepStream | Swift 6.2 |
|---------|---------------------|-----------|
| **Thread safety** | GIL + manual threading | Actors + Sendable, compiler-verified |
| **Memory** | NumPy copies, GC pauses | Value types, no GC, deterministic |
| **Dependencies** | DeepStream SDK, GStreamer, PyGObject, NumPy, OpenCV, Flask | TensorRT bindings, Hummingbird, turbojpeg, GStreamer (direct C API) |
| **Container size** | ~2 GB | ~200 MB |
| **Build time** | ~3-5 min (Docker on device) | **85 seconds** (native cross-compile) |
| **Build verification** | Runtime errors | Compile-time concurrency checking |
| **Hot path overhead** | Python → C++ → CUDA via GStreamer | Swift → TensorRT C API, zero intermediate layers |
| **Hardware decode** | `nvinfer` bundled with DeepStream | Explicit GStreamer pipeline, we wrote the glue |
| **VLM integration** | Custom Flask sidecar + custom HTTP | `AsyncHTTPClient` + OpenAI-compatible endpoint |

---

## Final Numbers

| Metric | Value |
|---|---|
| Swift source | ~4,160 lines across 12 files |
| C shim (GStreamer interop) | 164 lines (1 header + 1 modulemap) |
| Python replaced | 1,208 lines |
| Container image | 404 MiB |
| Build time (cross-compile) | 14s cold, 0.7s incremental |
| End-to-end FPS | **~20** (camera-limited) |
| Inference latency (mean) | 29.2 ms |
| Total frame latency (mean) | 29.3 ms |
| CPU usage | 54.5% (hardware decode) |
| Memory (RSS) | 686 MB |
| Tracker overhead | < 0.1 ms (essentially free) |
| VLM decode | 23.6 tok/s (Qwen3-VL-2B Q4_K_M via llama.cpp sidecar) |
| VLM TTFT | ~1.5 s per car |

Zero lines of Python, C++, or Objective-C in the detector. The only non-Swift code is a 164-line C shim for GStreamer type casts and buffer extraction.

---

## Head-to-Head: Python + DeepStream vs. Swift + TensorRT

Both detectors run on the same hardware (Jetson Orin Nano 8GB, JetPack 6.2.1) against the same 1080p/20fps RTSP camera, doing object detection with tracking and serving Prometheus metrics over HTTP.

### The Numbers

| | Python + DeepStream | Swift + TensorRT |
|---|---|---|
| **FPS** | ~20 (camera-limited) | ~20 (camera-limited) |
| **Inference latency** | ~30 ms (`nvinfer`) | 29.2 ms (TensorRT direct) |
| **CPU usage** | 80-100% | **54.5%** |
| **Memory (RSS)** | ~1.5 GB | **686 MB** |
| **Container image** | 124 MiB | 404 MiB |
| **Decode** | Software (`uridecodebin` fallback) | **NVDEC hardware** (`nvv4l2decoder`) |
| **Model** | YOLO11n (DeepStream `nvinfer`) | YOLO26n FP16 (TensorRT 10.3) |
| **Tracking** | NvDCF (DeepStream tracker) | IOU tracker (custom Swift) |
| **VLM second stage** | None | Qwen3-VL-2B via llama.cpp (planned) |
| **Build time** | Minutes (Docker on device) | **0.7s** incremental cross-compile |
| **Lines of code** | 1,208 (Python) | 4,160 (Swift) + 164 (C shim) |
| **Dependencies** | 28 imports + DeepStream SDK | 5 SPM packages |
| **Runtime deps** | Python 3.10, PyGObject, NumPy, Flask | Just Swift runtime |

### Where Swift Wins

**CPU and memory efficiency.** The Python version burns 80-100% CPU because the GIL serializes GStreamer callbacks, NumPy array copies, and Flask request handling onto one core. The Swift version uses structured concurrency — decode, inference, tracking, and HTTP serving run on separate cooperative tasks without contention. Memory is 2.2x lower because there's no Python interpreter, no NumPy array pool, and NVDEC keeps decoded frames in GPU NVMM memory instead of CPU-side RGB buffers.

**Hardware decode.** The Python version uses `uridecodebin` which silently falls back to software decode (`avdec_h264`) because the WendyOS `libv4l2.so` was misconfigured. The Swift version explicitly uses `nvv4l2decoder` with the corrected library chain, offloading H.264 decode to the dedicated NVDEC silicon. Both hit the same 20 FPS camera limit, but the Swift version does it with 30% less CPU — headroom that matters for multi-stream or VLM inference.

**Build cycle.** The Python version builds inside a Docker container on the Jetson itself (ARM, slow). Changing one line of `detector.py` triggers a layer rebuild that takes minutes. The Swift version cross-compiles on an x86 host in 0.7 seconds (incremental) and pushes a new container image over the LAN. The edit-deploy-test cycle is under 15 seconds.

**VLM integration.** The Swift version includes a `TrackFinalizer` that submits the best crop from each finalized track to a Qwen3-VL-2B sidecar for natural-language description. The Python version has no VLM path — adding one would mean integrating a separate inference server into the DeepStream pipeline.

### Where Python Wins

**Lines of code.** 1,208 lines vs 4,160. DeepStream handles a LOT — `nvstreammux`, `nvinfer`, `nvtracker`, `nvdsosd` are all turnkey GStreamer elements that each replace 200+ lines of Swift. The Swift version implements its own preprocessing, postprocessing, tracking, rendering, and HTTP server from scratch.

**Container image size.** 124 MiB vs 404 MiB. The Python image is smaller because the heavy lifting (TensorRT, GStreamer, DeepStream) comes from CDI injection at runtime. The Swift image bundles a static FFmpeg binary (51 MB), the TensorRT engine file (8 MB), and the ONNX model (15 MB) as resources.

**Ecosystem.** DeepStream is NVIDIA's supported production framework with documentation, forums, and a large community. The Swift-on-Jetson path is uncharted — every CDI issue, GStreamer version mismatch, and v4l2 plugin chain problem had to be debugged from first principles.

### The Debug Build Trap

One critical lesson: **Swift debug builds are 5x slower than release builds.** Early testing showed 4.8 FPS, which looked like a fundamental performance problem. It was just `-Onone`. The compiler can't inline the preprocessing loops (letterbox resize, float normalization, HWC→CHW transpose) that run on every frame. Always build with `-c release` for any performance comparison.

| Build mode | Decode | FPS | CPU | Memory |
|---|---|---|---|---|
| Debug + software | avdec_h264 | 4.8 | 100% | 2.1 GB |
| Release + software | avdec_h264 | 24.7 | 100% | 2.1 GB |
| Release + hardware | **nvv4l2decoder** | **~20** | **54.5%** | **686 MB** |

### The Bottom Line

For a blog post: Swift matches Python + DeepStream on throughput while using half the CPU and a third of the memory. The tradeoff is more code (4x the line count) and a harder integration path (no DeepStream safety net). For production at scale, the CPU headroom matters — it's the difference between running one camera stream and three on the same $250 device.

---

## War Stories

Things that took longer than they should have:

1. **TensorRT 8 vs 10 mismatch.** First build used `l4t-jetpack:r36.2.0` as the base. That ships TensorRT 8.6. The Jetson runs JetPack 6.2.1 with TensorRT 10.3. Binary linked `libnvinfer.so.8`, runtime has `libnvinfer.so.10`. Fix: switch base to `l4t-jetpack:r36.4.0`.

2. **Ubuntu 24.04 vs 22.04 ABI.** Runtime stage used `ubuntu:24.04`. FFmpeg libraries are `.so.60` there. Builder (l4t-jetpack) is on 22.04 with `.so.58`. Fix: match runtime base to builder ABI.

3. **CUDA stub library.** `tensorrt-swift` links `-lcuda`. That's the CUDA driver library, provided by the host via CDI at runtime — but we need a stub at link time. Fix: `ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs`.

4. **FFmpeg `-timeout` became "listen mode timeout"** in newer FFmpeg. Our old args made FFmpeg try to *listen* on the RTSP URL instead of connecting. Fix: switch to `-stimeout` (socket timeout).

5. **FFmpeg `-hwaccel v4l2m2m`** is not a valid hwaccel name in stock Ubuntu FFmpeg. Only `vdpau cuda vaapi drm opencl` are supported. Fix: removed the flag (and later: switch to GStreamer + `nvv4l2decoder` for the real solution).

6. **libv4l2 plugin dir.** See Part 6 above.

7. **SwiftPM `pkgConfig:` is not reliable under cross-compile.** Drop it and bake paths into the SDK toolset.json.

8. **Swift cross-link flags.** Took four attempts to figure out where `--target=aarch64-unknown-linux-gnu` actually needs to go. It's not the `linker` section.

9. **`dustynv/llama_cpp:*-r36.4.0`** images ship llama-cpp-python 0.3.8 which predates Qwen3-VL support. Rebuild llama-server from source with `-DCMAKE_CUDA_ARCHITECTURES=87`.

10. **Disk space.** The build host ran out of space mid-way through a Docker build. 355 GB of Yocto intermediate artifacts in `tmp/work/` needed to go. A specialized agent did the cleanup with explicit safety checks for stashed model weights.

---

## Branch History

```
5b6aa29 Add Swift 6.2 YOLO11n detector replacing Python/DeepStream runtime
8bf9a82 Fix critical bugs found during code review
e8ec484 Fix all compilation errors (verified with Swift 6.2.3)
9654b11 Fix all issues from thorough code review
0a8f34a Add architecture doc for Swift conference presentation
<current> Upgrade to YOLO26, add VLM pipeline, GStreamer integration,
         native cross-compile, and the LIBV4L2_PLUGIN_DIR discovery
```

---

# Q&A: The Deep Thinkers in the Audience

The questions a machine learning engineer, a CV engineer, a CUDA/TensorRT specialist, a Swift language lawyer, a Jetson BSP engineer, and a DevOps person might ask you after this talk. Answered honestly, with caveats where they exist.

## Model and ML Questions

**Q: Why YOLO26 and not YOLOv10, RT-DETR, or D-FINE?**

YOLOv10 is also NMS-free (it's where the one-to-one head idea came from) and was released about a year earlier. We picked YOLO26 because:

- It's from Ultralytics, which has the most mature export tooling (ONNX, TensorRT, CoreML, etc.)
- DFL is removed — cleaner ONNX graph than v10 which still has DFL
- Documentation and community support are better right now
- Small-target-aware label assignment is a genuine improvement for our outdoor surveillance use case

RT-DETR and D-FINE are transformer-based. They have better accuracy per FLOP but the attention layers are less friendly to TensorRT's kernel fusion and have higher constant overhead — worse fit for an Orin Nano with a tight latency budget.

**Q: You claim "COCO mAP 37.5%" for YOLO26n. Is that on the stock weights? What's the test set?**

Yes, stock weights from the Ultralytics release, measured on COCO val2017 by Ultralytics. We didn't re-run the benchmark ourselves. For our application (outdoor camera with cars and people, fixed viewpoint), domain-specific mAP would be lower — COCO is diverse and the nano model is small. Production deployment would call for fine-tuning on the deployment environment.

**Q: Why nano variant? Doesn't small give better accuracy?**

Orin Nano's integrated Ampere GPU peaks around 5 TFLOPS FP16. At 640×640 input, FP16 kernels:

- nano (1.5 GFLOPs): ~28 ms per inference
- small (6.5 GFLOPs): ~100 ms
- medium (22 GFLOPs): ~250 ms

For a 20 FPS target with a 50 ms per-frame budget, only nano fits with headroom. Upgrading to Orin NX (16 GB, ~10 TFLOPS) would make small feasible. The AGX Orin (~40 TFLOPS) could run medium or large. We stuck with nano because the story is about getting *any* model running on the cheap Jetson.

**Q: FP16 vs INT8?**

FP16 is the conservative choice: minimal accuracy loss from FP32 (usually under 0.3 mAP), well-supported by every TensorRT layer, no calibration dataset needed. INT8 gives ~1.5-2× speedup but:

- Requires either Post-Training Quantization (PTQ) with a calibration dataset, or Quantization-Aware Training (QAT) during model training
- PTQ accuracy loss depends on the model — YOLO variants usually lose 0.5-1.5 mAP
- More fragile: some layers don't have INT8 kernels and fall back to FP16 or FP32, complicating the engine
- Worth it for production if latency is critical; overkill for a blog post

FP8 is not supported on Orin (Blackwell and Hopper only). If you want higher throughput on Jetson, INT8 with QAT is the path.

**Q: How does YOLO26's one-to-one head actually eliminate duplicates without NMS?**

During training, the loss function runs the Hungarian algorithm to compute an optimal assignment between predictions and ground-truth boxes — each GT matches exactly one prediction. The matching cost combines classification score and bbox IoU. Predictions that are "close seconds" to the assigned one receive a penalty, which teaches the model to push them away from the GT.

At inference, because the model was trained to output distinct predictions, the output is already deduplicated. The 300 fixed slots are a design choice — you could set it to any number during training, but 300 is a good default that covers most scenes with headroom.

**Q: What if the true object count exceeds 300?**

The output is capped at 300. In practice, COCO images have at most ~70 objects, and your typical camera frame has far fewer. For crowd scenes (e.g., a train station), you'd want to retrain with a higher max-detections setting or use a larger model.

**Q: You're not using a ReID model for the tracker. How do you handle ID switches on occlusion?**

We use pure IoU-based association with a Kalman-filtered motion model. Configuration:

- `probationAge=3`: a tentative track becomes confirmed after 3 consecutive matched frames
- `maxShadowTrackingAge=30`: a confirmed track is predicted forward for up to 30 frames (1.5 s at 20 FPS) without a match before being pruned
- `minIouDiff4NewTarget=0.5`: re-association requires 50% IoU overlap

This works well for short occlusions (a car passing behind a light pole) but fails for long occlusions (a car driving behind a building for 5 seconds). For production, you'd add an appearance descriptor (a small CNN trained to produce a per-object embedding) and use it as a tiebreaker when IoU is ambiguous. That's a BoT-SORT / StrongSORT style tracker.

We didn't do that here because:
1. Our fixed-camera use case has relatively clean motion paths
2. Adding a ReID descriptor means a second neural network in the pipeline
3. It's out of scope for the blog post — the point is "Swift can drive a good tracker," not "we built the best tracker"

**Q: Kalman filter specifics — what's the state vector, process noise, measurement noise?**

State vector: 8 dimensions — `[cx, cy, w, h, vx, vy, vw, vh]` — center position, size, and velocities.

State transition matrix F (constant velocity):
```
[I4  I4]    where I4 is the 4x4 identity
[ 0  I4]
```

Measurement matrix H (4x8): selects `[cx, cy, w, h]` from the state.

Process noise Q: diagonal, `σ² = 2.0` for position/size, `σ² = 0.1` for velocities. These are the values the NvDCF tracker uses by default in DeepStream; we matched them to get behavior similar to the Python reference.

Measurement noise R: diagonal, `σ² = 4.0` for all four bbox components. Detector confidence doesn't directly scale R — we treat all detections equally regardless of confidence score.

Initial covariance P is diagonal with inflated velocity variance (100) to prevent the filter from being overconfident about velocity on the first frame.

We symmetrize P after each update to prevent numerical drift: `P = (P + P^T) / 2`.

**Q: Why greedy matching instead of Hungarian (global optimal)?**

Same reason as the NvDCF defaults: greedy is faster and accuracy loss is negligible for well-separated objects. Hungarian is O(n³) in the number of candidates; greedy is O(n² log n) for sorting, O(n²) for assignment. At our track counts (~3-10 active per frame) it doesn't matter either way, but the greedy implementation is simpler.

## TensorRT and GPU Questions

**Q: Why TensorRT and not ONNX Runtime with the TensorRT execution provider?**

ONNX Runtime adds a layer of indirection: ORT parses the ONNX graph, delegates to the TensorRT EP for the operators it can, and runs the rest on CPU or another EP. That's fine for portability across hardware but:

- Extra copies between ORT tensors and TensorRT buffers
- The ORT-TRT integration is less tight than using TRT directly
- We want Swift code talking to a single native library, not a chain of wrappers

Going direct to TensorRT gives us predictable behavior and minimum overhead. The downside is we're now NVIDIA-locked — porting to a Raspberry Pi or a Coral would require rewriting this layer.

**Q: Do you ever rebuild the TensorRT engine?**

First run: builds from `yolo26n.onnx` (~5-8 minutes on Orin Nano). The engine is then saved to disk as `yolo26n_b2_fp16.engine`. Subsequent starts deserialize the saved engine in <1 s.

Rebuild is required when:

- TensorRT version changes (e.g., upgrading JetPack)
- GPU SM version changes (moving from Orin to AGX Orin changes SM from 8.7 to 8.7 — same, so no rebuild. Moving to Thor would change to SM 10.x)
- Precision mode changes (FP16 → INT8)
- Dynamic shape profiles change
- ONNX file changes (e.g., retrained model)

If you deploy to heterogeneous hardware, you build the engine per target and ship the right one with each image. If your deployment is uniform, build once and ship the engine.

**Q: Static vs dynamic batch size?**

We use a dynamic shape profile with `min=1, opt=2, max=2`. This means:

- The same engine handles batch 1 and batch 2 inputs
- TensorRT selects tactics optimized for batch 2 at engine-build time
- At inference, you call `setInputShape(batch, ...)` before each enqueue to tell TRT what actual batch size to use

For single-stream detection we always use batch 1. For multi-stream with lock-step processing (two cameras), we use batch 2. No engine rebuild needed to switch between them.

Dynamic shapes have a small overhead at inference time (TRT checks the shape matches a supported profile, picks the right kernels). Worth it for flexibility.

**Q: What's the TensorRT workspace and why is 2 GB enough?**

The workspace is scratch memory TensorRT uses for:

- Tactic selection during engine building (needs the most)
- Intermediate activations that don't fit in persistent memory
- Plugin scratch space

`--memPoolSize=workspace:2048M` caps it at 2 GB. For YOLO26n at batch 2, actual workspace usage is well under 1 GB. Setting it larger doesn't help — it just reserves memory. Setting it too small can cause the engine build to fail with "out of memory" even though the final engine would fit.

On Orin Nano with 8 GB shared RAM, memory is tight. The kernel and other system services eat 1-2 GB, the Swift detector process is ~200 MB, the TensorRT engine is ~100 MB, and the llama.cpp sidecar is ~2.2 GB. We have maybe 3-4 GB of usable headroom.

**Q: How does `enqueueF32` work in tensorrt-swift? Is it truly async?**

Looking at the C++ shim in `tensorrt-swift/Sources/TensorRTNative/TensorRTNative.cpp`, `enqueueF32` does:

1. `cudaMemcpyAsync` the input from host to device (into a pre-allocated GPU buffer)
2. `context->enqueueV3` to schedule the inference on a CUDA stream
3. `cudaMemcpyAsync` the output from device back to host
4. `cudaStreamSynchronize` to wait for everything to complete

So while individual CUDA calls are async, the whole `enqueueF32` call is synchronous from Swift's perspective — it blocks the caller until the output buffer is populated.

A truly async design would return a handle immediately and let Swift `await` the completion via a continuation wired to `cudaEvent` callbacks. That's a future optimization; we haven't done it because the synchronous path is already fast enough for our latency target.

**Q: Are you using CUDA streams at all?**

One default stream per engine. No overlap between:

- Preprocessing CPU memcpy → GPU input buffer
- Inference kernel execution
- Output GPU buffer → host memcpy

On a Jetson with unified memory, the device-to-host copies are actually memory fences rather than real copies, so they're cheap. But there's room for optimization: run preprocessing for frame N+1 on the CPU while inference for frame N runs on the GPU. That would cut per-frame latency by ~5-10 ms in our case. Not critical yet.

**Q: What about cuDLA? Can you offload the YOLO to the DLA?**

The Deep Learning Accelerator is a separate hardware block on Jetson that runs a subset of TensorRT operators. On Orin Nano there's **one** DLA (Orin NX has two, AGX has two as well).

Running YOLO on the DLA would free up the main GPU for the VLM. But:

1. Not all TRT ops have DLA implementations. Some layers silently fall back to GPU, losing the locality benefit
2. The DLA is slower than the GPU for most workloads we care about (~30-50% of GPU throughput on Orin)
3. For the current 19 FPS target, the GPU has enough headroom to handle both YOLO and occasional VLM calls
4. Setting up cuDLA deployment requires `--useDLACore=0 --allowGPUFallback` flags to trtexec, separate engine builds, and extra testing

Worth investigating as a follow-up post. The likely winner is "DLA for YOLO, GPU dedicated to VLM" — zero contention, potentially higher sustained throughput. But only if your YOLO variant works well on DLA.

**Q: Why not NvBufSurface zero-copy? You mention it but don't use it.**

Zero-copy would look like:

1. GStreamer produces NvBufSurface-backed buffers from nvv4l2decoder (already does this)
2. Pipeline consumes the surface directly at the preprocessor / TRT input boundary — no copy to system memory
3. Preprocessing happens on GPU (NPP or a custom CUDA kernel for letterbox + normalize + transpose)
4. TRT reads from the same GPU memory

We didn't do this because:

1. Requires wrapping `nvbufsurface.h` as a Swift C module — doable, we already have the headers in the SDK
2. Requires modifying `tensorrt-swift` to accept CUDA device pointers instead of Swift `[Float]` arrays — we'd need to fork the package
3. Requires a GPU letterbox kernel — NPP can do it but there's glue to write
4. Current CPU-based path is already hitting the FPS target, so the work-to-value ratio is low for a blog post

It's the obvious next optimization if you want to push the Nano harder, and it's almost certainly a 2× FPS improvement. Follow-up post material.

**Q: Why FP16 plugins? Aren't TRT plugins usually INT8 or FP32?**

The YOLO26 ONNX graph uses standard ops (Conv, BatchNorm, SiLU, Concat, Reshape, Gather). TensorRT has native FP16 kernels for all of these — no custom plugins needed. The ONNX export does some preprocessing of anchor boxes that TRT handles via its built-in ops. So there are zero TensorRT plugins in our engine — everything is built-in kernels, all running at FP16.

If you use a model with unusual operators (e.g., DCN v2, deformable attention), you'd need to either write a TensorRT plugin in C++ or use a different quantization that avoids the unsupported op.

## Preprocessing and Video Questions

**Q: Why do you use `videoconvert` / `nvvideoconvert` to produce RGB instead of consuming native NV12?**

YOLO26 was trained on RGB input. Using NV12 directly would require either:

1. Retraining the model on NV12 (unusual; most datasets are RGB JPEGs or PNGs)
2. Writing a CUDA kernel to convert NV12 → CHW Float RGB on the fly
3. Using TensorRT's internal format converters

We took the path of least resistance: let GStreamer's `nvvideoconvert` (which uses the VIC hardware) convert to RGB and hand us bytes. The VIC is fast enough that this isn't a bottleneck.

For the zero-copy path (future optimization), we'd accept NV12-in-NVMM and do the conversion as part of the letterbox+normalize CUDA kernel, avoiding the intermediate RGB format entirely.

**Q: Why letterbox instead of stretching to 640×640?**

Letterboxing preserves aspect ratio. If you stretch a 1920×1080 frame to 640×640, objects get compressed horizontally (cars become squat, people become short). The model was trained with letterbox preprocessing, so inference on stretched frames produces worse detections.

Letterbox adds gray (value 114) padding to the short dimension. For 1920×1080 → 640×640:

- Scale factor: min(640/1920, 640/1080) = 0.3333 → scaled size 640×360
- Vertical padding: (640 - 360) / 2 = 140 on top and bottom
- Output: 640 wide × (140 + 360 + 140) tall = 640 × 640

The postprocessor undoes the letterbox by subtracting the padding offset and dividing by the scale factor, mapping detection coordinates back to original 1920×1080 space.

**Q: Why bilinear interpolation and not bicubic?**

Bilinear is what Ultralytics uses during training. Matching inference preprocessing to training preprocessing avoids a distribution shift. Bicubic is marginally better visually but negligibly different for neural network features.

Speed-wise, bilinear does 4 texel reads + 3 multiplications per output pixel. Bicubic does 16 reads + more math. Not a huge cost on modern hardware but bilinear is simpler and matches what the model expects.

**Q: Your preprocessor uses `Int(srcXf.rounded(.down))` instead of `Int(srcXf)`. Why?**

`Int(x)` in Swift truncates toward zero: `Int(-0.5)` is `0`, not `-1`. That's wrong for bilinear interpolation when you have negative coordinates (which happen at the letterbox edges). The fix is `Int(srcXf.rounded(.down))` which does `floor`: `Int((-0.5).rounded(.down))` is `-1`.

This was caught during the original code review. Classic off-by-one pixel error that would have produced visible aliasing at the padding boundaries.

**Q: Why divide by 255.0 for normalization?**

YOLO26 (and most vision models) expects input in [0, 1] range. The model was trained with `(pixel_value / 255.0)` normalization. No mean subtraction or standard deviation scaling — different from ImageNet-style `(pixel - mean) / std` preprocessing.

Some YOLO variants use additional normalization (e.g., `(pixel - 0.5) * 2` for [-1, 1] range). Always check your specific model.

**Q: You're running preprocessing on CPU. What's the per-frame cost?**

At 1920×1080 input, bilinear letterbox + normalize + HWC→CHW takes ~5-8 ms on one Cortex-A78AE core. That's 10-15% of our 50 ms budget — meaningful but not the bottleneck.

GPU preprocessing via NPP (or a custom CUDA kernel) would cut this to <1 ms and pipeline it with inference on a separate stream. Worth ~5 ms saved per frame, which would let us increase the target FPS or use a slightly bigger model.

**Q: Are you CPU-bound or GPU-bound?**

At 19.4 FPS we're effectively matching the camera's 20 FPS rate. The hot path is GPU inference (28 ms) + CPU preprocessing + tracker (negligible) + rendering (if there are MJPEG clients connected).

If we switched to a camera outputting 30 FPS, we'd see where the bottleneck is. Based on the numbers:
- GPU inference: 28 ms → caps at ~35 FPS single-stream
- CPU preprocessing: ~6 ms per frame, mostly overlapping with GPU
- Tracker + rendering: ~2 ms

So we'd see ~30-32 FPS on a 30 FPS camera, with inference as the limit.

## Swift Language and Concurrency Questions

**Q: What's the runtime cost of Swift's structured concurrency?**

Actor hops: ~100-500 ns depending on whether the actor is currently busy (contended) or idle (uncontended). On cooperative thread pool scheduling, a hop is a context switch but no kernel transition.

`Task { }` overhead: ~1-2 µs to create and schedule a task. The task itself is small (around 200 bytes on top of its closure).

`withCheckedContinuation`: ~50-100 ns to set up the continuation + whatever the resume path costs.

`async let`: similar to `Task { }` but with structured scope guarantees.

None of these are problematic at 20 Hz. At kHz rates (e.g., audio processing) the overhead starts to matter.

**Q: You use `Mutex` from swift-synchronization. Is that a sleep-on-contention mutex or a spinlock?**

Looking at the swift-synchronization source, `Mutex` on Linux uses `pthread_mutex_t` under the hood. It's not a pure spinlock — if another thread holds the mutex, yours will go to sleep and the kernel will wake it when the mutex is released.

For our MetricsRegistry, contention is very rare (hot path is one writer, Prometheus render is periodic), so the mutex is almost always uncontended and acquires in ~20-50 ns. On contention it falls back to the kernel path which is microseconds.

**Q: Why not use `os.unfair_lock` (or equivalent) for lower-overhead locking?**

Swift's `Mutex` is the standard cross-platform primitive and maps to platform-specific "fair enough" locks. On Apple platforms it's an unfair lock; on Linux it's `pthread_mutex_t`. Both are fast enough for our use.

If profiling showed the mutex as a hotspot, we'd consider lock-free metrics (atomic counters) or a per-thread reader-writer scheme. Current numbers don't justify the complexity.

**Q: What happens if an actor method `await`s inside a `withLock`?**

You'd deadlock, because Swift actors don't let you release the actor executor while holding a synchronous lock. The fix is to do all your work inside `withLock` synchronously, then release the lock and `await` only after.

In our code, `MetricsRegistry` is not an actor and doesn't hold the lock across `await` calls. `DetectorEngine` is an actor but uses actor isolation itself, not a mutex. Clean separation.

**Q: Sendable story — did you have to mark many things `@unchecked Sendable`?**

One: `SendableGstPointer<T>`. Everything else is structurally Sendable:

- `Detection`, `Frame`, `Track`, `BBox`, `LetterboxInfo`, `StreamConfig` — all structs of Sendable fields
- `YOLOPreprocessor`, `YOLOPostprocessor`, `FrameRenderer` — stateless utility types
- `MetricsRegistry` (with its internal `Mutex<RegistryState>`) — final class marked Sendable, mutation protected by mutex
- Actor types (`DetectorEngine`, `DetectorState`, `RTSPFrameReader`, `GStreamerFrameReader`, `VLMClient`, `TrackFinalizer`) — Sendable by virtue of being actors

Swift 6's strict concurrency checking catches sendability issues at compile time. After the initial learning curve, the checks become background noise — you write Sendable types by default and get compile errors only on mistakes.

**Q: What's the ARC cost of passing `[UInt8]` frames between actors?**

`[UInt8]` is a Swift value type with copy-on-write semantics. Passing it as a parameter is a refcount bump + pointer copy — roughly 10-20 ns. Since we never mutate the buffer after creation, the refcount stays at 1 and no deep copies happen.

The `Frame` struct (containing the `[UInt8]` + width + height) is also a value type; passing `Frame` across an actor boundary is similarly cheap.

Where it *would* matter: if you had multiple Tasks mutating the same `[UInt8]`, each mutation would trigger a copy because the refcount would be >1. Our code avoids this by making the buffer immutable after construction.

**Q: Why not use `UnsafeBufferPointer` throughout to avoid ARC entirely?**

You could, and it would shave nanoseconds. But `UnsafeBufferPointer` is — well — unsafe: you have to manage the lifetime manually, and Swift's strict concurrency checking is less helpful (you lose Sendable guarantees on raw pointers).

We use `UnsafeBufferPointer` inside tight inner loops (preprocessor, postprocessor, tracker) where the performance matters. We use `[UInt8]` at the module boundary where Sendable and ownership semantics are important.

## Native Cross-Compile Questions

**Q: Why not just build inside an aarch64 container with QEMU? The tooling is there.**

We did that first — it took 25 minutes per build. The QEMU user-mode emulator interprets aarch64 instructions on an x86 host, which is slow for CPU-intensive work like compiling Swift. You pay a ~10-20× slowdown vs native.

Native cross-compile uses a real x86 Swift compiler that just happens to emit aarch64 object files. No emulation. Hence the 17.6× speedup.

**Q: Can you cross-compile from an ARM Mac (Apple Silicon) too?**

In principle yes — Apple Silicon runs aarch64 natively, so there's no emulation step. You'd need:

- Swift 6.2.3 toolchain for macOS aarch64
- A Linux aarch64 SDK (the `wendyos_aarch64` bundle would need to be rebuilt for macOS host)
- Same sysroot overlays for GStreamer, CUDA, etc.

The Swift compiler can target Linux aarch64 from a macOS host using the Swift Static Linux SDK. That's probably the cleanest path for a Mac developer. We didn't test it; our build host is an x86 Linux machine.

**Q: Why `--use-ld=lld` specifically? What about GNU ld or gold?**

LLD is multi-architecture aware by default. GNU ld and gold are usually built for a specific target; installing a "cross gold" means a separate binary per target architecture. LLD is one binary that handles everything via `--target=` / `-m` flags.

Also, LLD is faster and has better error messages than ld or gold for mixed object-file issues, which we had plenty of.

**Q: How do you know your `-isystem` paths are in the right order?**

`-isystem` adds a directory to the *system* include path, which has lower priority than `-I` (user include path) but higher than the compiler's built-in system paths. Order within `-isystem` matters only for headers with the same name in multiple directories.

Our include paths don't collide — each is a distinct subsystem (CUDA, GStreamer, GLib) — so ordering doesn't matter in practice. We ordered them by likely frequency of use (CUDA first, GStreamer second) but it's cosmetic.

**Q: How does the CUDA toolkit overlay interact with TensorRT?**

The overlay puts CUDA 12.6 headers + libs at `$SDK/usr/local/cuda/` (with a symlink `cuda -> cuda-12.6`). TensorRT's Swift bindings hardcode `-I/usr/local/cuda/include` in `unsafeFlags`. When compiled against the SDK, the clang resolves `/usr/local/cuda/include` through the sysroot to `$SDK/usr/local/cuda/include`, which our symlink resolves to `$SDK/usr/local/cuda-12.6/targets/aarch64-linux/include`.

The last one is where the actual headers live — CUDA's layout puts architecture-specific headers under `targets/<triple>/`. We made a relative symlink to that dir from `$SDK/usr/local/cuda-12.6/include` so the conventional path works.

**Q: Why didn't you just patch `tensorrt-swift` to use a header search path variable?**

Good question — that would be the proper fix. The reason we didn't:

1. `tensorrt-swift` is an external package (`wendylabsinc/tensorrt-swift`) — patching it means forking
2. The absolute path is inside `unsafeFlags` which is private to the package, not an exposed setting
3. Fixing it properly would require some kind of build-time variable substitution in SwiftPM, which doesn't exist today

Our workaround (overlaying the SDK and baking paths in toolset.json) is ugly but keeps us off a fork. The clean fix requires upstream changes to `tensorrt-swift` or SwiftPM.

**Q: How fragile is this setup? What breaks next time you upgrade the SDK?**

Several things could break:

1. **Swift version upgrade** — toolset.json schema is versioned; we'd need to check compatibility
2. **SDK regeneration from Yocto** — if the overlay paths change, our patches stop applying
3. **CUDA version upgrade** — we symlinked `cuda → cuda-12.6`. A new JetPack might bring CUDA 13 and the symlink needs updating
4. **tensorrt-swift update** — they could change the hardcoded include path; we'd need to re-verify
5. **New system library targets** — if you add, say, Pango or Cairo, you need to re-overlay headers and re-run

The right long-term fix is to land the overlay scripts in `meta-wendyos-jetson` as part of the SDK build recipe. Then the Yocto-built SDK ships with everything pre-installed, and our toolset.json can live in the recipe too.

## Jetson and Systems Questions

**Q: What is CDI and why does Jetson use it instead of standard Docker volume mounts?**

CDI (Container Device Interface) is an open standard for describing device passthrough to containers. It's a CNCF project originally pushed by NVIDIA, Intel, and others. A CDI spec file (`/etc/cdi/nvidia.yaml` in our case) lists:

- Device nodes to bind-mount (`/dev/nvidia0`, `/dev/nvhost-*`, `/dev/v4l2-nvdec`)
- Libraries and files to bind-mount from host to container
- Environment variables to set
- Optional hooks to run at container startup

Container runtimes (containerd, CRI-O, Podman) can consume CDI specs natively. When a container requests a CDI device (e.g., `--device nvidia.com/gpu=all`), the runtime applies all mounts/env from the spec.

Jetson uses CDI because:

1. The NVIDIA userspace libs (CUDA, cuDNN, TRT, V4L2 plugins) are tied to the host kernel driver version and can't be containerized
2. Jetson has extra device nodes (V4L2 decoder/encoder, VIC, GPIO) that standard Docker doesn't know about
3. The same CDI spec can be consumed by containerd, CRI-O, Podman — no Docker-specific runtime needed
4. Versioning: the CDI spec lives on the device and describes what's available. Images don't need to know about it

Standard Docker volume mounts (`-v /usr/lib:/usr/lib`) would also work but are fragile (overwrite container libraries, path conflicts, permission issues).

**Q: How does `nvidia-ctk` generate the CDI spec?**

`nvidia-ctk cdi generate` (the NVIDIA Container Toolkit's CDI generator) reads the host's NVIDIA installation and produces a CDI YAML. On standard Linux it uses `nvidia-smi` and the CUDA runtime to enumerate devices. On Jetson, there's a different code path that reads:

- L4T package manifest files at `/usr/share/nvidia-container-passthrough/`
- CSV files at `/etc/nvidia-container-runtime/host-files-for-container.d/`
- Device node info from `/dev/`

The CSV format is flat: `<type>, <path>` with types:

- `dev` — character or block device node
- `lib` — shared library file (dependencies are resolved via `ldd`)
- `sym` — symbolic link
- `dir` — directory (mounted recursively)
- `env` — environment variable (ignored by nvidia-ctk 1.16, handled by post-processing script)

Writing a new CSV, restarting `edgeos-cdi-generate.service`, and checking the result is the WendyOS CDI workflow.

**Q: Why is `/dev/v4l2-nvdec` a symlink to `/dev/null`?**

Because on L4T r36.x, `nvv4l2decoder` is a userspace V4L2 shim, not a kernel driver. The fd anchor at `/dev/v4l2-nvdec` just needs to be "a char device that opens successfully" — the `/dev/null` node (major 1, minor 3) fits that description. libv4l2 and its plugin mechanism (see Part 6) handle all the actual V4L2 operations in userspace.

This is documented (barely) in the udev rules file at `/etc/udev/rules.d/99-tegra-devices.rules` on the device. The comment says "for upstream linux on Tegra platforms." It's a compatibility shim for code that expects a V4L2 device.

**Q: Why does WendyOS use Yocto instead of Ubuntu or Debian?**

Yocto gives full control over what goes into the image: kernel version, init system, package set, boot loader, filesystem layout. For an embedded device with specific hardware (NVIDIA Jetson) and real-time constraints (maybe), that control matters.

Ubuntu for Jetson (NVIDIA's reference) is fine for development but ships with more than you need: full desktop, systemd services for laptops, python3, etc. Stripping that down is harder than building up from Yocto.

WendyOS-the-distro is a custom Yocto configuration aimed at container hosts: minimal base, containerd + wendy-agent, CDI spec generation, OTA updates. No desktop, no Python runtime, no bloat.

**Q: What's `libtegrav4l2` and why is it closed-source?**

It's NVIDIA's Tegra-specific V4L2 implementation. It wraps the NVDEC hardware block (and similarly NVENC) into a V4L2-compatible interface for userspace consumers like `libgstnvvideo4linux2.so`.

Closed-source because it's proprietary: the codec control logic, bitstream handling, and memory management are NVIDIA IP. The alternative would be a GPL kernel driver, which would require open-sourcing the codec logic — not something NVIDIA wants to do.

You can't audit it, but it's stable and fast. Your choices are to use it or to not use the NVDEC hardware.

## Production, Ops, and Monitoring Questions

**Q: How do you monitor this in production?**

The Swift detector exposes Prometheus metrics at `/metrics`:

- `deepstream_fps{stream}` — gauge, current frames per second per stream
- `deepstream_active_tracks{stream}` — gauge, confirmed tracks currently active
- `deepstream_frames_processed_total{stream}` — counter, cumulative frames
- `deepstream_detections_total{stream, class_}` — counter, per-class detections
- `deepstream_inference_latency_ms{stream}` — histogram with buckets 5, 10, 20, 50, 100, 200 ms
- `deepstream_total_latency_ms{stream}` — histogram
- `deepstream_decode_latency_ms{stream}` — histogram (not currently populated; future)
- `deepstream_preprocess_latency_ms{stream}` — histogram (not currently populated)
- `deepstream_postprocess_latency_ms{stream}` — histogram (not currently populated)

For production you'd run Prometheus scraping at 5-10 s interval, Grafana dashboard with alerts on:

- FPS dropping below threshold (camera issue, decoder crash)
- Inference latency > 50 ms p99 (GPU thermal throttle, kernel contention)
- No active tracks for N minutes (camera dropped or scene empty)
- No new VLM descriptions (sidecar down)

Beyond Prometheus, the Swift process logs structured JSON via `swift-log` → stdout → wendy-agent → OTLP or similar.

**Q: What about OpenTelemetry?**

Not wired up in this project, but `swift-otel` is a mature SwiftPM package. You'd add it, wrap the detection loop with spans, and point an OTLP exporter at your backend. Natural extension, ~30 lines of code.

**Q: Security considerations?**

The current setup is demo-grade:

- HTTP `/metrics` is plain — no TLS, no auth
- CORS: `Access-Control-Allow-Origin: *` — open to any browser
- VLM sidecar on localhost — not exposed externally, but neither is the Swift detector to outside the device's network
- Container runs as root inside the container — should use an unprivileged user
- CDI GPU passthrough gives the container access to all GPU devices

For production:

- mTLS reverse proxy in front of `/metrics` and `/stream`
- Authentication via the wendy-agent's existing OAuth enrollment flow
- Non-root user in the container (`USER` directive in Dockerfile; add equivalent via swift-container-plugin env)
- Restrict CDI device access to only what the app needs (nvidia-ctk supports finer-grained device sets)

**Q: What happens when the model weights change? How do you roll out a new ONNX?**

Current flow: the ONNX is a resource in the container image. `wendy run` packages it into a new image layer. Deploying a new model = deploying a new container version.

First launch after a model change rebuilds the TensorRT engine (~5-8 minutes on Nano). Subsequent starts load the cached engine. The cache file is derived from the ONNX path but doesn't include a hash of the ONNX content — if you replace the ONNX without changing the path, the cached engine becomes stale. Fix: hash the ONNX contents and include the hash in the engine filename.

Alternative: mount the model from a persistent volume instead of baking it into the image. That lets you hot-swap the model without rebuilding the image. Depends on your operational preferences.

**Q: Do you A/B test models?**

Not in this project, but the pattern would be:

1. Run two detector containers in parallel, each with a different model
2. Route N% of incoming RTSP streams to each container
3. Compare metrics (FPS, track count, detection count distributions)
4. Cut over when satisfied

At 1-2 cameras per Jetson, the compute budget makes parallel models infeasible. You'd do A/B testing across a fleet of Jetsons: version A on half, version B on the other half, compare aggregate metrics in Prometheus.

**Q: How does the VLM quality compare to a commercial API (GPT-4V, Claude Sonnet)?**

Qualitatively: Qwen3-VL-2B is noticeably worse than frontier multimodal models. It confuses make/model, hallucinates details that aren't in the image, and has a distinct "I'm hedging" style unless you constrain it with tight prompts.

The tradeoff: Qwen3-VL-2B runs on-device with no network, no per-query cost, no privacy concerns. For an edge deployment where you're processing hundreds of events per hour, an on-device model is often the right choice even if the quality is lower.

If you need frontier quality, you'd add a network-backed fallback: use Qwen3-VL-2B for fast local classification and only escalate "uncertain" crops to a cloud API. Cost + latency + privacy tradeoff per request.

## Final "gotcha" Questions

**Q: Did you hit any real deadlocks or data races?**

One close call: the original FFmpeg subprocess reader had a blocking `readData(ofLength:)` call running on the cooperative thread pool. Under two concurrent streams, both threads were blocked on the pipe read, starving the pool and blocking the detection loop from making progress. Fix: move the blocking read to a dedicated `DispatchQueue` (not the global pool).

Same pattern applies to `gst_app_sink_pull_sample` in the GStreamer reader — we knew about it from the FFmpeg experience and used a dedicated queue from the start.

**Q: What was the single most time-consuming debugging session?**

The `libv4l2` plugin discovery (Part 6). We spent multiple iterations convinced the platform didn't support hardware V4L2M2M decode — looking at udev rules, checking `/proc/devices`, verifying kernel modules. The actual answer (userspace plugin in a non-default location) was buried in NVIDIA forum posts from years ago and isn't in any official documentation we found.

Close runner-up: the x86 linker poisoning issue (PKG_CONFIG returning host paths for the cross-compile), which took four rebuild cycles to trace to the env var interaction between SwiftPM and the system pkg-config.

**Q: Is there anything you'd do differently if starting over?**

Three things:

1. **Skip the Docker build entirely.** Start with native cross-compile and swift-container-plugin from day one. The Docker path was a distraction that cost us days
2. **Extend the SDK upstream first.** Patch `meta-wendyos-jetson/recipes-core/images/wendyos-sdk-image.bb` to include GStreamer, CUDA, DeepStream dev headers in the shipped SDK. That would have skipped the manual overlay dance
3. **Understand libv4l2 before trying to use nvv4l2decoder.** The "device node mapped to /dev/null" confusion sent us down a wrong path. One hour of reading NVIDIA's multimedia stack docs would have saved several hours of debugging

**Q: What's the one line of code that you're proudest of?**

From `YOLOPostprocessor.swift`:

```swift
for slot in 0 ..< Self.maxDetections {
    let offset = slot * Self.valuesPerDetection
    let confidence = base[offset + 4]
    guard confidence >= confidenceThreshold else { continue }
    // ...
}
```

Not because it's clever — it's trivial. But because it replaces 80+ lines of per-class NMS with greedy suppression, and the simplification is a direct consequence of picking the right model. Sometimes the best Swift code is the Swift code you get to delete.

**Q: What's next?**

Short list:

1. **Zero-copy NvBufSurface pipeline** — consume GStreamer's NVMM-backed buffers directly into TensorRT without a system-memory round trip. Probably 2× FPS improvement
2. **cuDLA offload for YOLO** — free the main GPU for the VLM. Requires benchmarking whether Orin's DLA supports YOLO26 operators well
3. **Proper SDK extension in meta-wendyos-jetson** — land the GStreamer, DeepStream, CUDA dev packages in the Yocto recipe so new developers don't have to overlay
4. **Multi-camera support with batched inference** — build the TRT engine with batch 4 or 8, process multiple streams in lock-step
5. **Blog post follow-up: on-device fine-tuning** — can we run a LoRA adapter on Orin to specialize YOLO26 for a specific camera's vantage point?

Each of these is a post on its own.

---

