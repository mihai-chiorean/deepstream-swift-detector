# MJPEG Decoupling: Design Options

## Problem

The Swift detector's GStreamer pipeline has a `tee` after `nvtracker` that splits into a detection branch and an MJPEG branch. The MJPEG branch (`nvvideoconvert → nvdsosd → nvvideoconvert → nvjpegenc → appsink`) shares the single Orin Nano GPU with nvinfer. TensorRT kernels aren't preemptible; the branches serialize at the GPU execution level.

Measured contention (MJPEG ON vs OFF, same pipeline, same RTSP source):

| Metric | MJPEG ON | MJPEG OFF | Delta |
|---|---|---|---|
| deepstream_preprocess_latency_ms (mean) | 31.5 | 26.2 | −5.3 ms (−17%) |
| nvmap iovmm peak | 224 MB | 181 MB | −43 MB |
| deepstream_fps | 20.1 | 20.1 | 0 (source-limited) |

Contention is real (~5 ms per frame, or 1/10 of a 50-ms frame budget). Worse: the branch encodes regardless of whether an HTTP client is watching.

Goal: make the inference loop NOT wait for MJPEG. If the stream lags / buffers / drops, that's fine — detections must go through at source rate (20 fps) regardless.

## Option 1 — Subscriber-gated valve + 5 fps rate cap

Insert a `valve drop=true` at the head of the MJPEG tee branch. Valve is closed by default; `MJPEGSidecar` opens it on first HTTP client subscribe, closes on last disconnect. Add `videorate max-rate=5` after the valve so when encoding IS happening, it's at ≤5 fps instead of 20 fps.

**Effects:**
- No HTTP client → zero MJPEG GPU work. Inference sees no contention.
- With a viewer → ~4× less contention (5 fps vs 20 fps encode rate).
- ~30 lines of Swift + GStreamer.

**Weaknesses:**
- When a viewer IS connected, residual ~1.3 ms contention (5 fps encode still hits the GPU).
- Rate cap via `videorate` has a known GLib ABI issue (`g_once_init_leave_pointer undefined`) in our container's CDI-injected GLib — may need workaround.

## Option 2 — Separate CUDA stream priority

nvinfer and nvjpegenc currently share the same CUDA context / stream. Put nvjpegenc (and the two `nvvideoconvert`s on the MJPEG branch) on a lower-priority CUDA stream so the driver preempts at kernel boundaries in favor of nvinfer.

**Effects:**
- Theoretically reduces contention without rate-capping.
- No architectural changes; no subscriber-awareness needed.

**Weaknesses:**
- DS plugins don't expose CUDA stream priority — would need a custom element, patched DS source, or env-var experimentation.
- TRT kernels are large and non-preemptible mid-kernel anyway; priority preemption only helps at kernel boundaries (coarse-grained).
- Medium effort; unclear payoff.

## Option 3 — Separate process (same machine)

Spin up an independent service that subscribes to the RTSP relay, re-decodes, re-encodes to MJPEG, and serves HTTP. No code-sharing with the detector.

**Effects:**
- Process isolation: own CUDA context, own GPU memory pool, own thread scheduler. Detector is bulletproof.
- Detector's pipeline loses the MJPEG branch entirely — simpler.

**Weaknesses:**
- 2× decode (detector + mjpeg service both pull from relay and decode).
- To render detection overlays, the mjpeg service needs either (a) its own `nvinfer` (double inference — terrible) or (b) a side-channel feed of detections from the detector (gRPC / Redis / websocket / shared memory). Complexity explosion.
- High effort; bad GPU-time tradeoff unless overlays are acceptable to omit.

## Option 4 — Decoupled detection stream (architecturally cleanest)

Detector pipeline has NO encoder. It only:
1. Runs detection + tracking (current detection branch only).
2. Publishes detections (per-frame JSON / protobuf / MessagePack) on a stream: gRPC server-streaming, WebSocket, or Redis pub/sub.

Separate lightweight video service:
1. Subscribes to the RTSP relay.
2. Re-publishes as WebRTC (or MJPEG, or HLS — pick based on latency/compat).
3. No overlays.

Browser:
1. Connects to the video stream (WebRTC `<video>` or MJPEG `<img>`).
2. Connects to the detection stream (WebSocket).
3. Draws bboxes on an overlay `<canvas>` synchronized by `frameNum`.

**Effects:**
- Detector is minimal and fast: no encoder, no sidecar. No possibility of contention.
- WebRTC gives sub-100ms glass-to-glass latency (vs MJPEG's ~500ms).
- Multiple viewers trivially (WebRTC multi-peer or HLS fan-out).
- Detection stream is reusable (logging, analytics, alerts).

**Weaknesses:**
- Highest effort: new service, new protocol, browser JS for overlay sync, frame-number clock alignment.
- Needs a plan for time-sync: bbox `frameNum` must match `<video>` decoded frame. Typically doable via SEI NAL units, or by advancing the overlay based on video playback time + known fps.
- If WebRTC, adds server-side signaling (SDP exchange, STUN/TURN maybe — inside a LAN probably not needed).

## Decision framing

| Dimension | Opt 1 | Opt 2 | Opt 3 | Opt 4 |
|---|---|---|---|---|
| Effort | Low | Medium | High | Highest |
| Contention when watched | Medium | Medium-Low | Zero | Zero |
| Contention when idle | Zero | Medium | Zero | Zero |
| Latency to viewer | ~500 ms | ~500 ms | ~500 ms | <100 ms |
| Multi-viewer | Constrained | Constrained | Yes | Yes |
| Overlay quality | Pixel-perfect | Pixel-perfect | Requires side-channel | Canvas-synced (risk: drift) |
| Architectural debt repaid | None | None | Some | Significant |

## Current recommendation (before team review)

Ship Option 1 immediately; leave Option 4 on the roadmap for when we have more than one viewer or need sub-200ms latency or want a first-class analytics pipeline.
