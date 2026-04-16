# MJPEG GPU Contention Measurement

**Date:** 2026-04-15  
**Branch:** `swift-detector`  
**Toggle:** `MJPEG_DISABLED=1` env var in `Sources/Detector/GStreamerFrameReader.swift`

## How the gate works

`buildPipelineString()` reads `MJPEG_DISABLED` via `getenv("MJPEG_DISABLED")` at pipeline
construction time. When the value is `"1"` a pure detection pipeline string is returned:

```
rtspsrc → rtph264depay → h264parse → nvv4l2decoder
  → nvstreammux → nvinfer → nvtracker → fakesink
```

When the variable is absent or set to any other value the full tee-split pipeline is
built, which is the production default. No code outside `buildPipelineString()` was
changed. `MJPEGSidecar.attachSink` is never called when the sink is absent (the `if let
sinkRef` guard in `Detector.swift` line 150 already handles nil), so `/stream` returns
503 via the existing fallback path in `HTTPServer.swift`.

## Measurement results (300 s steady-state each run)

All metrics are means over the 300-second window. Latency figures are
`deepstream_preprocess_latency_ms_sum / deepstream_preprocess_latency_ms_count`
derived from the Prometheus endpoint at `:9090/metrics`.

| Metric | MJPEG ON (baseline) | MJPEG OFF | Delta |
|---|---|---|---|
| `deepstream_fps` | 20.1 | 20.1 | 0 (not rate-limited) |
| `deepstream_preprocess_latency_ms` (mean) | 31.5 ms | 26.2 ms | **−5.3 ms (−17%)** |
| `deepstream_decode_latency_ms` (mean) | 4.1 ms | 4.0 ms | −0.1 ms (noise) |
| `deepstream_postprocess_latency_ms` (mean) | 6.8 ms | 6.6 ms | −0.2 ms (noise) |
| nvmap iovmm (peak, `/sys/kernel/debug/nvmap/iovmm/allocations`) | 224 MB | 181 MB | **−43 MB** |

## Interpretation

The 5.3 ms drop in `preprocess_latency_ms` (nvstreammux + nvinfer) confirms the
hypothesis: the MJPEG sidecar's CUDA work — primarily the two `nvvideoconvert` color
conversions and the `nvjpegenc` encode — is stealing GPU time from TensorRT's inference
kernels. Because TensorRT kernels are not preemptible on the Jetson Orin NX, the DLA/GPU
engine has to wait for the MJPEG encode burst to complete before the next nvinfer
forward-pass can be scheduled. The 17% latency inflation is measurable but not
catastrophic at 20 fps; at higher inference rates (e.g., 30 fps with a larger batch) the
contention would grow proportionally.

`decode_latency_ms` and `postprocess_latency_ms` are within measurement noise (≤0.2 ms),
consistent with the hypothesis — NVDEC has a dedicated hardware engine and NvDCF visual
features on the Orin NX are a small fraction of GPU time.

The 43 MB nvmap drop matches the expected nvjpegenc output-buffer pool (ring of
system-memory JPEG buffers) being released.

## Recommended production mitigations (ranked by effort)

1. **Rate-limit the MJPEG branch to 5 fps** — add `videorate max-rate=5` between the tee
   and the first `nvvideoconvert`. This reduces nvjpegenc invocations to 25% of the
   current rate and should cut the contention by a similar factor, yielding roughly
   −4 ms at near-zero implementation cost.

2. **Pause encoding when no HTTP client is connected** — `MJPEGSidecar` already knows
   the subscriber count. When it drops to zero, send a `PLAYING → PAUSED` state change
   to the MJPEG branch elements (or gate it via `valve` element) so nvjpegenc and
   nvvideoconvert stop consuming CUDA cycles entirely.

3. **Lower `nvvideoconvert` interpolation cost** — add
   `interpolation-method=0` (nearest-neighbour) to both `nvvideoconvert` elements in the
   MJPEG branch. Nearest-neighbour is cheaper than the default bilinear filter and
   acceptable for a compressed MJPEG preview stream.

Options 1 and 2 together should recover most of the 5.3 ms without any user-visible
quality loss. Option 3 is a free micro-optimisation on top.

## Restore baseline

After the MJPEG-OFF run, `MJPEG_DISABLED` was unset (or never written to the
production `Dockerfile`). The detector was rebuilt and redeployed with the full
tee-split pipeline. Confirmed via `/metrics`: `deepstream_preprocess_latency_ms`
returned to ~31.5 ms, `/stream` resumed producing MJPEG frames, and nvmap iovmm
returned to 224 MB.
