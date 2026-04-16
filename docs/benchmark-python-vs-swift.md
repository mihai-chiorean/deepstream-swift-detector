# Python vs Swift DeepStream Detector Benchmark

**Run date:** 2026-04-16 01:00:15 UTC
**Camera URL:** `rtsp://192.168.68.69:554/stream`
**Measurement window:** 300s per detector (warmup: 30s excluded from stats)
**Device:** 10.42.0.2


> **Python detector not runnable at benchmark time.**
> Reason: Container deepstream-vision runs monitor_proxy.py, not the DeepStream Python detector. The Python detector (detector/detector.py) has not been deployed to this device via wendy. The image deepstream-vision:latest on the device is the MJPEG/monitor proxy built from the root Dockerfile.committed — not the detector. To fix: build and push the detector image from detector/ directory with its own Wendy project, or deploy manually.
> All Python columns below show N/A.


## Device Info

| Property | Value |
|---|---|
| Jetson model | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| JetPack version | # R36 (release), REVISION: 4.4, GCID: 41062509, BOARD: generic, EABI: aarch64, DATE: Mon Jun 16 16:07:13 UTC 2025 |
| Total RAM | 7621 MB |

## Model Configuration

| Property | Python | Swift |
|---|---|---|
| Model file | yolo11n.onnx | yolo26n.onnx |
| Engine | model_b2_gpu0_fp16.engine (batch=2) | yolo26n_b2_fp16.engine (batch=1) |
| Classes | 80 | 80 |
| Input resolution | 640×640 | 640×640 |
| NMS | Custom YOLO lib (cluster-mode=2) | NMS-free end-to-end (cluster-mode=4) |
| Confidence threshold | 0.4 | 0.4 |

> **Warning:** The two detectors use different model architectures (YOLOv11n vs YOLOv2.6n).
> Detection count parity is informational only — differences in count do not indicate
> a correctness bug in either detector.

## Benchmark Results

### Throughput

| Metric | Python | Swift | Winner | Notes |
|---|---|---|---|---|
| FPS (mean) | N/A | 20.4 | N/A | Steady-state from `deepstream_fps` gauge |
| FPS (min) | N/A | 14.8 | — | |
| FPS (max) | N/A | 25.7 | — | |
| Total detections (window) | N/A | 20795 | — | Cannot compare (one or both detectors returned no data) |

### Cold-Start

| Metric | Python | Swift | Winner | Notes |
|---|---|---|---|---|
| Time to first detection | N/As | 0s | N/A | From task start → first non-zero `deepstream_detections_total` |

### Pipeline Latency

| Metric | Python | Swift | Winner | Notes |
|---|---|---|---|---|
| Mean pipeline latency | N/A ms | 50.08 ms | N/A | `deepstream_total_latency_ms_sum / _count` over measurement window |

### Memory — VmRSS (kB, sampled every 15s)

| Stat | Python | Swift | Winner | Notes |
|---|---|---|---|---|
| min | N/A | 551940.0 | — | |
| max | N/A | 574576.0 | — | |
| mean | N/A | 561481.6 | N/A | Lower = better |
| last | N/A | 574576.0 | — | |
| Trend | N/A | flat | — | >10% rise = potential leak |

### Memory — nvmap iovmm (kB, sampled every 15s)

> GPU unified memory allocations — the primary leak canary on Jetson.

| Stat | Python | Swift | Winner | Notes |
|---|---|---|---|---|
| min | N/A | 224484.0 | — | |
| max | N/A | 224484.0 | — | |
| mean | N/A | 224484.0 | N/A | Lower = better |
| last | N/A | 224484.0 | — | |

### GPU & Power (mean over measurement window)

| Metric | Python | Swift | Winner | Notes |
|---|---|---|---|---|
| GPU utilization | N/A% | 41.0% | N/A | `jetson_gpu_utilization_percent` |
| Total power draw | N/A W | 6.2 W | N/A | Sum of all rails from `jetson_power_watts` |
| System RAM used | N/A MB | 1857.8 MB | N/A | `jetson_ram_used_mb` (Jetson unified RAM) |

### Container Image

| Metric | Python | Swift | Winner | Notes |
|---|---|---|---|---|
| Image size (ctr) | N/A | 181.0 MiB | — | Compressed, from `ctr -n default images ls` |

## Raw Data

| Detector | CSV |
|---|---|
| Python | `N/A` |
| Swift | `/home/mihai/workspace/samples/deepstream-vision/docs/benchmark-data/swift.csv` |

Columns: `timestamp, fps, vmrss_kb, nvmap_kb, gpu_util_pct, gpu_power_w, gpu_ram_mb`

---

## What This Measures — And What It Doesn't

### What it measures
- **Steady-state throughput** (FPS) under real RTSP load from a live camera.
- **Per-process RSS** as a proxy for CPU-side memory growth. A rising trend
  over the 5-minute window is an early leak indicator but not conclusive.
- **nvmap iovmm allocations** — Jetson-specific GPU/iommu mappings. A
  monotonically increasing value here strongly suggests a GStreamer or
  DeepStream buffer leak.
- **End-to-end pipeline latency** from the histogram exposed by each detector.
  Includes RTSP jitter, decode, pre-process, inference, and post-process.
- **Power draw** across all voltage rails — useful for real-world deployment
  budgeting.

### What it does NOT measure
- **Detection accuracy / mAP.** FPS and detection count are throughput proxies,
  not quality metrics. The two detectors use *different model architectures*
  (YOLOv11n vs YOLOv2.6n), so count parity is informational.
- **Long-term memory stability.** 5 minutes is too short to catch slow leaks
  (e.g., one that grows at 1 MB/min would only be 300 MB — within normal
  variance). Re-run with `--duration 3600` for a meaningful leak test.
- **Multi-stream scalability.** Each detector is configured for a single RTSP
  stream in this benchmark.
- **Shared unified memory contention.** Jetson uses unified CPU/GPU memory.
  Running gpu-stats and the detector concurrently means both processes share
  the same physical RAM pool — context switches affect the numbers.
- **RTSP session variability.** The camera at `rtsp://192.168.68.69:554/stream` delivers a
  single session; packet loss or WiFi jitter affects both detectors, but not
  identically (different cold-start times mean different network conditions).
- **TensorRT engine build time.** The Python detector ships with a pre-built
  engine; if missing, the first run triggers an 8+ minute build that is
  excluded from the warmup window here.
- **VLM sidecar interaction.** VLM is disabled in this benchmark milestone.
  Re-run with VLM enabled to measure co-location overhead.

## Follow-up Runs

1. **Extended leak test** — `--duration 3600 --warmup 60` to observe nvmap
   and VmRSS over 1 hour. Add `--detector swift` only if the Python detector
   is still not buildable.
2. **Model parity** — export the same ONNX model (yolo26n or yolo11n) into
   both detectors and re-run; removes architecture variable from detection
   count comparison.
3. **Multi-stream** — update `streams.json` with 2–4 streams and benchmark
   FPS degradation curve per detector.
4. **Engine build time** — add a separate phase that forces engine rebuild
   (`--force-rebuild`) and records wall-clock build time; relevant for
   first-deploy CI timing.
5. **VLM co-location** — enable VLM sidecar and observe power, RAM, and FPS
   delta with the Python detector (since VLM was originally integrated only
   with Python).

