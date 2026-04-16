# Python vs Swift DeepStream Detector Benchmark

---

## Run 1 — Swift Only (main stream, sequential baseline)

> Python detector was not deployed at the time of Run 1. Swift ran alone on stream1 (1920x1080).
> All Python columns show N/A.

**Run date:** 2026-04-15 18:00 UTC
**Camera URL:** `rtsp://10.42.0.2:8554/relayed` (mediamtx relay of stream1 1920x1080)
**Measurement window:** 300s (warmup: 30s excluded)
**Device:** 10.42.0.2 — NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super

### Device Info

| Property | Value |
|---|---|
| Jetson model | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| JetPack version | R36 (release), REVISION: 4.4, GCID: 41062509 |
| Total RAM | 7621 MB |

### Model Configuration (Run 1)

| Property | Python | Swift |
|---|---|---|
| Model file | yolo11n.onnx (not deployed) | yolo26n.onnx |
| Engine | not deployed | yolo26n_b2_fp16.engine (batch=1) |
| Input resolution | N/A | 640x640 (infer), 1920x1080 (stream) |
| NMS | N/A | NMS-free end-to-end (cluster-mode=4) |
| Confidence threshold | N/A | 0.4 |

### Run 1 Results

| Metric | Python | Swift | Notes |
|---|---|---|---|
| FPS (mean) | N/A | 20.4 | Steady-state from `deepstream_fps` |
| FPS (min/max) | N/A | 14.8 / 25.7 | |
| Total detections (300s window) | N/A | 20,795 | |
| VmRSS (mean) | N/A | 561,482 kB | |
| VmRSS trend | N/A | flat | No leak signal |
| nvmap iovmm (mean) | N/A | 224,484 kB | Flat -- no leak. **Note: this Run-1 number predates Stage 2 (MJPEG branch removed); current Stage 2 steady-state is ~206 MB. The drop is from removing nvjpegenc + nvdsosd, not a relay-vs-direct effect.** |
| GPU utilization | N/A | 41.0% | |
| Total power draw | N/A | 6.2 W | |
| System RAM used | N/A | 1,858 MB | |
| Image size | N/A | 181 MiB | Compressed |
| Pipeline latency (mean) | N/A | 50.08 ms | |

---

## Run 2 -- Concurrent (Python + Swift, same 1920x1080 stream via mediamtx relay)

**Run date:** 2026-04-16 06:18-06:27 UTC
**Infrastructure:** mediamtx v1.17.1 relay at `rtsp://10.42.0.2:8554/relayed` fans out stream1 to both detectors simultaneously. Camera WiFi-only session limit is bypassed because mediamtx holds the single inbound session and multiplexes it.

> **STATUS (updated 2026-04-16 after Run 3): the "VIC starvation" framing
> below is SUPERSEDED.** Run 3 (300s steady-state with the same stack, after
> the Python detector was fixed) shows both detectors processing 6,349 vs 6,350
> frames over the window -- effectively identical throughput, ~21.2 FPS each.
> The 0.24 FPS Swift number recorded in Run 2 came from a transient state
> reached during/after the Python detector's silent crash, not from steady-state
> contention. The Run 2 numbers are preserved below as historical record; do
> not cite them as a conclusion about decoder contention. See Run 3 for the
> current head-to-head.

> **CRITICAL FINDING -- GPU Decoder Starvation:**
> The Jetson Orin Nano cannot sustain two concurrent DeepStream pipelines with
> hardware decoding (nvv4l2decoder) at full throughput. The first process to
> acquire the VIC hardware decoder dominates. In every run observed, Python
> (initialized first) ran at ~20 FPS while Swift was throttled to ~0.24 FPS.
> Swift eventually crashed due to resource starvation within 3-5 minutes.
> This is not a software bug -- it is a hardware constraint: the Jetson Orin Nano
> has one shared VIC NVDEC block for hardware H.264 decode.
>
> The comparison below reflects this reality: Python "wins" concurrently only
> because it grabbed the decoder first, not because of any algorithmic advantage.

**IMPORTANT CAVEAT -- Stream Setup:**
Run 2 originally targeted stream2 (640x360) for Python and stream1 (1920x1080) for Swift, to avoid the VIC contention. However, Mihai updated both detectors to use the mediamtx relay (stream1 via relay), making them use identical input. The model is now also identical (yolo26n_b2_fp16.engine, libnvdsparsebbox_yolo26.so) in both detectors. This is the right design for a fair model comparison, but it exposes the hardware decoder conflict directly.

### Model Configuration (Run 2)

Both detectors now use identical model and engine:

| Property | Python | Swift |
|---|---|---|
| Model file | yolo26n (via nvinfer_config.txt) | yolo26n |
| Engine | yolo26n_b2_fp16.engine (baked into image) | yolo26n_b2_fp16.engine (from persist cache) |
| Parser lib | libnvdsparsebbox_yolo26.so | libnvdsparsebbox_yolo26.so |
| Input stream | rtsp://10.42.0.2:8554/relayed (stream1, 1920x1080) | same |
| Metrics port | 9092 | 9090 |

### Concurrent Throughput (stable samples, both running simultaneously)

| Metric | Python | Swift | Notes |
|---|---|---|---|
| FPS (mean, stable window) | 19.97 | 0.24 | Swift starved -- GPU decoder contention |
| FPS (min) | 19.5 | 0.24 | |
| FPS (max) | 20.2 | 0.24 | |
| Total detections (5 stable samples) | ~1,900 est | ~4 | Swift barely processed any frames |
| Time to first detection | ~0s (already running) | ~0s (already running) | |

### Per-Process Memory (5 stable concurrent samples)

| Stat | Python VmRSS (kB) | Swift VmRSS (kB) | Notes |
|---|---|---|---|
| mean | 642,703 | 560,770 | Python higher: Flask+pyds+GI stack |
| min | 641,504 | 559,344 | |
| max | 644,488 | 562,248 | |
| Trend | flat | flat (then crash) | No leak before crash |

> **Python vs Swift VmRSS:** Python uses ~82 MB more RSS than Swift (643 MB vs 561 MB).
> The delta is attributable to Flask, PyGObject, GI, and prometheus-client overhead --
> not the DeepStream pipeline itself.

### nvmap iovmm -- SHARED (critical leak canary)

| State | nvmap (kB) | Notes |
|---|---|---|
| Solo Swift (Run 1) | 224,484 (flat) | No leak, stable over 300s |
| Concurrent (both running) | 421,520 (flat) | ~2x the solo value -- expected for two pipelines |
| After Swift crash | 215,420 | Returns to near-solo level when one process exits |

> **nvmap verdict: CLEAN.** The shared nvmap allocations doubled when both pipelines ran
> concurrently (224,484 -> 421,520 kB), which is expected behavior for two independent
> GPU buffer pools. Critically, the value was **flat** during the stable window -- no
> monotonic growth. This rules out the old GStreamer buffer leak pattern. The Python
> detector does not have the leak.

### Concurrent GPU & Power (shared)

| Metric | Solo Swift | Concurrent | Notes |
|---|---|---|---|
| GPU utilization | 41.0% | 21.0% avg (peaks 74%) | Spiky -- GPU serializes the two pipelines |
| Total power draw | 6.2 W | 5.82 W | Lower than expected -- Swift barely running |
| System RAM used | 1,858 MB | 2,590 MB | +732 MB for Python container stack |

### Container Images

| Detector | Image | Size |
|---|---|---|
| Python | `detector:latest` | 518.7 MiB |
| Swift | `detector-swift:latest` | 181.0 MiB |

> Swift image is 2.9x smaller than Python. Python image includes: Flask, pyds (DeepStream
> Python bindings), PyGObject, GStreamer Python introspection, prometheus-client, numpy,
> OpenCV, Pillow, OpenTelemetry. Swift image is a pre-compiled binary with minimal runtime.

### Concurrent Raw Data

| Detector | CSV |
|---|---|
| Python (concurrent) | `/home/mihai/workspace/samples/deepstream-vision/docs/benchmark-data/concurrent-python.run2.csv` |
| Swift (concurrent) | `/home/mihai/workspace/samples/deepstream-vision/docs/benchmark-data/concurrent-swift.run2.csv` |

Columns: `timestamp, fps, vmrss_kb, nvmap_kb, gpu_util_pct, gpu_power_w, gpu_ram_mb`

(Run 2 CSVs were renamed with a `.run2.csv` suffix when Run 3 reused the
canonical concurrent-{python,swift}.csv paths.)

---

## Run 3 -- Concurrent (Python detector stable, fair head-to-head)

**Run date:** 2026-04-16 20:36-20:42 UTC (sampler clock recorded local PDT
in the CSVs with a `Z` suffix; the wall-clock window in true UTC is as stated
here).
**Camera URL:** `rtsp://10.42.0.2:8554/relayed` -- both detectors reading the
same stream1 (1920x1080) via mediamtx, verified in each detector's
`streams.json`.
**Measurement window:** 300s steady-state (warmup: 30s excluded).
**Detectors:** both already running under containerd at sample start; neither
restarted by the benchmark harness.
**Sampling:** custom non-disruptive sampler (`/tmp/bench_sampler.sh`) was used
in place of `scripts/benchmark.sh` because the latter would have overwritten
this report and had restart logic incompatible with the "do not touch the
running detectors" constraint. Sampler does the same column collection at the
same `SAMPLE_INTERVAL=15s`; CSVs are byte-compatible with the Run 2 schema.

> **HEADLINE FINDING -- approximate parity, no starvation.**
> With both detectors stable on the same relay stream, throughput is
> effectively equal: Python processed 6,349 frames in the 300s window, Swift
> processed 6,350. Both averaged ~21.2 FPS computed from frame deltas, with
> instantaneous FPS gauges scattered between ~16 and ~28 FPS at each 15s
> sample tick (this is camera/jitter variance, not a stable difference between
> detectors). The "VIC NVDEC contention dominates" conclusion from Run 2 is
> NOT reproduced here -- under steady state, the two pipelines coexist
> without starvation. Concurrent contention is REAL (the GPU oscillates 0-98%
> between sample ticks and instantaneous FPS spikes are wider than in Run 1's
> solo run) but at this duty cycle the hardware multiplexes adequately.

### Model Configuration (Run 3)

Identical to Run 2 -- both detectors run yolo26n_b2_fp16.engine via
libnvdsparsebbox_yolo26.so on the relayed stream1. No model/stream confounds.

| Property | Python | Swift |
|---|---|---|
| Engine | yolo26n_b2_fp16.engine (batch=1) | yolo26n_b2_fp16.engine (batch=1) |
| Parser lib | libnvdsparsebbox_yolo26.so | libnvdsparsebbox_yolo26.so |
| Input stream | rtsp://10.42.0.2:8554/relayed | same |
| Metrics port | 9092 | 9090 |
| Process pattern | `detector.py` | `/app/Detector` |

### Concurrent Throughput (20 samples over 300s)

| Metric | Python | Swift | Notes |
|---|---|---|---|
| FPS gauge mean | 20.54 | 20.42 | `deepstream_fps`, instantaneous gauge |
| FPS gauge p50 | 19.95 | 19.94 | |
| FPS gauge p95 | 28.09 | 25.02 | High p95 reflects bursty decode + camera jitter |
| FPS gauge min/max | 16.34 / 28.66 | 14.52 / 28.21 | Per-sample range |
| Frames in window (delta) | 6,349 | 6,350 | From `deepstream_frames_processed_total` -- the most reliable throughput number |
| Avg FPS from frame delta | 21.16 | 21.17 | Frame delta / 300s |
| Detections in window (delta) | 19,100 | 18,512 | Class set differs slightly (Python has labelled classes; both run yolo26n) |
| Pipeline latency (`deepstream_total_latency_ms`) | not exported by Python build | 50.00 ms (mean over 6,350 frames) | **NOT glass-to-glass.** This is `gst_util_get_timestamp()` inter-frame interval at the probe -- a throughput-smoothness signal, not a latency one. |

### Per-Process Memory (20 samples over 300s)

| Stat | Python VmRSS (kB) | Swift VmRSS (kB) | Notes |
|---|---|---|---|
| mean | 796,605 | 676,268 | Python +120 MB over Swift; bigger gap than Run 2's +82 MB. Python has been up longer than at Run 2 sample time. |
| min / max | 796,244 / 796,796 | 676,268 / 676,268 | Both effectively flat over 5 min |
| drift (last - first) | +0.07% | 0.00% | No leak signal in either |

> **VmRSS gap is wider in Run 3 than Run 2 (120 MB vs 82 MB).** Most likely
> Python has accumulated steady-state heap during longer uptime; the leak
> drift over the 5-minute window is essentially zero in both detectors, so
> this is not a leak. Swift's VmRSS is also bigger than its Run 1 reading
> (561 MB) -- this aligns with the long-uptime pattern noted in HANDOFF
> (post-3hr Swift VmRSS was ~668 MB; Run 3 sample is 676 MB after even longer
> uptime). Hypothesis: both detectors hit a plateau a couple of hundred MB
> above their just-started baselines. Not measured to convergence.

### nvmap iovmm -- SHARED (critical leak canary)

| Stat | Value | Notes |
|---|---|---|
| min | 421,520 kB | |
| max | 421,520 kB | |
| mean | 421,520 kB | |
| Drift over 300s | 0.00% | Bit-exact across 20 samples |

> **nvmap is rigorously flat at 421,520 kB across 300s.** Same value as Run 2
> (which was also 421,520 kB), so concurrent-mode steady-state for the two
> pipelines reproduces. No leak in either detector. The shared pool stayed
> within sampling resolution of the prior reading -- this is the strongest
> evidence Run 3 produces.

### Concurrent GPU & Power (shared, 20 samples)

| Metric | mean | p50 | p95 | min | max | Notes |
|---|---|---|---|---|---|---|
| GPU utilization | 56.9% | 60.5% | 98% | 0% | 98% | Spiky -- GPU serializes the two pipelines but no longer at the cost of one being starved |
| Total power draw | 7.62 W | 7.52 W | 8.43 W | 6.60 W | 8.47 W | Higher than Run 2 (5.82 W) because both detectors are doing real work |
| System RAM used | 2,154 MB | 2,154 MB | 2,157 MB | 2,149 MB | 2,157 MB | +296 MB above Solo-Swift baseline (1,858 MB), consistent with Python container's footprint |

### Sampler Output (20 ticks @ 15s)

```
[+0s]   py_fps=28.09 sw_fps=14.52 nvmap=421520kB gpu=97% pwr=8.23W
[+15s]  py_fps=20.16 sw_fps=21.22 nvmap=421520kB gpu=36% pwr=7.28W
[+30s]  py_fps=19.92 sw_fps=19.95 nvmap=421520kB gpu=53% pwr=7.40W
[+45s]  py_fps=17.78 sw_fps=17.08 nvmap=421520kB gpu=52% pwr=7.52W
[+60s]  py_fps=19.04 sw_fps=25.02 nvmap=421520kB gpu=49% pwr=7.67W
[+75s]  py_fps=21.56 sw_fps=15.02 nvmap=421520kB gpu= 1% pwr=7.32W
[+90s]  py_fps=19.83 sw_fps=19.93 nvmap=421520kB gpu=22% pwr=8.43W
[+105s] py_fps=19.95 sw_fps=19.97 nvmap=421520kB gpu=98% pwr=7.56W
[+120s] py_fps=20.92 sw_fps=15.02 nvmap=421520kB gpu= 0% pwr=8.07W
[+135s] py_fps=20.55 sw_fps=18.34 nvmap=421520kB gpu=81% pwr=7.79W
[+150s] py_fps=19.98 sw_fps=24.54 nvmap=421520kB gpu=56% pwr=7.44W
[+165s] py_fps=19.94 sw_fps=24.82 nvmap=421520kB gpu=97% pwr=8.47W
[+180s] py_fps=28.66 sw_fps=28.21 nvmap=421520kB gpu=76% pwr=8.19W
[+195s] py_fps=16.34 sw_fps=24.88 nvmap=421520kB gpu= 0% pwr=6.60W
[+210s] py_fps=19.90 sw_fps=19.94 nvmap=421520kB gpu=82% pwr=7.36W
[+225s] py_fps=20.03 sw_fps=19.50 nvmap=421520kB gpu=65% pwr=7.12W
[+240s] py_fps=19.63 sw_fps=19.73 nvmap=421520kB gpu= 3% pwr=7.28W
[+255s] py_fps=18.67 sw_fps=18.92 nvmap=421520kB gpu=98% pwr=7.75W
[+270s] py_fps=18.43 sw_fps=21.75 nvmap=421520kB gpu=96% pwr=7.43W
[+285s] py_fps=21.51 sw_fps=20.04 nvmap=421520kB gpu=76% pwr=7.52W
```

The instantaneous FPS gauge oscillates per tick but the cumulative frame delta
shows both detectors processed essentially the same number of frames. Both
were alive at the start of sampling and at the end (verified via `ctr -n
default tasks ls` post-run).

### Run 3 Raw Data

| Detector | CSV |
|---|---|
| Python (Run 3) | `/home/mihai/workspace/samples/deepstream-vision/docs/benchmark-data/concurrent-python.csv` |
| Swift (Run 3) | `/home/mihai/workspace/samples/deepstream-vision/docs/benchmark-data/concurrent-swift.csv` |

Columns: `timestamp, fps, vmrss_kb, nvmap_kb, gpu_util_pct, gpu_power_w, gpu_ram_mb`

> Note on CSV timestamps: the sampler writes `date '+%Y-%m-%dT%H:%M:%SZ'`
> which formats local time but tags it with `Z`. The recorded timestamps are
> PDT (UTC-7), not UTC. True UTC of the sampling window is as stated above.
> This bug is inherited from `scripts/benchmark.sh` and affects Run 2's CSVs
> equivalently.

### What Run 3 changes vs Run 2

1. **Throughput parity established.** Python and Swift process the same number
   of frames per second under concurrent load. There is no decoder-monopoly
   effect.
2. **Run 2's "VIC starvation" claim is superseded.** That number reflected a
   transient post-crash state of the Python detector, not steady-state
   contention.
3. **VmRSS gap (Python > Swift) reproduced and widened.** ~120 MB vs Run 2's
   82 MB. Both consistent with Python's runtime overhead; both flat over the
   5-min window.
4. **nvmap reproduces bit-exact at 421,520 kB.** This is reassuring: the
   shared GPU buffer-pool footprint is reproducibly the same across
   concurrent runs.
5. **GPU utilization is bursty (0-98% per 15s sample) but mean 57%.** Both
   pipelines share the GPU; under this duty cycle they fit. A heavier model,
   higher resolution, or VLM co-location would likely break this.

### Caveats specific to Run 3

- 20 samples is small. The instantaneous FPS spread (16-28 FPS gauge) is
  wider than the steady-state truth (~21.2 FPS from frame deltas) -- prefer
  the frame-delta number as the headline.
- Python's `deepstream_total_latency_ms` was not exported in this build (the
  histogram metric returned empty in baseline/final reads). Swift's reported
  50.00 ms mean is **inter-frame interval** (`gst_util_get_timestamp` at the
  probe, see HANDOFF and Caveats), not glass-to-glass latency.
- Run 3 does not measure pure-compute nvinfer latency. The
  `preprocess_latency_ms` and `inference_latency_ms` histograms straddle
  nvstreammux queueing -- they are throughput proxies, not compute. Real
  YOLO26n FP16 compute on Orin Nano is ~8-12 ms per Nsight runs (not
  re-measured live for Run 3).
- `deepstream_total_latency_ms` is the same kind of pseudo-metric in Run 3
  as in earlier runs (see Caveats below). It is reported here for continuity,
  not to be cited as a latency SLO.

---

## Top-Level Comparison Summary

| Metric | Swift (solo Run 1) | Run 2 concurrent (superseded) | Run 3 concurrent Python | Run 3 concurrent Swift |
|---|---|---|---|---|
| FPS (gauge mean) | 20.4 | 19.97 / 0.24 (Py / Sw) | 20.54 | 20.42 |
| FPS (frame-delta over window) | n/a | n/a | 21.16 | 21.17 |
| Frames in 300s window | n/a | n/a | 6,349 | 6,350 |
| VmRSS (mean, kB) | 561,482 | 642,703 / 560,770 | 796,605 | 676,268 |
| VmRSS drift over window | flat | flat (then crash) | +0.07% | 0.00% |
| nvmap iovmm (kB) | 224,484 (flat) | 421,520 (flat, shared) | 421,520 (flat, shared) | 421,520 (flat, shared) |
| GPU utilization (mean) | 41.0% | 21% avg (peaks 74%) | 56.9% (shared, 0-98% range) | shared |
| Power (W) | 6.2 | 5.82 | 7.62 | shared |
| Image size | 181.0 MiB | 518.7 / 181.0 MiB | 518.7 MiB | 181.0 MiB |

> Run 1 is solo Swift, Run 2 is the superseded concurrent run with Python
> crashing partway through, Run 3 is the fair concurrent run with both
> detectors stable on the same relay stream.

---

## Key Findings

### 1. Concurrent coexistence: partial success

Both processes launched, registered metrics endpoints, and produced output simultaneously. The infrastructure (mediamtx relay, separate ports 9090/9092, wendy app isolation) worked correctly. Coexistence at the OS/container level is proven.

### 2. Hardware decoder contention is the binding constraint

The Jetson Orin Nano's VIC NVDEC block is a shared, non-multiplexed resource. Two nvv4l2decoder instances compete for it. The first process to grab the decoder runs at full FPS; the second is throttled to ~0.24 FPS (1 frame per ~4 seconds). This is not a software issue and cannot be worked around without switching one pipeline to software decoding (avdec_h264) or using NVDEC engine isolation (not available on Orin Nano).

### 3. nvmap stayed flat -- no leak

The shared GPU buffer pool (nvmap iovmm) was stable at 421,520 kB while both detectors ran. No monotonic growth observed during the stable window. The Python detector does not carry the old leak pattern.

### 4. Python is heavier than Swift

- Image: 518.7 MiB vs 181.0 MiB (2.9x)
- VmRSS: 643 MB vs 561 MB (+82 MB; Python runtime overhead)
- Deploy time: Python image built in ~32 minutes (cross-compiled arm64 via QEMU buildx); Swift built in minutes via a pre-built base image

### 5. Model/stream parity achieved for future runs

Both detectors now use yolo26n_b2_fp16.engine + libnvdsparsebbox_yolo26.so + mediamtx relay. Architecture and stream resolution are no longer confounds.

---

## Caveats

1. **Hardware decoder starvation dominates the concurrent result.** The FPS disparity (20 vs 0.24) is a first-come-first-served VIC NVDEC allocation artifact, not a software quality difference.
2. **Concurrent run had instability.** Swift crashed 3-5 times during testing due to GPU resource exhaustion. The stable sample window was approximately 5 minutes before each crash.
3. **Python solo FPS not measured in a clean isolated run.** The concurrent run provides the first Python FPS data: ~20 FPS on the relay stream (same as Swift solo), suggesting comparable algorithmic throughput when the GPU decoder is not shared.
4. **nvmap sample count is small** (5-8 stable concurrent samples). A 1-hour run is needed for high confidence.
5. **VLM was disabled** throughout both runs per explicit requirement.

---

## What This Measures -- And What It Doesn't

### What it measures
- Steady-state throughput (FPS) under real RTSP load from a live camera
- Per-process RSS as a proxy for CPU-side memory growth
- nvmap iovmm allocations (Jetson GPU buffer pool -- flat = no leak)
- End-to-end pipeline latency (Run 1 Swift: 50.08 ms mean)
- Power draw across all voltage rails
- Concurrent coexistence at container level

### What it does NOT measure
- Detection accuracy / mAP
- Long-term memory stability (5 min is too short for slow leaks)
- Software decode throughput (a fair concurrent comparison would use avdec_h264 for one pipeline)
- TRT engine build time (Python engine was pre-baked into the image)
- VLM sidecar interaction (VLM disabled)

---

## Follow-up Runs

1. **Software decode for one pipeline** -- run Python with avdec_h264 (CPU decode) and Swift with nvv4l2decoder (hardware decode). This avoids VIC contention and gives each pipeline full throughput at the cost of ~25% more CPU on the Python side.
2. **Python solo FPS baseline** -- run Python alone on the relay stream for 300s to establish a clean solo baseline comparable to Swift Run 1.
3. **Extended leak test** -- `--duration 3600 --warmup 60` solo runs to observe nvmap over 1 hour.
4. **VLM co-location** -- enable VLM sidecar and measure power, RAM, FPS delta.
5. **Model parity sequential run** -- both detectors now use yolo26n; run solo sequential benchmark to isolate Python vs Swift pipeline overhead from model differences.

---

## Updated Files

| File | Path |
|---|---|
| Benchmark harness | `/home/mihai/workspace/samples/deepstream-vision/scripts/benchmark.sh` |
| This report | `/home/mihai/workspace/samples/deepstream-vision/docs/benchmark-python-vs-swift.md` |
| Run 1 Swift CSV | `/home/mihai/workspace/samples/deepstream-vision/docs/benchmark-data/swift.csv` |
| Run 2 Python concurrent CSV (renamed) | `/home/mihai/workspace/samples/deepstream-vision/docs/benchmark-data/concurrent-python.run2.csv` |
| Run 2 Swift concurrent CSV (renamed) | `/home/mihai/workspace/samples/deepstream-vision/docs/benchmark-data/concurrent-swift.run2.csv` |
| Run 3 Python concurrent CSV | `/home/mihai/workspace/samples/deepstream-vision/docs/benchmark-data/concurrent-python.csv` |
| Run 3 Swift concurrent CSV | `/home/mihai/workspace/samples/deepstream-vision/docs/benchmark-data/concurrent-swift.csv` |
