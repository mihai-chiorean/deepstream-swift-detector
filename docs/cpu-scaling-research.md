# CPU scaling research — what actually uses CPU, and whether multi-stream reveals a language-layer asymmetry

**Status:** research notes, 2026-04-16. No code has been changed based on this; it's the design input for a future multi-stream benchmark.

**Prompted by:** the single-stream 1.96× CPU ratio measured in Run 3 (Swift 26.6% of one core, Python 52.1% of one core, identical fps at ~21). That ratio is small enough to read as micro-optimization; the question is whether the gap widens non-linearly when you scale stream count. If it does, that's a blog headline; if not, the 1.96× is a footnote.

---

## Q1 — What still consumes CPU?

Both pipelines: rtspsrc → rtph264depay → h264parse → nvv4l2decoder → nvstreammux → nvinfer → nvtracker → fakesink. Swift skips the post-tracker `nvvideoconvert → nvdsosd` stage after Stage 2; Python used to have a cv2-based frame extract path that was removed for relay parity but the probe code remains.

| Stage | Where the CPU goes |
|---|---|
| `rtspsrc` / `rtpjitterbuffer` | Per-packet TCP read, RTP demux, jitter-buffer reorder + timer thread. Non-zero on 1080p25. |
| `rtph264depay` / `h264parse` | NAL scanning, SPS/PPS tracking, AU framing. Pure software. |
| `nvv4l2decoder` | VIC NVDEC hardware does the decode. CPU side = V4L2 ioctl + buffer queue/dequeue threads. Small but real. |
| `nvstreammux` | CPU pointer juggling for NVMM buffers, batch formation, pad-sync timer. Allocates `NvDsBatchMeta` per batch on CPU side. |
| `nvinfer` pre-process | GPU (CUDA): `NvBufSurfTransform` letterbox + normalize. Negligible CPU past launch. |
| `nvinfer` forward | GPU (TensorRT FP16). CPU = enqueue + `cudaStreamSync`. |
| `nvinfer` post-process (`NvDsInferParseYolo26`) | **CPU.** Compiled C++ shared lib called per batch on the streaming thread. Cheap (300 slots, simple float compare loop) but not free. Same code in both detectors. |
| `nvtracker` (NvDCF) | Mostly CUDA; NvDCF maintains tracker-state + Hungarian assignment on CPU per frame. Documented as partly CPU-bound on Jetson. |
| **Pad probe** | **The split.** Swift: `wendy_nvds_flatten` walks `obj_meta_list` in C, copies POD into Swift `Detection`, yields to AsyncStream. Bounded, no allocator pressure. Python: `pyds.NvDsObjectMeta.cast(...)`, attribute access per object, `defaultdict` increments, list-of-dicts construction, `Counter` per object — all under the GIL. |
| HTTP / WebSocket server | Swift: Hummingbird on its own NIO EventLoop. Python: Flask threaded server + `prometheus_client.Histogram.observe` (internally locked Python). |
| Python-only overhead | `mjpeg_client_count` lock per frame; periodic `gc.collect()` every 500 frames; idle MJPEG path still costs per-frame Python. |

**The 1.96× ratio is consistent with the probe callback being the dominant CPU work past the fixed HW-driver floor.** pyds object-meta walks plus `Counter.inc()` per detection per frame are ~1 ms-class; Swift's C-shim flatten is sub-100 µs. At ~21 fps × ~3 detections you'd expect roughly the gap we measure.

---

## Q2 — GIL and scaling

1. **pyds pad-probe holds the GIL.** pyds is a pybind11 binding; the probe callback bounces into Python via `PyEval_AcquireThread`. The whole `osd_sink_pad_buffer_probe` in `detector.py` (lines ~768-935) executes with the GIL held. **All streams in one Python process serialize through one GIL-holding probe per buffer.** In-process multi-stream scaling for Python is GIL-capped at roughly one core of probe work.
2. **The custom parser `.so` (`libnvdsparsebbox_yolo26.so`) is GIL-free in both detectors.** nvinfer calls it from its own native streaming thread; pyds isn't on that path.
3. **NvDCF tracker** runs internal CUDA + a CPU thread pool. Generally not single-core-bound but doesn't scale linearly past ~2 streams on Orin Nano (per NVIDIA forum reports).
4. **NVDEC on Orin Nano** is rated ~4K30 H.264. 1080p25 = ~1/8 capacity; roughly 6-8 concurrent streams before NVDEC saturates as a resource. (Run 2's 0.24 fps starvation with just 2 detectors was likely process-level V4L2 driver state thrash, not raw NVDEC throughput — concurrent decoders on Run 3 coexist cleanly.)
5. **nvinfer YOLO26n FP16 on Orin Nano** compute budget is ~8-12 ms per batch (per `HANDOFF.md` §8 caveat). At batch=1 that's ~80-120 fps theoretical; the 21 fps ceiling is the camera, not the model. GPU has headroom.

**Swift can saturate cores. Python cannot.** That is the defensible asymmetric-scaling claim — if the experiment backs it up.

---

## Q3 — Are we actually under-utilized?

Mostly yes, with one thing to watch.

- 26.6% / 52.1% of ONE core, summed across all threads per process. A 6-core system has ~590% headroom by simple ps math.
- The direct `top -H` check on 2026-04-16 21:11 found **zero threads above 80% across 4 samples** — no single-thread bottleneck right now.
- The 1.96× ratio is probably real, not a context-switch artifact. pyds Python-side work per frame is known to be multi-hundred-µs to ms; at 21 fps × 3 detections that arithmetic explains the gap.
- **What could lie:** if Python's probe runs on one thread at ~20% of a core at K=1, it scales linearly to ~80% at K=4 and 100% at K=5 — **fps collapses for ALL streams past that knee.** Swift's probe thread is <10% at K=1 so the knee is 5-6x further out. The per-thread view in the multi-stream test would make this visible.

Counter-evidence to the "plenty of headroom" story: some of the 26.6% Swift baseline is irreducible driver/IPC + kernel time from RTP/V4L2 ioctls and streaming-thread context switches. That fraction does not shrink with more cores.

---

## Proposed experiment — multi-stream fan-out

**Goal:** plot detector CPU% (whole-process, % of one core) vs K parallel 1080p25 streams, K ∈ {1, 2, 4, 6, 8}, Swift vs Python.

**Source — use synthetic sources to avoid mediamtx as the bottleneck:**

```bash
# On Jetson, spin K independent publishers into mediamtx:
for i in $(seq 1 8); do
  gst-launch-1.0 -q videotestsrc is-live=true pattern=ball ! \
    video/x-raw,width=1920,height=1080,framerate=25/1 ! \
    x264enc tune=zerolatency bitrate=4000 speed-preset=ultrafast ! \
    rtspclientsink location=rtsp://127.0.0.1:8554/synth$i &
done
```

Point each detector's `streams.json` at `rtsp://127.0.0.1:8554/synth1..K`. Rebuild `nvstreammux` with `batch-size=K`.

**Required Swift change:** `buildPipelineString` in `GStreamerFrameReader.swift` currently wires one rtspsrc branch. Needs to wire K input branches into one streammux. Python's detector already has a `for i, stream in enumerate(enabled_streams)` loop — bumping `streams.json` is enough.

**Per K, 60 s warmup + 300 s sample, every 5 s:**

```bash
# Per-process CPU (% of one core)
awk '{print $14+$15+$16+$17}' /proc/<PID>/stat

# Per-thread breakdown — captures GIL-bound saturation
top -H -b -n 1 -p <PID> | head -30

# GPU + NVDEC + thermal
tegrastats --interval 1000 --logfile /tmp/tegra-K$K.log

# fps per stream
curl -s http://10.42.0.2:9090/metrics | grep deepstream_fps
```

**Total wall time:** ~30 min per detector per K-step × 5 K-steps × 2 detectors ≈ 2-3 hours including rebuilds.

**Headline chart:** X = K streams, Y = "process CPU % of system" (100% = one core, 600% = all cores). Two lines: Swift, Python. Secondary annotation: fps/stream. The inflection point is the story.

**Confounds to call out:**
- NVDEC saturates around K=6-8; both detectors plateau in fps regardless of CPU. Mark on chart.
- mediamtx CPU rises with K — use gst-launch publishers as sources instead of fanning one camera through mediamtx.
- Thermal throttling on a 30+ min sustained run on a fanless Orin Nano. Watch `tegrastats` temps; abort >85 °C.
- Disk at 92% — clean before pushing K-variant images.
- Swift code change to support batch-size=K pipeline branches is out-of-scope for the current read-only state — this experiment requires eng work before it can run.

---

## Recommendation

**Half-day experiment, run it before the blog headline.**

The single-stream 1.96× CPU gap is honest but underwhelming as a port-justification story — "we use half the CPU on a stream that uses <30% of one core" reads as micro-optimization. The K-vs-CPU curve, if Swift stays linear while Python's GIL-capped probe thread plateaus around K=2-3, is the load-bearing chart. If Python doesn't plateau (pyds drops the GIL more than expected during nvinfer wait, or NVDEC saturates before the probe does), that's also a finding, and the blog should NOT claim "Swift scales further."

**One sentence for the blog:** the multi-stream curve is the headline; the single-stream 1.96× ratio is a footnote that motivates the test, not a conclusion that stands alone.
