# CLAUDE.md

Orientation for a Claude Code session working in this repo. Read this first.

## What this repo is

A Swift port of an NVIDIA DeepStream object-detection pipeline on Jetson Orin Nano. The canonical thesis is in the blog post at <https://mihaichiorean.com/blog/swift-doing-nvidias-job/> — read that for the *why*. This doc is about the *where things live* and *what not to change without thinking*.

**Published sibling:** [`wendylabsinc/samples`](https://github.com/wendylabsinc/samples) at `samples/deepstream-vision/` is the upstream; this repo was extracted from it via `git subtree split`. When in doubt about whether a non-code artifact belongs in this repo vs. upstream, default to upstream — this repo is the *focused reference implementation* that goes with the blog post. Anything author-private, product-specific, or session-state lives upstream, not here.

## Layout

```
detector-swift/      ← Swift detector. Primary subject of the blog post.
  Sources/Detector/       → Swift entry point + HTTP/WebSocket server (Hummingbird + NIO)
  Sources/CGStreamer/     → C shim that walks NvDsBatchMeta linked lists
  Sources/CNvdsParser/    → YOLO26 bbox parser (.so loaded by nvinfer at runtime)
  ARCHITECTURE.md         → pipeline diagram, metrics, cross-compile + CDI mechanics
  Dockerfile              → aarch64 container; pre-built Swift binary baked in
  nvinfer_config.txt      → TensorRT engine + custom parser wiring
  tracker_config.yml      → NvDCF config
  streams.json            → RTSP source config
  wendy.json              → Wendy CLI deploy descriptor

detector/            ← Python sibling for comparison. Same engine, same parser, same tracker.
gpu-stats/           ← tegrastats Prometheus exporter. Sidecar.
vlm/                 ← Qwen3-VL inference server. Sidecar, not required for detection.

monitor.html         ← Browser monitor. WebRTC video + WebSocket detections overlay.
monitor_proxy.py     ← Same-origin reverse proxy in front of the detector + mediamtx.
scripts/             ← Benchmark harness + utility scripts.
start.sh             ← Local bring-up helper.

docs/                ← Technical reference (benchmarks, CPU scaling, tracker research,
                       MJPEG decoupling notes, mediamtx relay config, raw CSV data).

OPERATIONS.md        ← Build/deploy recipes. Expect JetPack r36.4.4 + DeepStream 7.1.
README.md            ← Public-facing entry point.
```

## The hard constraints

1. **Swift must not touch NVMM surfaces.** The whole blog post is about why. If you find yourself wanting to `gst_buffer_map` a video frame from Swift, stop — that's the mistake the post documents.
2. **Pad probe on the `nvtracker` src pad, not anywhere else.** The `DetectionStream` contract assumes metadata arrives in the C probe callback from that specific point in the graph. Moving it breaks the AsyncStream fan-out and the reconnect logic.
3. **The class filter lives on `nvinfer`, not `nvtracker`.** DS 7.1's `nvtracker` does not expose `operate-on-class-ids` as a GObject property (this was caught at build time). Use `filter-out-class-ids=<semicolon-list>` on `nvinfer`. See `docs/tracker-and-transport-research.md`.
4. **Never use `--restart-unless-stopped` on a pipeline change that hasn't settled.** A tight crash loop on NVDEC will wedge driver state; wendy-agent's own `ContainerMonitor` will also restart on abnormal exit, so escape-loops are easy to enter.
5. **Keep the `deepstream_reconnects_total` metric wired up.** The rtspsrc auto-reconnect path in `detector-swift/Sources/Detector/GStreamerFrameReader.swift` relies on this being visible to catch "fps frozen at last value" silently.

## Test environment we ran against

- Jetson Orin Nano 8 GB
- **JetPack r36.4.4** (DeepStream 7.1)
- **WendyOS** (Yocto-based, in active development — CDI specs and SDK contents change between releases)
- RTSP camera at 1080p H.264, 25 FPS source → detector averages ~21 FPS end-to-end
- YOLO26n FP16 TensorRT engine + `libnvdsparsebbox_yolo26.so`

If the environment differs, expect plumbing work: CDI bind-mounts for DeepStream runtime libs, a Swift cross-compile SDK with DeepStream headers + linker stubs. See the "OS plumbing that wasn't turnkey" section of `README.md` for specifics.

## Numbers that should not drift silently

- **~21 FPS** end-to-end (camera-limited at 25 FPS).
- **26.6 %** of one CPU core for Swift; **52.1 %** for the Python sibling on the same workload.
- **`nvmap` pool: 421,520 kB** flat — matched to the kilobyte across two concurrent runs. This is DeepStream's pool, not the host process's; neither language contributes.
- **VmRSS: Swift 676 MB, Python 797 MB**, both flat over 5-minute windows.

If any of these move materially on a change you made, investigate before committing.

## Session-private docs are gitignored

The `.gitignore` excludes:

- `docs/HANDOFF.md`, `docs/blog-swift-detector-port.md`, `docs/blog-angle-proposal.md`, `docs/reliability-audit.md`, `docs/billboard-scanner-design.md`, `detector-swift/PORT_PLAN.md`

These exist in the upstream `wendylabsinc/samples` working tree; they are intentionally kept out of this public repo and its history. If you regenerate any of them here, they will not commit — by design. The `pre-cleanup-20260420` tag on this repo points at the last commit before these were filtered out; it exists as a recovery anchor only.

## Typical tasks + where to look first

- **Pipeline change:** `detector-swift/Sources/Detector/GStreamerFrameReader.swift` → `buildPipelineString`. Mirror the change in `detector/detector.py` if you want to keep the head-to-head honest.
- **Metadata extraction change:** `detector-swift/Sources/CGStreamer/nvds_shim.c` → the `wendy_nvds_flatten` walker. Keep it allocation-free on the hot path.
- **Metrics / HTTP change:** `detector-swift/Sources/Detector/HTTPServer.swift`, `Metrics.swift`, `DetectionBroadcaster.swift`.
- **Container / deploy:** `detector-swift/Dockerfile`, `detector-swift/wendy.json`. For non-Wendy hosts see `OPERATIONS.md`.
- **Benchmark:** `scripts/benchmark.sh`; output under `docs/benchmark-data/`.
- **Tracker / model tuning:** `detector-swift/nvinfer_config.txt`, `detector-swift/tracker_config.yml`. Research notes in `docs/tracker-and-transport-research.md`.

## When you change anything meaningful

- **Re-run the review pipeline for the blog post upstream** if you're also touching the narrative: `npm run review:once <path>` in `~/workspace/mihaichiorean-com/`. See the upstream `samples` repo's working tree for the draft source.
- **Don't silently regress the CPU ratio.** If Swift's per-process CPU climbs above ~30 % on the baseline workload, find out why before shipping.
- **Don't add a dependency you wouldn't want in a container that ships only a Swift binary + a small C shim.** This is a 181 MiB image on purpose.

## Out of scope for this repo

- Anything about Mihai's writing voice, blog angles, session orientation for `edge-builder-1`, or WendyOS agent internals. Those live upstream.
- Multi-tenant / multi-camera fan-out. The pipeline is single-stream by design at the moment; adding K-stream support is noted as open work in the blog post.
- VLM orchestration beyond the existing `vlm/` sidecar. If you need deeper VLM integration, work in `wendylabsinc/samples` where the broader context lives.
