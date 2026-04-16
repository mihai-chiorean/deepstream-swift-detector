# Session Handoff — Swift Detector on Jetson Orin Nano

**Date:** 2026-04-15 (Stage 2 shipped earlier today; Python benchmark rerun needed)
**Branch:** `swift-detector` (10 commits ahead of origin; uncommitted working tree — see §12)
**Repo root:** `/home/mihai/workspace/samples/deepstream-vision/`
**Role of this doc:** orient a fresh Claude session. Do NOT paraphrase the authoritative sources linked here; go read them. This file exists so a cold session can find the right door in under 2 minutes.

---

## 1. TL;DR

- **Swift detector is running and stable on the Jetson** (`ctr -n default tasks ls` on `10.42.0.2`: `detector-swift` PID 9754, RUNNING). Metrics show 20 fps, 3 active tracks, 217k frames processed since last restart, nvmap flat at 206 MB, VmRSS 668 MB.
- **Stage 2 landed earlier today (2026-04-15):** MJPEG branch deleted, video served by `mediamtx` WebRTC (WHEP on `:8889/relayed/whep`), monitor.html consumes it via `<video>` + `<canvas>` overlay with PTS sync via `requestVideoFrameCallback`. `monitor_proxy.py` proxies `/detector/*`, `/gpu/*`, `/webrtc/*` on `:8001` from the dev host.
- **mediamtx is RTSP-relaying the single-session camera** (`192.168.68.69:554`) to `rtsp://10.42.0.2:8554/relayed`, so the detector and anyone else pulls from the relay — never from the camera directly.
- **What's broken:** Python detector dies silently during TRT warmup (~90 s into sampling). `:9092/metrics` returns nothing right now. Hypothesis: pyds segfault on first inference. Benchmark report §"Concurrent window: inconclusive" documents this.
- **Docs inconsistency (resolved 2026-04-15):** `ARCHITECTURE.md`, `mjpeg-decoupling-synthesis.md`, and `benchmark-python-vs-swift.md` previously framed ~201 MB nvmap as a relay-vs-direct win; that framing was refuted as measurement noise. Fixed — all three now state ~206 MB steady-state under Stage 2, with the ~23 MB drop from 224 MB attributed to removing the MJPEG branch (nvjpegenc + nvdsosd surfaces), not the relay path. A direct-vs-relay A/B confirmed both paths hit the same 206 MB pool.
- **Next obvious move:** pick ONE of (a) get Python detector stable for a fair concurrent benchmark, (b) commit the ~20 uncommitted files into 2–3 logical commits (Stage 2 impl + docs + benchmark data).

---

## 1b. Mini-glossary (alphabetical)

Short definitions for terms used without first-use expansion elsewhere in this file.

- **CDI** — Container Device Interface. A Kubernetes/containerd standard (implemented by `nvidia-ctk`) that declares host bind-mounts and env vars to inject into a container at task start. `wendy.json`'s `"type": "gpu"` triggers it here.
- **cluster-mode=4** — an `nvinfer` config knob selecting the bounding-box clustering strategy. Mode 4 is "no clustering" — required for YOLO26's 1:1 detection head which already emits final boxes.
- **DPB** — Decoded Picture Buffer. H.264 decoder-internal pool of recently-decoded reference frames; `nvv4l2decoder` sizes one per pipeline.
- **ICE** — Interactive Connectivity Establishment. The WebRTC handshake step that enumerates candidate IP:port pairs and negotiates which actually routes between peers. Relevant because mediamtx advertises both `10.42.0.2` (USB-C) and `192.168.68.70` (WiFi); the browser picks the WiFi one.
- **nvbufsurface** — NVIDIA's GPU-memory image buffer type used throughout DeepStream; the thing `gst_buffer_map` must NOT touch.
- **NvDCF** — NVIDIA Discriminative Correlation Filter. The tracker implementation used by `nvtracker` (via `libnvds_nvmultiobjecttracker.so`), configured in `tracker_config.yml`.
- **NVMM** — NVIDIA Memory Management. The GPU/VIC-visible memory pool DeepStream surfaces live in, distinct from CPU-side system RAM.
- **nvmap** — the NVIDIA kernel memory allocator backing NVMM. Usage is visible at `/sys/kernel/debug/nvmap/iovmm/clients`. Steady-state on this stack is ~206 MB.
- **RVFC** — `requestVideoFrameCallback`, a Chromium/Safari browser API that fires per-frame with media-time metadata. Used by `monitor.html` to align canvas bbox draws with the WebRTC video frame's PTS. Firefox has no RVFC → fallback to `requestAnimationFrame` with degraded sync.
- **WHEP** — WebRTC-HTTP Egress Protocol. A simple HTTP-POST-based signalling scheme (offer in, SDP answer out as `201 Created`) used by mediamtx to expose a WebRTC consumer endpoint at `:8889/relayed/whep`.

---

## 2. Device topology

```
                 WiFi (badgers den, 192.168.68.0/22)
                  │
  ┌───────────┐   │   ┌────────────────────────────┐      ┌────────────────────────┐
  │  macbook  │───┼───│  dev host (this machine)   │──────│  Jetson Orin Nano 8GB  │
  │ (Mihai)   │   │   │  edge-builder-1            │      │  WendyOS / L4T r36.4.4 │
  └─────┬─────┘   │   │  x86_64 Linux              │      │                        │
        │         │   │  wlp*   192.168.68.75      │      │  wlP1p1s0 192.168.68.70│
        │         │   │  usb*   10.42.0.1/24 (NCM) │──────│  usb0     10.42.0.2/24 │
        │         │   │  tailscale 100.68.x        │ USB-C│                        │
        │         │   └──────────┬─────────────────┘      └───────────┬────────────┘
        │         │              │                                    │
        │ WiFi    │   :8001 HTTP │ (serves monitor.html + proxy)      │ :8554 RTSP, :8889 WebRTC (HTTP+UDP)
        │ (mDNS)  │              │                                    │ :9090 detector-swift, :9091 gpu-stats
        └─────────┴──────────────┘                                    │ :5000 registry, :22 ssh, (:9092 Python — down)
                  │                                                   │
                  │                                                   │
                  │                           WiFi (single-session)   │
                  │                           RTSP pull               │
                  │                                          ┌────────┴────────┐
                  └──────────────────────────────────────────│  RTSP camera    │
                                                             │  192.168.68.69  │
                                                             │  rtsp://:554/   │
                                                             │  stream1 1080p  │
                                                             └─────────────────┘
```

**Reachability from macbook:**

| Target | Path | Works? |
|---|---|---|
| `edge-builder-1.local:8001` (monitor.html + proxy) | WiFi → dev host | yes |
| `10.42.0.2:*` (Jetson USB-C) | **NOT routed from macbook** | no — must go via proxy |
| `10.42.0.2:8189/udp` (WebRTC media) | ICE adds `192.168.68.70:8189` → WiFi to Jetson direct | yes — UDP flows peer-to-peer over WiFi, only WHEP HTTP goes via proxy |
| `192.168.68.70` (Jetson WiFi) | direct | yes, but WiFi is flaky |

**Rule of thumb:** dev-host ↔ Jetson uses `10.42.0.2` over USB-C (1–3 ms RTT, rock-solid). Camera forces the Jetson onto WiFi for one RTSP pull. Macbook uses `edge-builder-1.local:8001` and never talks to the Jetson directly except for WebRTC UDP media.

For one-time USB-C NetworkManager setup, see OPERATIONS.md §"USB-C setup".

---

## 3. Access cookbook

Copy-paste commands. When in doubt, prefer USB-C (`10.42.0.2`).

### SSH to the Jetson
```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR root@10.42.0.2
```
Fallback over WiFi: `root@192.168.68.70` (drops frequently).

### Metrics endpoints
```bash
# Swift detector
curl -s http://10.42.0.2:9090/metrics | grep '^deepstream_'
# gpu-stats (tegrastats exporter)
curl -s http://10.42.0.2:9091/metrics | grep '^jetson_'
# Python detector (currently DOWN — empty response)
curl -s http://10.42.0.2:9092/metrics | head
# mediamtx HTTP API
curl -s http://10.42.0.2:9997/v3/paths/list
```

### Via the dev-host proxy (same-origin under :8001)
```bash
curl -s http://localhost:8001/detector/metrics | grep '^deepstream_'
curl -s http://localhost:8001/gpu/metrics | grep '^jetson_'
```

### RTSP relay (always prefer over direct camera)
```
rtsp://10.42.0.2:8554/relayed             # what both detectors should consume
```
Camera is single-session (§10); mediamtx holds the single pull and fans out.

### WebRTC WHEP (video for the browser)
```
http://10.42.0.2:8889/relayed/whep        # direct POST offer, 201 answer
http://localhost:8001/webrtc/relayed/whep # via proxy (what monitor.html uses)
```
ICE candidates advertise `10.42.0.2:8189` and `192.168.68.70:8189`; UDP media flows peer-to-peer — the proxy only signals WHEP HTTP.

### Detections WebSocket
```
ws://10.42.0.2:9090/detections                 # direct (from dev host)
ws://localhost:8001/detector/detections        # via proxy (from macbook)
```
Proxy does raw TCP splice on `Upgrade: websocket` (see `monitor_proxy.py:42`).

### Browser monitor
```
http://edge-builder-1.local:8001/   # from macbook, mDNS
http://192.168.68.75:8001/          # same, bare IP if mDNS is flaky
http://localhost:8001/              # from dev host
```

### Camera — don't hit it directly
Single-session RTSP at `rtsp://jetson:jetsontest@192.168.68.69:554/stream1`. mediamtx holds the session. Never curl / ffprobe this directly — you'll fight mediamtx for the one slot.

### Jetson WiFi restoration (after reboot)
```bash
ssh root@10.42.0.2 'nmcli con up "badgers den"'
```
The Jetson's NM connection reliably fails to come up after reboot on WendyOS. Camera relay needs this.

### Detector logs
```bash
ssh root@10.42.0.2 'tail -f /var/lib/wendy-agent/storage/detector-cache/detector.log'
```

### Stop / start detector
Full recipe in OPERATIONS.md §"Stopping / starting the detector". Summary:
```bash
# stop
ssh root@10.42.0.2 'ctr -n default tasks rm detector-swift 2>/dev/null
                    ctr -n default containers rm detector-swift 2>/dev/null'
# build + push + start
cd /home/mihai/workspace/samples/deepstream-vision/detector-swift
source ~/.local/share/swiftly/env.sh
swift build --swift-sdk 6.2.3-RELEASE_wendyos-device_aarch64 -c release
WENDY_AGENT=10.42.0.2 wendy run -y --detach
# apply cgroup cap (MANDATORY — see §10)
ssh root@10.42.0.2 'CONTAINER=detector-swift /usr/local/bin/detector-cap'
```

---

## 4. Pipeline flows (end-to-end)

### Detection path (GStreamer NVMM, zero buffer-map)
```
RTSP camera (single-session, 1080p H.264)
  │  rtsp://192.168.68.69:554/stream1
  ▼
mediamtx (on Jetson, :8554) ──── relays & fans out ────┐
  │  rtsp://10.42.0.2:8554/relayed                     │
  ▼                                                    │
Swift detector container (detector-swift, :9090)       │
  rtspsrc → rtph264depay → h264parse                   │
    → nvv4l2decoder            (NVDEC, NVMM out)       │
    → nvstreammux batch-size=1 1920x1080               │
    → nvinfer config=/app/nvinfer_config.txt           │
         (YOLO26n FP16 engine + libnvdsparsebbox_yolo26.so, cluster-mode=4)
    → nvtracker name=wendy_tracker (NvDCF)             │
    → fakesink                                         │
        ▲                                              │
        │ pad probe on nvtracker.src (nvds_shim.c)     │
        │ reads NvDsBatchMeta as GstMeta, extracts     │
        │ GST_BUFFER_PTS → WendyDetection[]            │
        │ → @_cdecl → DetectionStream.ingest()         │
        │ → AsyncStream<DetectionFrame>                │
        ▼                                              │
Detector.runStreamDetectionLoop                        │
  → Metrics (histograms, counters)                     │
  → DetectionBroadcaster.distribute(_)                 │
      → per-client AsyncStream continuation map        │
      → HummingbirdWebSocket /detections               │
                                                       │
                                                       │
### Video path (Stage 2, no encoder in detector)       │
                                                       │
mediamtx WebRTC publisher  ◄──────────────────────────┘
  (same relayed H.264, RTP repacketized)
  WHEP HTTP :8889 (signalling)
  UDP :8189 (media, ICE direct)
  │
  ▼
monitor_proxy.py (dev host, :8001)  ── only signals HTTP proxy
  │
  ▼
Browser <video id="webrtc-video">  (UDP media peer-to-peer over WiFi)

### Overlay path (PTS-accurate sync in the browser)
Browser WebSocket /detector/detections  (via proxy splice)
  → detectionRing[]  (last ~60 messages keyed by ptsNs)
  → video.requestVideoFrameCallback(metadata.mediaTime)
      → mediaTime * 1e9 + ptsOffsetNs ≈ frame PTS
      → pick detection with min |ptsNs - estPts|
      → canvas.drawImage + fillRect bboxes (source-space coords, CSS stretches)
```

Source of truth for the detection pipeline string:
`/home/mihai/workspace/samples/deepstream-vision/detector-swift/Sources/Detector/GStreamerFrameReader.swift` (`buildPipelineString`).

---

## 5. DeepStream + Containers + Wendy — how it hooks together

This section is the one Mihai explicitly called out as necessary for a cold session.

### How the Swift detector container is built

Cross-compile happens on the dev host (x86_64), not on the Jetson:

```bash
swift build --swift-sdk 6.2.3-RELEASE_wendyos-device_aarch64 -c release
```

The SDK bundle lives at `~/.swiftpm/swift-sdks/6.2.3-RELEASE_wendyos-device_aarch64.artifactbundle/`. It contains the WendyOS sysroot (DeepStream headers, CUDA stubs, GStreamer, Tegra headers — see §5 "WendyOS SDK story" below) and a `toolset.json` that passes `--target=aarch64-unknown-linux-gnu` to clang for both compile and link.

Then `wendy run` wraps `docker buildx` against a **project-local `detector-swift/Dockerfile`** (the root `Dockerfile` is from a different earlier try — do not confuse):

1. buildx builds an arm64 image in the `wendy-builder0` BuildKit container on the dev host.
2. Image is pushed to the Jetson's containerd registry at `10.42.0.2:5000`.
3. `wendy run` tells the wendy-agent on the Jetson to `ctr -n default run` a task from that image.
4. `wendy.json` declares `"type": "gpu"` which triggers CDI injection at task start.

**There is no `swift package build-container-image` path anymore.** `OPERATIONS.md` was rewritten 2026-04-15 to document the `wendy run` workflow as the build path; the old CLI is gone. Verified empirically.

### What's IN the container image

Content list (from `detector-swift/Dockerfile`):
- `ubuntu:24.04` base
- Pre-built Swift `Detector` binary + Swift runtime `.so`s from `swift-runtime/`
- `nvinfer_config.txt`, `tracker_config.yml`, `labels.txt`, `streams.json`
- `libnvdsparsebbox_yolo26.so` (custom YOLO26 bbox parser, cross-compiled against DS includes)
- `yolo26n_b2_fp16.engine` (TRT engine, pre-built on device and copied back)
- `yolo26n.onnx` (fallback; nvinfer rebuilds engine from ONNX if engine missing)
- (Removed in Stage 2: `libgstvideodrop-stub.so`, `cairo-stage/` — no longer needed without nvdsosd/videodrop)

**All of these are `COPY`'d into the image at Dockerfile build time — not mounted via `wendy run --resources` at runtime.** Updating the engine, the parser, or any config therefore requires a container rebuild + redeploy; `wendy run` does not re-read these from disk at task start. (CDI injection at task start handles host-provided libs — DeepStream, CUDA, NVDEC — separately; see the next subsection.)

**Explicitly NOT in the image:**
- DeepStream SDK runtime libs
- CUDA, cuDNN, TensorRT runtime
- GStreamer NVMM plugins
- `libnvv4l2.so`, Tegra userspace multimedia shim
- Any `/dev/nv*` or `/dev/v4l2-nvdec` nodes

### How DeepStream actually gets into the container at runtime — CDI

On the Jetson, `nvidia-ctk` (NVIDIA Container Toolkit) has generated `/etc/cdi/nvidia.yaml` from CSV files at `/etc/nvidia-container-runtime/host-files-for-container.d/` (`l4t-deepstream.csv`, `drivers.csv`, `l4t.csv`, `l4t-gstreamer.csv`).

When `wendy.json` declares `"type": "gpu"`, wendy-agent tells containerd to apply the CDI spec. That CDI spec bind-mounts, at task start:

- DeepStream 7.1 libs from `/opt/nvidia/deepstream/deepstream-7.1/lib/` → same path inside container
- GStreamer NVMM plugins (`nvv4l2decoder`, `nvstreammux`, `nvinfer`, `nvtracker`, etc.)
- CUDA runtime libs, cuDNN, TensorRT runtime
- `libnvv4l2.so` + `libv4l2_nvvideocodec.so` + `libtegrav4l2.so` (the userspace NVDEC shim chain)
- Device nodes: `/dev/v4l2-nvdec`, `/dev/nvidia0`, `/dev/nvhost-*`
- Env vars: `GST_PLUGIN_PATH`, `LIBV4L2_PLUGIN_DIR`, `LD_LIBRARY_PATH`

(`LIBV4L2_PLUGIN_DIR` is injected via a `fix-cdi-gstreamer-paths.sh` post-processing script that patches the YAML — `nvidia-ctk 1.16.2` silently ignores `env` entries in CSV. See `ARCHITECTURE.md` §7.)

That's why the 181 MiB image has a working NVDEC + nvinfer + NvDCF pipeline despite containing no NVIDIA libs.

### WendyOS SDK story — what got de-risked first today

The Swift SDK sysroot ships from a Yocto build via `meta-wendyos-jetson`. Two relevant Yocto files:

- `meta-wendyos-jetson/recipes-devtools/deepstream/deepstream-7.1_%.bbappend` — extends the `deepstream-7.1-dev` package's `do_install` to copy `*.h`/`*.hpp` from `$D/opt/nvidia/deepstream/deepstream-7.1/sources/includes/` into `$D/usr/include/` (where clang looks via `--sysroot`), and puts linker stubs `libnvdsgst_meta.so` + `libnvds_meta.so` at `$D/usr/lib/`. `INSANE_SKIP:${PN}-dev += "dev-elf"` silences the Yocto QA check that would reject non-versioned `.so` names. `PRIVATE_LIBS:${PN}-dev` stops Yocto from registering duplicate shlib providers.
- `meta-wendyos-jetson/recipes-core/images/wendyos-sdk-image.bb` — SDK image includes only `deepstream-7.1-dev` (pulling the full `deepstream-7.1` runtime causes a `libv4l2.so.0` RPM transaction conflict with `tegra-libraries-multimedia-v4l`).

Regenerate SDK tarball:
```bash
bitbake wendyos-sdk-image
# produces wendyos-sdk-image-*.tar.bz2 under ~/workspace/build/tmp/deploy/images/...
# resync into ~/.swiftpm/swift-sdks/6.2.3-RELEASE_wendyos-device_aarch64.artifactbundle/
```

Full rationale in `ARCHITECTURE.md` §6 "Swift SDK and Cross-Compile".

### Why we don't use the "pure Package.swift" path

`swift-container-plugin` would build an OCI image directly from Swift metadata, no Dockerfile. **Not used here.** `wendy run` shells out to `docker buildx` against the project's Dockerfile. The Swift SPM build produces a binary that the Dockerfile `COPY`s. Verified empirically (the git log shows the swap: commit `4da868b` and preceding work use Dockerfile + wendy run; prior to Stage 1 we used the plugin).

### `ctr` vs `docker` namespace split

Two container runtimes on the Jetson:
- **containerd namespace `default`** — detector-swift, registry, deepstream-vision (old Python detector container). All managed with `ctr -n default tasks/containers ls/start/delete`. This is what `wendy run` deploys into.
- **docker** — only `llama-vlm` (the Qwen3-VL sidecar). Nothing detector-related.

On the dev host, `docker` is used only for `buildx_buildkit_wendy-builder0` (the BuildKit builder wendy launches). Don't conflate with the Jetson.

### Python detector (as it exists today)

Location: `/home/mihai/workspace/samples/deepstream-vision/detector/`
- `detector.py` — pyds-based DeepStream pipeline, Flask HTTP server, prometheus-client
- `Dockerfile`, `entrypoint.sh` — separate image, deployed via `wendy run` from that dir
- Ports: `METRICS_PORT=9092` (env var, separate from Swift)
- Uses the same `yolo26n_b2_fp16.engine` + `libnvdsparsebbox_yolo26.so` as Swift — model parity achieved in benchmark Run 2
- **Currently unstable** — dies silently within ~90 s of start during TRT warmup. Hypothesis: pyds segfault. Container task was running as `deepstream-vision` at one point (still listed in `ctr containers ls`) but `:9092` returns empty now. Run `ssh root@10.42.0.2 'ctr -n default tasks ls'` to see current state.

See §9 for reactivation plan.

### Historical / stub source files

- `Sources/Detector/RTSPFrameReader.swift` is now a ~38-line stub that only holds the `StreamConfig` and `StreamsConfig` types used to parse `streams.json`. The old FFmpeg-subprocess decode path that gave the file its name is dead code — removed when `GStreamerFrameReader` took over in the Stage 1 port. Kept only for the config types; not on any runtime path.

---

## 6. monitor.html flow

Served by `monitor_proxy.py` on the dev host, `:8001`. Same-origin is required for WebSocket without CORS headaches.

### Proxy routes
| Path prefix | Upstream | Notes |
|---|---|---|
| `/` | local file `monitor.html` | stdlib `http.server` |
| `/detector/*` | `http://10.42.0.2:9090/` | GET, POST; raw TCP splice on `Upgrade: websocket` |
| `/gpu/*` | `http://10.42.0.2:9091/` | tegrastats exporter |
| `/webrtc/*` | `http://10.42.0.2:8889/` | GET/POST/PATCH/DELETE — WHEP signalling |

UDP media for WebRTC does **not** go through the proxy. Browser's ICE picks `192.168.68.70:8189` (Jetson WiFi) and flows directly macbook↔Jetson over WiFi. See `monitor_proxy.py:18-26` docstring.

### WebRTC WHEP flow in the browser (monitor.html:730-830)
1. `RTCPeerConnection` creates SDP offer with a video recvonly transceiver
2. `fetch(whepUrl, { method: 'POST', body: offer.sdp, headers: Content-Type: application/sdp })`
3. mediamtx responds 201 with answer SDP (and `Location:` header for ICE trickle / teardown)
4. `setRemoteDescription(answer)` → ICE completes → `<video>` starts playing

### Detection overlay (monitor.html:840-950 roughly)
- WebSocket to `/detector/detections`
- Messages `{ frameNum, ptsNs, timestampNs, detections: [...] }` pushed into a ~60-entry ring buffer
- `video.requestVideoFrameCallback(cb)` fires once per presented video frame with `metadata.mediaTime` (seconds since stream start)
- First 8 frames: collect `ptsNs` from nearest ring-buffer entry, compute median offset → `ptsOffsetNs`
- Subsequent frames: `estPts = mediaTime * 1e9 + ptsOffsetNs`, pick ring-buffer entry minimizing `|ptsNs - estPts|`, draw bboxes
- Canvas buffer sized to `video.videoWidth × video.videoHeight` (source space); CSS stretches to display size; bbox coords from Swift are in source-frame pixels, drawn 1:1
- Firefox fallback: `requestAnimationFrame` + "latest detection" — ~50 ms off under motion

### Debug panel (top-right, `#debug-panel`, monitor.html:425-480)

| Field | Meaning |
|---|---|
| `WS` | WebSocket state: connected/disconnected/retrying |
| `messages` | Cumulative detection messages received |
| `last ptsNs` | Most recent detection's GST_BUFFER_PTS in ns |
| `ptsOffset calibrated` | `yes` after 8 samples, else `no (N/8 samples)` |
| `ptsOffsetNs` | Computed offset in ms (median over calibration window) |
| `mediaTime` | Current `video` element's mediaTime (playback clock, seconds) |
| `video intrinsic` | `videoWidth × videoHeight` (source resolution from WebRTC) |
| `video display` | `offsetWidth × offsetHeight` (CSS-rendered size) |
| `canvas buf` | Canvas backing buffer dimensions; should match intrinsic once RVFC fires |
| `RVFC active` | `yes` if `requestVideoFrameCallback` registered successfully |
| `draw calls` | Cumulative canvas draws (should tick at ~20 Hz) |
| `ring size` | Current detection-buffer length (caps at ~60) |

If `ptsOffset calibrated` stays `no`: no detections in the ring buffer when frames arrive — check `WS` state and `messages` counter. If `canvas buf` stays `300×150 (default)`: video hasn't fired `loadedmetadata` yet — WHEP probably never completed.

### Known browser caveats
- **Chrome**: RVFC native, sub-frame-accurate. Baseline.
- **Firefox**: no RVFC → RAF fallback → ~50 ms drift on motion. Document, don't fix.
- **Safari**: untested on macbook. WebRTC + WHEP should work; RVFC yes on 15.4+.

---

## 7. Current runtime state

As of writing (2026-04-15 latest sample), via `ctr -n default tasks ls` on `10.42.0.2`:

| Task | PID | Status | Notes |
|---|---|---|---|
| `detector-swift` | 9754 | RUNNING | 20 fps, 3 tracks, 217k frames, VmRSS 668 MB |
| `3df151d7...` (registry) | 1469 | RUNNING | containerd registry at `:5000` |
| `deepstream-vision` (old Python task) | — | container exists, not listed as task | Python detector; not currently active |

Containers: `3df…` (registry), `deepstream-vision`, `detector-swift`.

Services on Jetson (systemd):
- `mediamtx.service` — active, provides RTSP relay + WebRTC
- `tegrastats-exporter.service` — active, provides `:9091/metrics`

Dev host:
- `monitor_proxy.py` running as PID 2812593, `:8001` (`ps` confirms)
- `docker` buildx builder `wendy-builder0` (used only at build time)
- `mkdocs serve` on `:8000` (unrelated — Mihai's docs site)

VLM (`llama-vlm` docker container on Jetson): status not probed; assume stopped during benchmark runs.

Jetson disk: `/dev/nvme0n1p2` 22 GB total, **19 GB used, 1.8 GB free, 92% utilization**. Watch this on redeploys — see §10.

nvmap iovmm single-client total: **206 MB** for detector-swift (from `/sys/kernel/debug/nvmap/iovmm/clients`). Flat — no leak.

Verify for yourself:
```bash
ssh root@10.42.0.2 '
  ctr -n default tasks ls
  cat /sys/kernel/debug/nvmap/iovmm/clients
  systemctl --state=running | grep -E "mediamtx|tegra"
  df -h /
'
ps -ef | grep monitor_proxy | grep -v grep
```

---

## 8. Recent measurements (honest)

Authoritative source: `docs/benchmark-python-vs-swift.md` and the tables in `ARCHITECTURE.md` §5 "Stage 2 — Decoupled Video via WebRTC" and the head-to-head at the bottom.

### Swift solo (Stage 2, Run 1)
- FPS: 20.4 mean (14.8–25.7) — camera-limited
- VmRSS: ~561 MB (bench harness) / 595 MB (Stage 2 doc) / 668 MB (currently, after 3+ hr uptime) — the higher current number is consistent with long-running heap behavior, not a leak
- nvmap iovmm: **206 MB flat** (verified today via `/sys/kernel/debug/nvmap/iovmm/clients`)
- Preprocess latency histogram mean: ~26.5 ms
- Inference latency histogram mean: ~22.8 ms
- Postprocess latency histogram mean: ~14 ms (nvtracker NvDCF)
- GPU util: ~41%, power ~6.2 W

### Critical caveat (to preserve in all reporting)

**The 22.8 / 26.5 ms histograms straddle `nvstreammux` queueing — they are NOT pure compute.** Real `nvinfer` compute for YOLO26n FP16 on Orin Nano is **~8–12 ms** per Nsight Systems (not currently instrumented in the running pipeline; estimation from prior Nsight runs). Reporting the histogram as "inference latency" is directionally right but off by a factor of ~2–3x on compute.

`deepstream_total_latency_ms` is *not* end-to-end glass-to-glass — it's **inter-frame interval** via `gst_util_get_timestamp()` at the probe. A throughput-smoothness metric (spikes reveal jitter), not a latency metric. True glass-to-glass would need PTS instrumentation at `rtspsrc` input, which is not plumbed through.

### Refuted / corrected claims

- **"mediamtx relay adds ~20 MB of nvmap vs direct camera (181→201 MB)"** — **refuted** as a measurement artifact. Both direct and relay reads land in the same 200–225 MB band under Stage 2; steady-state is ~206 MB. All three affected docs (`ARCHITECTURE.md`, `docs/mjpeg-decoupling-synthesis.md`, `docs/benchmark-python-vs-swift.md`) updated 2026-04-15.

### Python (concurrent Run 2)
Python ran ~19.97 fps for ~90 s then exited. Swift was simultaneously starved to 0.24 fps. The benchmark report attributes this to VIC NVDEC hardware contention (one block, first-come-first-served). Rerun required after Python stability fix (§9).

---

## 9. Open issues + further work (prioritized)

### P0 — Python detector stability
- **Symptom:** silent death ~90 s into startup, before or during TRT warmup
- **Hypothesis:** pyds segfault on first inference (it's pyds-bound and we just brought in the YOLO26 parser for parity — could be ABI mismatch on the object-meta path)
- **Repro:** `WENDY_AGENT=10.42.0.2 wendy run -y --detach` from `detector/`, then `curl http://10.42.0.2:9092/metrics` — returns empty / connection refused within 90 s
- **Investigate:**
  1. `ssh root@10.42.0.2 'ulimit -c unlimited; ...'` (pre-start) then inspect `/var/lib/wendy-agent/storage/*/core.*`
  2. Add `wendy device logs` follow and grep for `Segmentation fault` / `pyds_add_meta` / `NvDsInfer`
  3. Compare `detector/nvinfer_config.txt` vs `detector-swift/nvinfer_config.txt` — custom parser path and symbol name must match
- **Blocking:** fair concurrent benchmark (P1 below)

### P1 — Fair parallel benchmark
Blocked on P0. Once Python is stable, rerun the 300 s window with both detectors reading from the relay, include 1-hour leak test. The benchmark Run 2 "0.24 fps starvation" result is real but uninformative as a software comparison — it's a hardware-level VIC conflict. Design options listed in `benchmark-python-vs-swift.md` "Follow-up Runs" §.

### P1 — Fix the refuted "relay tax" claim in docs *(RESOLVED 2026-04-15)*
- `ARCHITECTURE.md` (lines 26 / 285 / 490 / 501), `docs/mjpeg-decoupling-synthesis.md` STATUS line, and `docs/benchmark-python-vs-swift.md` Run 1 nvmap row all updated to ~206 MB steady-state with the ~23 MB drop attributed to MJPEG removal, not the relay path. A/B result: direct and relay both reach 206 MB. No residual 201 MB / "relay tax" framing in any doc.

### P2 — mediamtx + camera WiFi robustness
Camera RTT is 163 ms over WiFi; when the camera stutters, `rtspsrc` inside the detector can hit EOS. Detector pipeline then needs an explicit teardown + rebuild. Current behavior: Swift's GStreamer actor handles this at init time, but not mid-stream. Add reconnect logic to `GStreamerFrameReader` (watch bus for EOS / ERROR messages, tear down, wait N seconds, rebuild).

### P2 — `GST_DEBUG` plugin-load failure noise
`g_once_init_leave_pointer: undefined symbol` during pipeline setup (GLib ABI mismatch between Ubuntu 24.04 base and CDI-injected JetPack GLib). Adds ~200 ms to every pipeline restart. Cosmetic. Likely fixable by forcing plugin discovery to skip the failing gst-plugins-bad entries via `GST_PLUGIN_SYSTEM_PATH` restriction.

### P3 — `wendy run --json` CLI bug
`--json` flag is passed through to the container entrypoint instead of being consumed by the `wendy run` CLI itself. Separate bug to file against wendy. Workaround: don't use `--json` on `wendy run` today.

### P3 — Documentation cleanup *(partially resolved 2026-04-15)*
- *(done)* Refuted-measurement paragraphs corrected across `ARCHITECTURE.md`, `mjpeg-decoupling-synthesis.md`, and `benchmark-python-vs-swift.md`.
- *(done)* `OPERATIONS.md` rewritten to use `wendy run` workflow and the correct `6.2.3-RELEASE_wendyos-device_aarch64` SDK name; added WebRTC/WHEP/mediamtx section; port table corrected.
- *(done)* `PORT_PLAN.md` top banner marks it COMPLETE as of commit `415cc8c` and points at this HANDOFF for current state. Body preserved as historical record; some body sections still reference the pre-port `build-container-image` workflow and software-decode baseline — acceptable because the top banner frames the doc as historical.

### P3 — Commit the uncommitted tree
See §12. Suggested grouping:
- **Commit A: Stage 2 pipeline changes** — detector-swift/Sources/** changes, Package.swift, Dockerfile, ARCHITECTURE.md, nvinfer_config.txt changes, deleted MJPEGSidecar.swift. (Excludes Python detector churn.)
- **Commit B: Python detector coexistence + relay config** — detector/** changes, monitor_proxy.py webrtc support, monitor.html WebRTC client, scripts/benchmark.sh, docs/rtsp-relay.md, docs/mjpeg-decoupling-*.md.
- **Commit C: Benchmark run data + blog revisions** — docs/benchmark-*.md, docs/benchmark-data/*.csv, docs/blog-swift-detector-port.md edits.
- New Dockerfile at repo root (`Dockerfile`) — check whether intentional or stray before committing.

### 9.5 — Diagnosing FPS=0 or a stalled detector

If `curl http://10.42.0.2:9090/metrics | grep deepstream_fps` shows `0` (or the `/metrics` socket refuses) this tree is the fastest triage. Work top-to-bottom; stop at the first failing check.

1. **Is the task actually running?**
   ```bash
   ssh root@10.42.0.2 'ctr -n default tasks ls | grep detector'
   ```
   If status is `STOPPED` (or no row at all), jump to step 7.

2. **What does the detector log say?**
   ```bash
   ssh root@10.42.0.2 'tail -30 /var/lib/wendy-agent/storage/detector-cache/detector.log'
   ```
   Look for: `EOS on pipeline`, `state-change failed`, `CUDA_ERROR_*`, `NvBufSurface`, `Segmentation fault`. EOS generally means the upstream RTSP source disappeared → continue to step 3–5.

3. **Is the camera reachable from the Jetson?**
   ```bash
   ssh root@10.42.0.2 'ping -c 1 -W 2 192.168.68.69'
   ```
   Loss means WiFi dropped → step 4.

4. **WiFi up on the Jetson?**
   ```bash
   ssh root@10.42.0.2 'nmcli con up "badgers den"'
   ```
   (See §10 #12.) Retry step 3 after it comes up.

5. **Is mediamtx publishing the relay?**
   ```bash
   ssh root@10.42.0.2 'systemctl is-active mediamtx'
   curl -s http://10.42.0.2:9997/v3/paths/list | jq '.items[] | {name, ready, readers}'
   ```
   If `relayed` is missing or `ready: false`, restart it: `ssh root@10.42.0.2 'systemctl restart mediamtx'`.

6. **Is the relay accepting RTSP?**
   ```bash
   timeout 3 curl -sfI -X OPTIONS rtsp://10.42.0.2:8554/relayed || echo 'relay not accepting'
   ```

7. **Force a clean detector restart** (from the dev host):
   ```bash
   ssh root@10.42.0.2 'ctr -n default tasks rm detector-swift 2>/dev/null
                       ctr -n default containers rm detector-swift 2>/dev/null'
   cd /home/mihai/workspace/samples/deepstream-vision/detector-swift
   WENDY_AGENT=10.42.0.2 wendy run -y --detach
   ssh root@10.42.0.2 'CONTAINER=detector-swift /usr/local/bin/detector-cap'
   ```
   (No rebuild is needed if you haven't touched the Swift sources.)

8. **Driver-level failures?**
   ```bash
   ssh root@10.42.0.2 'dmesg -T | grep -iE "nvmap|oom|nvgpu|v4l" | tail -20'
   ```
   NVMM allocation failures, OOM kills, or `nvgpu` resets here mean the device needs a power-cycle — not just a container restart.

If all eight checks pass but FPS is still 0, the pad probe may not be firing. Check `NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1` is set in the container env (§10 #8) and that the `/detections` WebSocket is emitting (`websocat -n1 ws://10.42.0.2:9090/detections | head`). If detections flow but `deepstream_fps` stays 0, suspect Metrics-registry corruption and restart via step 7.

---

## 10. Gotchas / session-specific knowledge a new agent MUST know

1. **Camera is single-session.** `rtsp://192.168.68.69:554/stream1` accepts exactly one client. mediamtx on the Jetson is that client. Anyone else — including `ffprobe`, a second detector, a manual test — must pull from `rtsp://10.42.0.2:8554/relayed`, not the camera.

2. **`wendy run` ALWAYS uses a local Dockerfile + docker buildx.** There is no swift-native container build in this repo today. (`OPERATIONS.md` documented this correctly as of 2026-04-15; the old `swift package build-container-image` recipe is gone.)

3. **BuildKit caches aggressively.** If after `wendy run` the `md5sum` of your built `Detector` binary on the dev host differs from the binary inside the deployed container, restart the builder: `docker restart buildx_buildkit_wendy-builder0`.

4. **Jetson root disk at 92%.** `df -h /` shows 1.8 GB free. During redeploys, `ctr -n default images prune` / `images rm` old detector-swift image tags before pushing a new one, or the registry push will silently ENOSPC.

5. **Never use `--restart-unless-stopped` on an unstable build.** A tight restart loop (crash → containerd restart in <1 s) hoses NVDEC driver state, pegs CPU, and times out SSH. Has forced two power-cycles in prior sessions. Only use `--restart-unless-stopped` after the build has been stable for 5+ minutes.

6. **Apply `/usr/local/bin/detector-cap` after every `wendy run`.** Containerd sets `oom_score_adj=-998` by default, which means the kernel OOM-killer takes out networking before it takes out the detector. `detector-cap` applies a 5 GiB cgroup cap + resets the adj.

7. **`deepstream_total_latency_ms` ≠ glass-to-glass.** It's inter-frame interval via `gst_util_get_timestamp`. Don't report it as a latency SLO.

8. **`NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1` env gates the per-component (`preprocess`, `inference`, `postprocess`) histograms.** Don't unset this — the decoupled histograms silently stop updating and the `/metrics` output looks normal.

9. **`cairo-stage/` is GONE post-Stage-2.** Don't re-add. `nvdsosd` was the only thing that linked cairo; it's out of the pipeline.

10. **`MJPEG_DISABLED` env var is gone post-Stage-2.** The pipeline is pure detection; there is no MJPEG branch to disable.

11. **WebSocket detection fan-out uses a per-client continuation map in `DetectionBroadcaster`, NOT the shared `DetectionStream`.** The shared stream uses `.bufferingNewest(4)` and would silently drop frames across multiple WS clients. Each WS client gets its own `AsyncStream<String>` continuation, fed by the broadcaster's `distribute(_)` call. See `Sources/Detector/DetectionBroadcaster.swift`.

12. **WiFi on Jetson doesn't auto-recover after reboot.** After `sudo reboot`, always `ssh root@10.42.0.2 'nmcli con up "badgers den"'` or the camera relay will sit dark.

13. **VLM (`llama-vlm` docker container) holds ~2 GB unified memory.** Stop it for memory-constrained tests: `ssh root@10.42.0.2 'docker stop llama-vlm'`. Unified memory on Orin Nano means "unified" with detector's GPU allocations — VLM running reduces nvinfer headroom materially.

14. **`gst-plugin-scanner` ABI mismatch.** Set `GST_PLUGIN_SCANNER=""` at detector start; external scanner binary from Ubuntu 24.04 is ABI-incompatible with CDI-injected L4T GStreamer 1.22. In-process scanning works.

15. **`cluster-mode=4` in `nvinfer_config.txt` is mandatory for YOLO26.** YOLO26's 1:1 head already emits final detections. Any other cluster-mode corrupts the output.

---

## 10b. Debug gotchas + process lessons (from this session)

Meta-knowledge from how we actually got here — different from the runtime gotchas in §10.

### Build + deploy traps

- **BuildKit caches binaries by path, not content.** `md5sum` of local vs in-container binary is the only reliable signal that a deploy took. When they differ: `docker restart buildx_buildkit_wendy-builder0`. Hit this 4× in one session.
- **qemu-aarch64 `apt-get install` is glacial** — `libcairo2` + deps took 12+ min and never finished. When you need aarch64 packages, check `~/jetson/Linux_for_Tegra/rootfs/` FIRST. Staging raw `.so` files (see `scripts/stage-cairo.sh`) is seconds vs minutes.
- **Stacked `wendy run` processes don't abort each other** — they queue on buildkit and compound. Always `pkill -9 -f "wendy run"` before retrying.
- **Chained shell commands can half-fail silently.** `ssh ...; wendy run ...` patterns ate failures mid-chain. Run as separate commands; verify each exit code.
- **Dockerfile overrides wendy-native Swift build** — `wendy run` always uses the local Dockerfile + docker buildx if one exists. There is no swift-container-plugin path even when `wendy.json` declares `language: swift`.

### Measurement traps

- **Units on `NvDsMetaCompLatency` timestamps took three deploys to land.** `×1000` → `÷1000` → no scaling. DS 7.1 stores them as milliseconds-since-epoch in `gdouble`; diff is already in ms. Sanity-check magnitude against a known-real number before trusting a new metric.
- **Prometheus text format with label sets breaks naive awk parsers.** `awk -F'[ }]'` silently returns empty fields on `metric{label="x"} value`. **Dump raw `.prom` files per sample, parse offline.** That one fix would have saved the first two benchmark runs.
- **Benchmarks started before both detectors were producing frames wasted ~30 min.** Gate measurement windows on `deepstream_fps > 15` scraped from each detector's `/metrics`, not on "deploy returned success."
- **`now() - GST_BUFFER_PTS` gives bogus ~7000 ms "latency"** because rtspsrc PTS derives from RTP clock, not pipeline base_time. "Total latency" from GStreamer timestamps is NOT wall-clock; requires rtspsrc-arrival instrumentation for a real end-to-end measurement. Our `deepstream_total_latency_ms` is actually inter-frame interval via `gst_util_get_timestamp` — throughput-smoothness, not latency.
- **Unit-mismatched histograms populate cleanly with wrong buckets.** The metric "works" and the browser shows numbers; they're just 1000× wrong. Only magnitude sanity-checks catch this.

### Agent orchestration traps

- **Two agents editing the same file race.** Stage 1 impl ran concurrent with an emergency benchmark agent; the benchmark agent set `ENV MJPEG_DISABLED=1` to work around a pipeline stall, defeating Stage 1's valve logic. **Rule: no concurrent agents on overlapping files unless explicitly partitioned.**
- **Agents don't self-limit on hopeless build paths.** One ran 2.5 hours on a stuck qemu-apt before its timeout hit. Set generous timeouts AND watch wall time.
- **Agents that need to deploy MUST have explicit "no deploy" instructions** when others are mid-run. Otherwise they assume exclusive control.
- **Codex rounds catch over-claiming almost every time.** Every doc we shipped had hand-waved or strawmanned something. **Budget at least one codex pass per doc you publish.**
- **Stale `SendMessage`-style resumption isn't available in this harness.** Once an agent exits, it's gone; the "continue this agent via SendMessage" hint in the completion message is misleading. Design prompts to be self-contained.

### Runtime gotchas not in §10

- **GStreamer plugin registry caches failures across restarts.** If a plugin failed to load once (e.g., `videodrop` ABI issue), the blacklist sticks. Entrypoints should `rm -rf ~/.cache/gstreamer-1.0/` — the Python detector does this; the Swift detector inherits it from CDI.
- **Zombie background processes compound over long sessions.** Old `docker pull` hangs from hours earlier, stuck `wendy run`s, abandoned `buildkit-runc`. Before assuming a clean slate: `ps -ef | grep -E "wendy|docker buildx|buildkit-runc"`.
- **Disk pressure drifts.** Started at 100% full on Jetson root; cleanup brought to 92%; ongoing builds threatened the margin. Monitor `df -h /` on the Jetson periodically during long sessions.
- **OOMs don't always land in `dmesg` on this kernel.** When a container exits silently: `ulimit -c unlimited` + `/proc/sys/kernel/core_pattern` + `wendy device logs --follow` are all needed to capture the cause. Python detector died four times without a single usable exit trace.
- **`monitor_proxy.py` grew organically** during this session: added `/webrtc/` HTTP route + WebSocket TCP-splice tunnel + existing `/detector` `/gpu` routes. It now has four route families. **The proxy is its own API contract going forward.** Document its routes when touching it.

### SDK / CDI gotchas

- **WendyOS SDK ships top-level DS headers but NOT all transitive includes.** `nvdsmeta.h` was present; `nvll_osd_struct.h` (which it includes) was not. Always verify transitive `#include` chains when a new header target fails. Yocto bbappend now handles it.
- **`tegra-libraries-multimedia-v4l` is removed from the SDK** to break an RPM file conflict with upstream `libv4l` over `/usr/lib/libv4l2.so.0`. Safe for the SDK (cross-compile sysroot) but the `RDEPENDS += "libv4l"` in the meta-tegra bbappend is a real packaging bug worth filing upstream.
- **DS runtime libs come from CDI bind-mount, not the container image.** Container images are minimal ubuntu:24.04 + binary + resources. Never bake DS into the image — it would collide with the CDI-injected version at runtime.
- **`wendy run --json` CLI bug** leaks `--json` into the container entrypoint. Swift's ArgumentParser rejects it, container exits immediately. Use `wendy run` (no `--json`) when the container binary parses its own args.

### Hypothesis hygiene

- **Our "8 ms reclaim" claim was speculative; measured was 5.3 ms.** Codex caught hand-waving twice. **Tag hypotheses as hypotheses in docs until measured.**
- **Our "181 → 201 MB relay tax" claim got cited as fact across three docs before A/B refuted it** (the 181 figure was a measurement taken during pipeline warmup, not steady state). Mental model: if a number surprises you and you only saw it once, it's a reading, not a fact. **Re-measure before citing.**
- **"videodrop is broken" was a mis-diagnosis** — the 4-frames-then-stall stall was actually the camera's UDP RTP dropping, fixed by `rtspTransport: tcp` in mediamtx. The videodrop stub ABI issue is real but a separate problem. Be careful about attributing a symptom to the first suspect plugin in the log.

### Camera + relay specific

- **The camera at `192.168.68.69:554` is strictly single-session.** This causes spooky action at a distance: running the detector direct while mediamtx is also running = detector gets a silent CLOSE_WAIT and never sees frames. Always go through the relay.
- **mediamtx → camera leg uses UDP by default**, which timed out repeatedly. Fixed by setting `rtspTransport: tcp` in `/etc/mediamtx.yml`. Camera's RTCP sender reports were also dropping; TCP interleaved is strictly better on this network.
- **Camera WiFi RTT is ~163 ms** — borderline. Expect occasional EOS events; the detector currently can't auto-reconnect from EOS. Worth adding rtspsrc `retry` / `timeout` properties.

---

## 11. Pointers to other docs

All paths absolute.

| Doc | Purpose |
|---|---|
| `/home/mihai/workspace/samples/deepstream-vision/detector-swift/PORT_PLAN.md` | The original, standalone port plan for Python → Swift DeepStream. Historical but still the best "why does this design exist" + NVMM-leak root-cause doc. Complete as of commit `415cc8c`. |
| `/home/mihai/workspace/samples/deepstream-vision/detector-swift/ARCHITECTURE.md` | Architecture of record. Pipeline diagram, metrics table, Stage 2 design, cross-compile & CDI mechanics, head-to-head Python vs Swift. Refuted "201 MB relay" framing fixed 2026-04-15. |
| `/home/mihai/workspace/samples/deepstream-vision/OPERATIONS.md` | Device access recipes, USB-C setup, service table, per-port list, `wendy run` build/deploy commands, WebRTC/WHEP/mediamtx section, common gotchas. Rewritten 2026-04-15 to match Stage 2 reality. |
| `/home/mihai/workspace/samples/deepstream-vision/docs/mjpeg-decoupling-synthesis.md` | Team review of MJPEG decoupling options → Stage 1 / Stage 2 decision. Both stages shipped; STATUS header updated 2026-04-15 with a HISTORICAL banner noting the body is future-tense as-of design time — see this HANDOFF for current state. |
| `/home/mihai/workspace/samples/deepstream-vision/docs/mjpeg-decoupling-design.md` | Earlier, more exhaustive options analysis feeding the synthesis. |
| `/home/mihai/workspace/samples/deepstream-vision/docs/mjpeg-contention-measurement.md` | Measurement notes that kicked off the decoupling work. |
| `/home/mihai/workspace/samples/deepstream-vision/docs/benchmark-python-vs-swift.md` | Benchmark report: Run 1 (Swift solo), Run 2 (concurrent, VIC-starved). "Caveats" section is essential before citing any number. |
| `/home/mihai/workspace/samples/deepstream-vision/docs/benchmark-data/` | Raw CSVs from `scripts/benchmark.sh`. |
| `/home/mihai/workspace/samples/deepstream-vision/docs/blog-swift-detector-port.md` | Blog post in Mihai's voice. Don't rewrite; pointer-only for context. |
| `/home/mihai/workspace/samples/deepstream-vision/docs/rtsp-relay.md` | mediamtx RTSP-relay config notes. |
| `/home/mihai/workspace/samples/deepstream-vision/docs/billboard-scanner-design.md` | Downstream product design (v4.0) that uses this detector stack. Not required reading for detector work. |
| `/home/mihai/workspace/samples/deepstream-vision/docs/dockerfile-nvparser-stage-example.txt` | Dockerfile fragment example for the custom parser build stage. Useful when regenerating `libnvdsparsebbox_yolo26.so`. |

Source of truth for the pipeline itself:
- `/home/mihai/workspace/samples/deepstream-vision/detector-swift/Sources/Detector/GStreamerFrameReader.swift`
- `/home/mihai/workspace/samples/deepstream-vision/detector-swift/Sources/Detector/Detector.swift`
- `/home/mihai/workspace/samples/deepstream-vision/detector-swift/Sources/Detector/DetectionBroadcaster.swift`
- `/home/mihai/workspace/samples/deepstream-vision/detector-swift/Sources/Detector/HTTPServer.swift`
- `/home/mihai/workspace/samples/deepstream-vision/detector-swift/Sources/CGStreamer/nvds_shim.c`
- `/home/mihai/workspace/samples/deepstream-vision/detector-swift/Sources/CNvdsParser/nvdsparsebbox_yolo26.cpp`
- `/home/mihai/workspace/samples/deepstream-vision/detector-swift/nvinfer_config.txt`
- `/home/mihai/workspace/samples/deepstream-vision/detector-swift/tracker_config.yml`
- `/home/mihai/workspace/samples/deepstream-vision/monitor.html`
- `/home/mihai/workspace/samples/deepstream-vision/monitor_proxy.py`

---

## 12. Git state snapshot

Branch: `swift-detector` (10 commits ahead of `origin/swift-detector`, not pushed).

```
Changes not staged for commit:
    modified:   detector-swift/ARCHITECTURE.md
    modified:   detector-swift/Dockerfile
    modified:   detector-swift/Package.swift
    modified:   detector-swift/Sources/CGStreamer/nvds_shim.c
    modified:   detector-swift/Sources/CGStreamer/nvds_shim.h
    modified:   detector-swift/Sources/CGStreamer/shim.h
    modified:   detector-swift/Sources/Detector/Detector.swift
    modified:   detector-swift/Sources/Detector/GStreamerFrameReader.swift
    modified:   detector-swift/Sources/Detector/HTTPServer.swift
    deleted:    detector-swift/Sources/Detector/MJPEGSidecar.swift
    modified:   detector-swift/Sources/Detector/Metrics.swift
    modified:   detector-swift/streams.json
    modified:   detector/Dockerfile
    modified:   detector/entrypoint.sh
    modified:   detector/nvinfer_config.txt
    modified:   detector/streams.json
    modified:   docs/benchmark-python-vs-swift.md
    modified:   docs/blog-swift-detector-port.md
    modified:   monitor.html
    modified:   monitor_proxy.py
    modified:   scripts/benchmark.sh

Untracked files:
    Dockerfile                                    (stray? verify before adding)
    detector-swift/.dockerignore
    detector-swift/Sources/Detector/DetectionBroadcaster.swift
    detector-swift/gst-videodrop-stub.c           (Stage 1 artifact — probably delete)
    docs/benchmark-data/concurrent-python.csv
    docs/benchmark-data/concurrent-swift.csv
    docs/benchmark-data/parallel.csv
    docs/mjpeg-contention-measurement.md
    docs/mjpeg-decoupling-design.md
    docs/mjpeg-decoupling-synthesis.md
    docs/rtsp-relay.md
```

Suggested commit grouping (apply one at a time; do NOT commit without explicit user request per agent policy):

1. **`Stage 2: remove MJPEG branch; adopt mediamtx WebRTC for video`** — all of `detector-swift/**` (sources, Package.swift, Dockerfile, ARCHITECTURE.md, streams.json, new DetectionBroadcaster.swift, `.dockerignore`, deleted MJPEGSidecar.swift). Add `detector-swift/gst-videodrop-stub.c` to the commit only to immediately delete it in the same commit (Stage 1 leftover). Body: reference `docs/mjpeg-decoupling-synthesis.md`.

2. **`Python detector + benchmark coexistence via mediamtx relay`** — `detector/**`, `monitor_proxy.py` (WebRTC), `monitor.html` (WebRTC client + debug panel), `scripts/benchmark.sh`, `docs/mjpeg-contention-measurement.md`, `docs/mjpeg-decoupling-design.md`, `docs/mjpeg-decoupling-synthesis.md`, `docs/rtsp-relay.md`.

3. **`Benchmark run data + blog revisions`** — `docs/benchmark-python-vs-swift.md`, `docs/benchmark-data/*.csv`, `docs/blog-swift-detector-port.md`.

4. **Verify the root-level `Dockerfile`** (185 bytes, dated Apr 15). If it's leftover from a debug session, `git clean`. If it's intentional, add in a small standalone commit with the rationale.

Run `git status` at the start of any session to confirm this list hasn't drifted.

---

## 13. Fresh-session first-moves checklist

A new Claude session should, in order:

1. **Read this file (you are here).**
2. `ssh root@10.42.0.2 'ctr -n default tasks ls'` to confirm detector is running.
3. `curl -s http://10.42.0.2:9090/metrics | grep deepstream_fps` to confirm detection is flowing.
4. `ps -ef | grep monitor_proxy | grep -v grep` on dev host to confirm proxy is up.
5. `cd /home/mihai/workspace/samples/deepstream-vision && git status` to see what's uncommitted.
6. Skim `detector-swift/ARCHITECTURE.md` §5 "Stage 2" and §"Head-to-Head" for current-shape context.
7. Skim `docs/benchmark-python-vs-swift.md` "Caveats" before making any claim about performance.
8. Open `docs/mjpeg-decoupling-synthesis.md` STATUS line to confirm you're in the Stage 2 world.
9. Then — and only then — decide which P0/P1 in §9 to pick up.

If metrics are gone / task is not running — something broke between sessions. Start with `ssh root@10.42.0.2 'dmesg -T | grep -iE "oom|killed" | tail -10'` and detector log at `/var/lib/wendy-agent/storage/detector-cache/detector.log`.
