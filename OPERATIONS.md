# Operations: topology, monitor proxy, device access

This file captures the deployment topology and the operational glue that
isn't in the code. Pair with `detector-swift/PORT_PLAN.md` for the detector
port work itself.

## Topology

```
┌─────────────┐                    ┌────────────────────┐                   ┌──────────────────┐
│     Mac     │ ─── WiFi router ─── │   edge-builder-1   │ ─── USB-C 1:1 ──── │  Jetson Orin     │
│             │ ─── ethernet ────── │  (Ubuntu host)     │                   │  Nano 8 GB       │
│             │    192.168.100.x   │                    │                   │  (WendyOS)       │
└─────────────┘                    │ wlp3s0  192.168.68.75 │                   │ wlP1p1s0 .68.70  │
                                   │ enp2s0  192.168.100.1 │                   │ usb0     10.42.0.2│
                                   │ enx…    10.42.0.1/24  │                   │                   │
                                   │ tailscale 100.68.…    │                   │                   │
                                   └────────────────────┘                   └──────────────────┘
                                                                                   ▲
                                                                                   │ WiFi
                                                                                   │
                                                                          ┌────────┴────────┐
                                                                          │  RTSP camera    │
                                                                          │  192.168.68.69  │
                                                                          └─────────────────┘
```

**Key links**:

| link | purpose | reliability |
|---|---|---|
| Mac ↔ edge-builder-1 | load the monitor UI | either WiFi (`.68.x`) or direct ethernet (`192.168.100.1`) — both work |
| edge-builder-1 ↔ Jetson over **USB-C** (`10.42.0.1` ↔ `10.42.0.2`) | SSH, registry push, all proxy traffic | fast (~1–3 ms), rock-solid |
| edge-builder-1 ↔ Jetson over **WiFi** (`.68.75` ↔ `.68.70`) | fallback only | flaky — 45–300 ms RTT, packet loss |
| Jetson ↔ RTSP camera over WiFi | the actual video source | same flaky WiFi segment |

**Rule of thumb**: anything going to/from the Jetson from the build host
should prefer `10.42.0.2`. The camera is the only thing on `.68.x` the
Jetson can't avoid.

## USB-C setup (one-time)

Both sides ship with NetworkManager configured as `ipv4.method: shared` on
the USB gadget interface, which means both want to *be* the DHCP server —
neither acts as a client and the subnet stays silent. Fix (one-time, on
the Jetson):

```bash
ssh root@192.168.68.70 '
  nmcli con modify usb-gadget ipv4.method manual ipv4.addresses 10.42.0.2/24
  nmcli con down usb-gadget 2>/dev/null
  nmcli con up usb-gadget
'
```

After this, `edge-builder-1` at `10.42.0.1/24` and the Jetson at
`10.42.0.2/24` reach each other directly over the USB-C cable. This
survives reboots on both sides.

## Monitor UI + proxy

`monitor.html` is a static page with vanilla JS that reads Prometheus
metrics, subscribes to the `/detections` WebSocket, and plays video
via a WebRTC `<video>` element (Stage 2). It's served by
`monitor_proxy.py` — a stdlib Python reverse proxy running on
`edge-builder-1`.

### What the proxy does

```
GET /                       → serves monitor.html from the repo root
GET /detector/<path>        → proxies to http://10.42.0.2:9090/<path>
GET /gpu/<path>             → proxies to http://10.42.0.2:9091/<path>
POST /webrtc/<path>/whep    → proxies to http://10.42.0.2:8889/<path>/whep (WHEP signalling only)
GET /detector/detections    → WebSocket splice (Upgrade handled via raw TCP)
```

Streaming responses (`multipart/x-mixed-replace` — legacy, no longer
used by the Swift detector) are forwarded chunked, never buffered.
WebSocket upgrades on `/detector/detections` are spliced at the TCP
layer. WebRTC media (UDP on `:8189`) does **not** go through the
proxy — ICE advertises the Jetson's addresses directly and the
browser's WebRTC stack connects peer-to-peer over WiFi. Only WHEP
signalling is proxied. Same-origin under port 8001, so no CORS.

### Starting / maintaining the proxy

```bash
cd ~/workspace/samples/deepstream-vision
python3 monitor_proxy.py
# listens on http://0.0.0.0:8001/
```

It's a foreground process; run it under a terminal multiplexer or systemd
if you want it surviving reboots.

Environment overrides (defaults shown):
- `JETSON_HOST=10.42.0.2`
- `PROXY_PORT=8001`

### URLs

| from | URL | notes |
|---|---|---|
| Mac on WiFi | `http://edge-builder-1.local:8001/` | mDNS resolves via Bonjour |
| Mac on WiFi (bare IP) | `http://192.168.68.75:8001/` | same thing, if mDNS flaky |
| Mac over direct ethernet | `http://192.168.100.1:8001/` | dedicated link to edge-builder-1 |
| Localhost on edge-builder-1 | `http://localhost:8001/` | dev testing |

### monitor.html behavior

The JS detects `window.location.protocol !== 'file:'` and uses same-origin
paths `/detector/...`, `/gpu/...`, and `/webrtc/...`. The device-input
box is preserved for backwards compat (loading the file directly via
`file://`), but is ignored when served through the proxy.

Video is a `<video>` element driven by a WHEP POST (`POST
/webrtc/relayed/whep`). Detections arrive on a WebSocket
(`/detector/detections`) and draw on a `<canvas>` overlay aligned via
`requestVideoFrameCallback` (Chrome) or RAF (Firefox) using the
detection message's `ptsNs`. If the video element errors, the JS
retries the WHEP handshake.

## Video delivery (Stage 2, WebRTC / WHEP)

Stage 2 (2026-04-15) replaced the detector's MJPEG branch with a
mediamtx-backed WebRTC feed. The detector no longer encodes JPEGs.

**Components on the Jetson:**

- `mediamtx` (systemd unit) — ingests `rtsp://192.168.68.69:554/stream1`
  once and publishes it as:
  - `rtsp://10.42.0.2:8554/relayed` — for any RTSP consumer (Swift
    detector, Python detector, ffprobe, etc.). Never hit the camera
    directly; it is single-session.
  - `http://10.42.0.2:8889/relayed/whep` — WHEP endpoint for browser
    `<video>` clients. RTP repacketization only, no transcode.
  - `http://10.42.0.2:9997/v3/paths/list` — mediamtx HTTP API, useful
    for "is the relay live?" checks.

**Browser flow:**

1. `<video>` element POSTs an SDP offer to `/webrtc/relayed/whep` (proxied
   to `:8889`).
2. mediamtx replies `201 Created` with an SDP answer.
3. ICE candidates advertise `10.42.0.2:8189` and `192.168.68.70:8189`.
4. Browser picks the WiFi-reachable candidate (`.68.70`) and UDP media
   flows peer-to-peer Jetson → browser over WiFi.

**Port `:8189/udp`** is the media plane. The dev-host proxy is not in the
path — the browser talks to the Jetson directly once WHEP signalling
has established the session.

**Verify mediamtx is publishing:**

```bash
systemctl is-active mediamtx                                     # on the Jetson
curl -s http://10.42.0.2:9997/v3/paths/list | jq '.items[].name' # should list "relayed"
timeout 3 curl -sfI -X OPTIONS rtsp://10.42.0.2:8554/relayed     # RTSP reachable
```

**Browser compatibility:** Chrome / Safari use `requestVideoFrameCallback`
for frame-accurate overlay sync. Firefox falls back to `requestAnimationFrame`
(degraded sync — boxes may trail by ~1 frame).

## Jetson services at a glance

| port | service | under | notes |
|---|---|---|---|
| `22` | sshd | systemd | primary access |
| `5000` | container registry | containerd task | containerd registry target for `wendy run`'s buildx → registry push |
| `8090` | VLM sidecar (llama-server, Qwen3-VL) | Docker (`llama-vlm`) | optional; hold ~2 GB unified memory when running |
| `8554` | RTSP relay (mediamtx) | systemd `mediamtx.service` | `/relayed` — both detectors consume this |
| `8889` | WebRTC / WHEP (mediamtx) | systemd `mediamtx.service` | `/relayed/whep` — browser video |
| `8189/udp` | WebRTC media (mediamtx) | systemd `mediamtx.service` | ICE-advertised; browser ↔ Jetson direct |
| `9090` | Swift detector HTTP | containerd task `detector-swift` | `/metrics`, `/detections` (WebSocket), `/healthz`, `/api/vlm_*` |
| `9091` | tegrastats exporter | systemd `tegrastats-exporter.service` | Prometheus CPU/GPU/temp gauges |
| `9092` | Python detector HTTP (if deployed) | containerd task `deepstream-vision` | `/metrics`; currently unstable — see `docs/HANDOFF.md` §9 |
| `9997` | mediamtx HTTP API | systemd `mediamtx.service` | `/v3/paths/list` etc. |

## Device-access reference

### SSH

Always prefer the USB-C path:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    root@10.42.0.2
```

Falls back to `root@192.168.68.70` if USB-C is disconnected, but WiFi can
drop at any time on this device.

### Recovering WiFi after a Jetson reboot

The Jetson's WiFi NM connection tends **not** to come up automatically
after reboot (WendyOS quirk). RTSP from the camera needs it. Run:

```bash
ssh root@10.42.0.2 'nmcli con up "badgers den"'
```

(`badgers den` is the SSID for this network — check `nmcli con show` if
it ever changes.)

### Restarting the container registry

If a `wendy run` push fails with a connection-refused on port 5000, the
registry container's task has stopped:

```bash
ssh root@10.42.0.2 '
  REG=$(ctr -n default containers ls | grep -i registry | head -1 | awk "{print \$1}")
  ctr -n default tasks start -d "$REG"
'
```

### Stopping the VLM sidecar

For memory-constrained tests (it holds ~2 GB unified memory, out of the
8 GB total system RAM):

```bash
ssh root@10.42.0.2 'docker stop llama-vlm && docker update --restart no llama-vlm'
```

The `docker update --restart no` prevents the VLM from auto-starting after
a reboot. To re-enable:

```bash
ssh root@10.42.0.2 'docker update --restart unless-stopped llama-vlm && docker start llama-vlm'
```

### Stopping / starting the detector

The current build/deploy workflow uses `wendy run` against a project-local
`detector-swift/Dockerfile`. There is no `swift package build-container-image`
path anymore — that CLI is gone as of the Stage 1 / Stage 2 work. Engine
files, the custom parser, and config files are `COPY`'d into the image at
Dockerfile build time, not mounted via `--resources` at runtime.

```bash
# stop
ssh root@10.42.0.2 'ctr -n default tasks rm detector-swift 2>/dev/null
                    ctr -n default containers rm detector-swift 2>/dev/null'

# build (cross-compile x86 → aarch64; note the `-device` in the SDK name)
cd /home/mihai/workspace/samples/deepstream-vision/detector-swift
PATH="$HOME/.local/share/swiftly/bin:$PATH" swift build \
  --swift-sdk 6.2.3-RELEASE_wendyos-device_aarch64 --product Detector -c release

# deploy (wendy run handles buildx + push + ctr run)
WENDY_AGENT=10.42.0.2 wendy run -y --detach   # omit --restart-unless-stopped on untrusted builds

# apply cgroup cap (MANDATORY — oom_score_adj defaults to -998 otherwise,
# which means OOM kills networking before the detector)
ssh root@10.42.0.2 'CONTAINER=detector-swift /usr/local/bin/detector-cap'
```

See `docs/HANDOFF.md` §3 for the authoritative copy and §10 for the list
of deploy gotchas (BuildKit cache, disk pressure, `--restart-unless-stopped`
footgun).

## Quick-check recipes

Detector live metrics:
```bash
curl -s http://10.42.0.2:9090/metrics | grep '^deepstream_' | head
curl -s http://edge-builder-1.local:8001/detector/metrics | grep '^deepstream_' | head  # through the proxy
```

Detection WebSocket rate from the shell (counts frames for 5 s):
```bash
timeout 5 websocat -n1 ws://10.42.0.2:9090/detections 2>/dev/null | wc -l
```

GPU stats:
```bash
curl -s http://10.42.0.2:9091/metrics | grep '^jetson_' | head
```

mediamtx paths (confirm relay is publishing):
```bash
curl -s http://10.42.0.2:9997/v3/paths/list | jq '.items[] | {name, ready, readers}'
```

Detector logs (full history across restarts):
```bash
ssh root@10.42.0.2 'tail -f /var/lib/wendy-agent/storage/detector-cache/detector.log'
```

Kernel OOM events (after any mysterious detector death):
```bash
ssh root@10.42.0.2 'dmesg -T | grep -iE "oom|killed" | tail -10'
```

Per-process RSS (single sample):
```bash
ssh root@10.42.0.2 '
  PID=$(cat /sys/fs/cgroup/system.slice/detector-swift/cgroup.procs | head -1)
  awk "/VmRSS/{print \$2/1024\" MB\"}" /proc/$PID/status
'
```

Top system memory consumers:
```bash
ssh root@10.42.0.2 'ps aux --sort=-rss | head -6'
```

## Common session gotchas (hard-won)

- **"Detector dies silently every 1–2 minutes."** The Stage-0 NVDEC +
  `appsink + gst_buffer_map` path had an NVMM leak that OOMed the
  container. Fixed as of the Stage 1 port (pad-probe on `nvtracker.src`
  reads `NvDsBatchMeta`; no buffer mapping). If you still see it, you
  regressed the probe path — check `Sources/CGStreamer/nvds_shim.c`.
- **"SSH banner times out even though ping works."** The Jetson is
  CPU-starved, usually from a detector tight-restart loop. If you
  deployed with `--restart-unless-stopped` on a build that crashes
  rapidly, you'll need to power-cycle the device. Never use
  `--restart-unless-stopped` on a build whose stability you haven't
  confirmed.
- **Video panel shows nothing in the browser.** Either the detector
  died (check metrics via proxy), the camera WiFi dropped (`nmcli con
  up "badgers den"` on the Jetson), or mediamtx isn't publishing
  (`systemctl is-active mediamtx` + `curl .../v3/paths/list`). See
  `docs/HANDOFF.md` §9.5 for the full FPS=0 decision tree.
- **VLM descriptions look great but detector crashes faster.** The VLM
  sidecar consumes ~2 GB unified memory; reduces detector headroom. Kill
  it with the `docker stop llama-vlm` recipe above.

## Files in this repo

| path | role |
|---|---|
| `monitor.html` | dashboard (served by the proxy) |
| `monitor_proxy.py` | stdlib reverse proxy, port 8001 |
| `start.sh` | deploys all three apps (detector/gpu-stats/vlm) via wendy |
| `detector-swift/` | the Swift detector we're actively porting — see `PORT_PLAN.md` |
| `detector/` | original Python + DeepStream detector (reference architecture) |
| `vlm/` | llama.cpp sidecar config/Dockerfile |
| `gpu-stats/` | tegrastats Prometheus exporter |
