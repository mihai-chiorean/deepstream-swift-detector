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
metrics and streams MJPEG from the detector. It's served by
`monitor_proxy.py` — a stdlib Python reverse proxy running on
`edge-builder-1`.

### What the proxy does

```
GET /                  → serves monitor.html from the repo root
GET /detector/<path>   → proxies to http://10.42.0.2:9090/<path>
GET /gpu/<path>        → proxies to http://10.42.0.2:9091/<path>
```

Streaming responses (`multipart/x-mixed-replace`) are forwarded chunked,
never buffered, so MJPEG arrives live. Same-origin under port 8001, so
no CORS concerns from the browser. The proxy binds `0.0.0.0` so it's
reachable from the Mac over either ethernet or WiFi.

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
paths `/detector/...` and `/gpu/...`. The device-input box is preserved
for backwards compat (loading the file directly via `file://`), but is
ignored when served through the proxy.

The `<img id="video-stream">` tag has an `onerror` retry: if the stream
fails, it waits 2 s and re-sets `src` with a `?retry=<timestamp>` param
to bust any browser connection cache. So a transient detector restart
clears itself — no manual reload needed.

## Jetson services at a glance

| port | service | under | notes |
|---|---|---|---|
| `22` | sshd | systemd | primary access |
| `5000` | container registry | containerd task | receiver for `swift package build-container-image` push |
| `8090` | VLM sidecar (llama-server, Qwen3-VL) | Docker (`llama-vlm`) | optional; hold ~2 GB unified memory when running |
| `9090` | Swift detector HTTP | containerd task `detector-swift` | `/metrics`, `/stream`, `/healthz`, `/api/vlm_*` |
| `9091` | tegrastats exporter | systemd `tegrastats-exporter.service` | Prometheus CPU/GPU/temp gauges |

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

If `swift package build-container-image` fails with a connection-refused
on port 5000, the registry container's task has stopped:

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

```bash
# stop
ssh root@10.42.0.2 'ctr -n default tasks kill -s SIGKILL detector-swift; sleep 2
                    ctr -n default tasks delete detector-swift
                    ctr -n default containers delete detector-swift'

# start (note: does NOT rebuild Swift code — must swift build first)
cd ~/workspace/samples/deepstream-vision/detector-swift
source ~/.local/share/swiftly/env.sh
swift build --swift-sdk 6.2.3-RELEASE_wendyos_aarch64 -c release
swift package --swift-sdk 6.2.3-RELEASE_wendyos_aarch64 --allow-network-connections all \
  build-container-image --from 10.42.0.2:5000/swift-detector-base:latest \
  --allow-insecure-http both --product Detector \
  --repository 10.42.0.2:5000/detector-swift --architecture arm64 \
  --resources streams.json:/app/streams.json \
  --resources labels.txt:/app/labels.txt \
  --resources yolo26n.onnx:/app/yolo26n.onnx \
  --resources yolo26n_b2_fp16.engine:/app/yolo26n_b2_fp16.engine \
  --resources ffmpeg:/app/ffmpeg
WENDY_AGENT=10.42.0.2 wendy run -y --detach  # omit --restart-unless-stopped on untrusted builds

# apply cgroup cap (MANDATORY — oom_score_adj defaults to -998 otherwise,
# which means OOM kills networking before the detector)
ssh root@10.42.0.2 'CONTAINER=detector-swift /usr/local/bin/detector-cap'
```

## Quick-check recipes

Detector live metrics:
```bash
curl -s http://10.42.0.2:9090/metrics | grep '^deepstream_' | head
curl -s http://edge-builder-1.local:8001/detector/metrics | grep '^deepstream_' | head  # through the proxy
```

Detector stream rate from curl:
```bash
timeout 5 curl -s -N -o /tmp/sample.bin http://10.42.0.2:9090/stream
echo "$(stat -c%s /tmp/sample.bin)b / $(grep -c '^--frame' /tmp/sample.bin) frames in 5s"
```

GPU stats:
```bash
curl -s http://10.42.0.2:9091/metrics | grep '^jetson_' | head
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

- **"Detector dies silently every 1–2 minutes."** See the port plan —
  the NVDEC + `appsink + gst_buffer_map` path has an NVMM leak that OOMs
  the container. Workaround in current baseline: software decode via
  `avdec_h264`. Real fix: the DeepStream pad-probe port.
- **"SSH banner times out even though ping works."** The Jetson is
  CPU-starved, usually from a detector tight-restart loop. If you
  deployed with `--restart-unless-stopped` on a build that crashes
  rapidly, you'll need to power-cycle the device. Never use
  `--restart-unless-stopped` on a build whose stability you haven't
  confirmed.
- **Stream shows black in the browser after a few seconds.** Either
  the detector died (check metrics via proxy), or you're hitting the
  flaky WiFi path because the proxy is down (restart `monitor_proxy.py`).
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
