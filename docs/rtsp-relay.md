# RTSP Relay — deepstream-vision parallel benchmark

## Tool + Version

**mediamtx v1.17.1** — statically-linked Go binary, aarch64 build (`linux_arm64`).

Chosen over a GStreamer `rtspclientsink` pipeline because:
- Single ~60 MB static binary, zero runtime dependencies on WendyOS/Yocto.
- Trivial YAML config, no GStreamer element wiring.
- ~36 MB RSS at steady state vs ~80–120 MB for an equivalent GStreamer server pipeline.
- Native multi-reader fan-out: one upstream RTSP pull, unlimited downstream readers with independent flow control.

## Deployment

Deployed directly on the Jetson (`root@10.42.0.2`) as a systemd unit.

### Files

| Path | Purpose |
|------|---------|
| `/usr/local/bin/mediamtx` | Binary (downloaded from GitHub releases) |
| `/etc/mediamtx.yml` | Configuration |
| `/etc/systemd/system/mediamtx.service` | systemd service |

### Key config (`/etc/mediamtx.yml`)

```yaml
logLevel: info
logDestinations: [stdout]
rtsp: true
rtspAddress: :8554
rtspEncryption: "no"
rtspTransports: [tcp, udp]
rtmp: false
hls: false
webrtc: false
srt: false
paths:
  relayed:
    source: rtsp://jetson:jetsontest@192.168.68.69:554/stream1
    sourceOnDemand: false   # always pull, even with no readers
    maxReaders: 0           # unlimited concurrent downstream clients
```

### Commands

```bash
# Download binary (run on Jetson or transfer via scp)
wget https://github.com/bluenviron/mediamtx/releases/download/v1.17.1/mediamtx_v1.17.1_linux_arm64.tar.gz -O /tmp/mediamtx.tar.gz
tar xzf /tmp/mediamtx.tar.gz -C /tmp
cp /tmp/mediamtx /usr/local/bin/mediamtx && chmod +x /usr/local/bin/mediamtx

# Enable + start
systemctl daemon-reload
systemctl enable mediamtx
systemctl start mediamtx
systemctl status mediamtx
```

Survives reboots via the `[Install] WantedBy=multi-user.target` directive.

## Downstream URL

```
rtsp://10.42.0.2:8554/relayed
```

No credentials required for downstream consumers (mediamtx config uses `any` user with no password by default). The camera credentials (`jetson:jetsontest`) are embedded in the config source URL only.

### Camera constraint

The TP-LINK camera at `192.168.68.69:554` enforces a **single RTSP session** limit. Only mediamtx holds that session; all detectors subscribe to the relay. The Swift detector must not point directly at the camera while the relay is running — they cannot coexist.

## Verification

```bash
# Single client
ffprobe -v error -rtsp_transport tcp rtsp://10.42.0.2:8554/relayed -show_streams

# Expected: codec_name=h264, width=1920, height=1080, avg_frame_rate=20/1
# Plus audio: codec_name=pcm_alaw, sample_rate=8000 (G.711 passthrough)
```

## Concurrency Test Results

Two simultaneous `ffprobe` sessions, each reading for 30 seconds:

| Client | codec | width | height | avg_frame_rate | nb_read_packets |
|--------|-------|-------|--------|----------------|----------------|
| 1 | h264 | 1920 | 1080 | 20/1 | 597 |
| 2 | h264 | 1920 | 1080 | 20/1 | 597 |

Both sessions ran in parallel (mediamtx logs show two concurrent sessions `b1e1f43e` and `4604f498`), received identical packet counts, and completed cleanly. No serialization observed.

## Resource Usage (60 s steady state, one camera session held)

Measured on Jetson Orin (WendyOS edgeOS 0.10.5 / aarch64):

| Metric | Value |
|--------|-------|
| VmRSS | 36.5 MB |
| VmPeak | 1280 MB (virtual; normal for Go runtime) |
| Threads | 11 |
| CPU (5 s avg, idle, no readers) | ~0.4% |
| CPU (expected, 2 readers) | ~1–2% |

System context at measurement: 1083 MB used of 7620 MB total; 6537 MB available.

## Friction + Caveats

1. **Single-session camera limit.** The TP-LINK camera drops additional RTSP connections with `EOF` immediately after announcing tracks. The relay eliminates this by being the sole upstream client. Any process that connects directly to the camera while mediamtx is running will get `Operation not permitted` (RTSP 403-equivalent from the camera firmware).

2. **Camera EOF on dual PLAY.** When mediamtx tried UDP RTP (default), the camera dropped it. Forcing `sourceProtocol: tcp` in config — or equivalently, listing only `tcp` in `rtspTransports` to influence the source pull — stabilizes the session. The v1.17.1 binary auto-negotiates TCP for the source after one retry.

3. **Disk space.** `/dev/nvme0n1p2` is at 92% (19 GB / 22 GB used). The binary is ~60 MB. No container image was pulled; the binary runs directly. Avoid pulling large container images without first freeing space.

4. **No auth stripping needed.** Downstream consumers connect to `rtsp://10.42.0.2:8554/relayed` without credentials. The mediamtx default auth policy (`any` user, no password, read + publish actions) permits anonymous reads.

5. **Swift detector.** The `detector-swift` container task was stopped to allow mediamtx to acquire the camera session. The next task will re-point the Swift detector to `rtsp://10.42.0.2:8554/relayed` before restarting it.

6. **WiFi recovery.** If the Jetson drops off the network: `ssh root@10.42.0.2 'nmcli con up "badgers den"'`. mediamtx will reconnect to the camera automatically on `Restart=on-failure` (RestartSec=5s).
