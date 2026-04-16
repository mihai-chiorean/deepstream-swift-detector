STATUS: Both stages complete as of 2026-04-15. Stage 2 measurements: preprocess 26.5 ms (was ~31.5 ms), inference 22.8 ms, nvmap ~206 MB steady-state (down from 224 MB with the MJPEG branch; the ~23 MB drop is from removing nvjpegenc + nvdsosd, not from the relay path — relay vs. direct A/B both reach 206 MB), VmRSS ~595 MB. WHEP 201 verified via direct Jetson (:8889) and via monitor_proxy (:8001/webrtc/relayed/whep).

**HISTORICAL:** this document describes the plan as-of design time — both Stage 1 and Stage 2 shipped. The body below is written in future tense from the original authoring. For current state, pipeline shape, and measurement caveats, see `docs/HANDOFF.md`. This file is preserved as a decision record.

# MJPEG Decoupling: Team Synthesis

## Team positions

| Reviewer | Pick | Core argument |
|---|---|---|
| gpu-engineer | **Option 4 with RTP passthrough** | Zero GPU cost. Permanently removes `nvjpegenc + 2× nvvideoconvert + nvdsosd`. Reclaims ~8 ms of VIC/GPU budget. Option 1 is a band-aid. |
| backend-architect | **Option 4 via mediamtx+webrtcsink + WebSocket detections** | `mediamtx` already in stack → Option 4 is cheap now. Option 1 accrues technical debt immediately. |
| frontend-developer | **Option 1 server + canvas+ReadableStream client** | Tab-glancer user, 5 fps is legible, WebRTC browser sync is fragile (RVFC Firefox gap, SEI insertion non-trivial). |
| swift-backend | **Option 1 — shipped correctly** | Four specific gotchas (name the valve, `try_pull_sample`, `videodrop` not `videorate`, thin `g_object_set` shim). Frame-sync is a hidden 2–3× effort multiplier on full Option 4. |

## The real disagreement

GPU + backend see Option 4 as **architecturally committed to, already-cheap-with-mediamtx, sub-100ms and multi-viewer for free**. They reject Option 1 as debt.

Frontend + swift-backend see Option 4 as **a product feature** — browser-side frame alignment and clean teardown under tab-close/TCP-drop conditions — that the server-side elegance hides.

Both sides are correct about what they're seeing. The correct answer is **not a compromise option**; it's a staged sequence where Option 1's scope is deliberately minimal (stop wasting GPU when idle) and Option 4 is the **committed destination** with a concrete path.

## Staged architecture

### Stage 1 — kill idle waste (ship this week)

Goal: detection loop never waits on MJPEG when no one is watching. Residual contention while one viewer is connected is tolerable for now.

**Server — detector-swift:**

1. **Named valve + frame drop in the pipeline string.** Replace the MJPEG branch with:
   ```
   tee. ! queue max-size-buffers=2 leaky=downstream
        ! valve name=mjpeg_valve drop=true
        ! videodrop rate=5
        ! nvvideoconvert
        ! nvdsosd
        ! nvvideoconvert
        ! capsfilter caps=video/x-raw,format=RGBA
        ! nvjpegenc quality=85
        ! appsink name=mjpeg_sink
   ```
   - Use `videodrop` from `gst-plugins-bad` — not `videorate` — because the CDI-injected GLib + Ubuntu base collision raises `g_once_init_leave_pointer: undefined symbol` on `libgstvideorate.so`. `videodrop` is drop-only (no timestamp interpolation), which is what we want: rate ≈ 1/4 of source, no frame synthesis.
   - `capsfilter` between nvvideoconvert and nvjpegenc locks caps so valve reopen doesn't renegotiate and skip the first frame.
   - `valve name=mjpeg_valve` is mandatory: otherwise `gst_bin_get_by_name` can't retrieve a handle to toggle `drop` at runtime.

2. **Thin C shim** in `Sources/CGStreamer/shim.h`:
   ```c
   static inline void wendy_gst_set_bool_property(
       GstElement *element, const gchar *name, gboolean value
   ) { g_object_set(G_OBJECT(element), name, value, NULL); }
   ```

3. **Pull-loop resilience**: switch `MJPEGSidecar`'s pull from the blocking `wendy_gst_pull_sample` to `wendy_gst_try_pull_sample` (already in `shim.h`) with a 200 ms timeout. Without this, the pull blocks indefinitely when the valve is closed.

4. **Valve toggle at 0↔1 subscriber boundary** in `MJPEGSidecar.subscribeFrames` / `unsubscribe`. `g_object_set("drop", ...)` is thread-safe on a PLAYING pipeline; no state transition needed.

5. **WebSocket `/detections` endpoint** via HummingbirdWebSocket. Per-client subscriber pattern — **not the shared `DetectionStream`**, which uses `.bufferingNewest(4)` and would silently drop frames across multiple clients. The correct shape mirrors `MJPEGSidecar`: a map of `client-id → AsyncStream<Data>.Continuation`, fed by a broadcast loop that reads from `DetectionStream` and fans out.
   - Message schema: `{ frameNum, timestampNs, detections: [...] }`.
   - Handler wraps `outbound.write` in `do/catch` + `defer { unsubscribe }` so tab-close / TCP drop cleans up.

**Client — monitor.html:**

1. **Replace `<img src="/stream">` with two stacked elements** and a canvas overlay:
   ```html
   <div class="video-container">
     <img id="video-stream" src="/stream" />
     <canvas id="detection-overlay"></canvas>
   </div>
   ```
   Keep `<img>` for the MJPEG stream — browsers decode multipart/x-mixed-replace JPEGs efficiently and there's no need to parse `--boundary` in JS. The overlay `<canvas>` sits absolutely-positioned on top.

2. **`/detections` WebSocket** on page load, stored in a `latestDetections` slot keyed by frameNum.

3. **Bbox render loop on `requestAnimationFrame`** — not on JPEG arrival. The `<img>` gives us no per-frame event anyway. At 60 Hz RAF, the canvas draws the most-recent WebSocket detection every tick.
   - **Honest caveat — the overlay will be semantically wrong under motion.** MJPEG via `<img>` lags the real world by roughly the JPEG encode+send interval (100–300 ms on a 5 fps feed). Detections via WebSocket arrive ~50 ms after the frame they describe. The canvas draws the freshest detection against a stale JPEG. A car moving at 30 mph travels ~2 m in 150 ms; at 1920×1080 from a fixed camera that's easily 40–80 pixels of horizontal displacement. **Boxes will visibly trail the car in the JPEG by roughly a car-length.** Acceptable for "is the detector running and finding cars?" Not acceptable for "where exactly is the car?". Stage 2 fixes this via PTS-based alignment and sub-100 ms WebRTC video.

**Stage 1 success criteria:**
- No HTTP client: `deepstream_preprocess_latency_ms ≈ 26 ms` (matches MJPEG-OFF measurement), `nvmap ≈ 181 MB`.
- One client: preprocess ≈ 28 ms, nvmap ≈ 200 MB.
- `deepstream_fps` steady at 20 regardless.
- `/detections` WebSocket broadcasts cleanly; Nth viewer doesn't starve clients 1..N-1.

### Stage 2 — delete MJPEG from the detector (starts concurrently with Stage 1; de-risking begins now)

Goal: reclaim the GPU budget currently consumed by the MJPEG branch. Measured with a full MJPEG-on vs MJPEG-off A/B: **5.3 ms saved on preprocess** (31.5 → 26.2 ms) plus 43 MB of nvmap. The gpu-engineer estimated ~8 ms factoring in VIC-pool and render-latency we don't directly measure. Either way it's 10–16% of the 50 ms frame budget — material, not rounding.

Remove `nvjpegenc + 2× nvvideoconvert + nvdsosd + valve + videodrop` from the detector graph. Detector becomes pure detection: decode → mux → infer → track → pad probe → done.

**De-risking tasks to start THIS WEEK in parallel with Stage 1:**

- Configure `mediamtx` to publish the relayed stream via WebRTC (`webrtc` protocol). Verify a `<video>` element on the macbook can consume it **through the `monitor_proxy.py` over the USB-C NCM link** (this is the non-obvious part — WebRTC's SDP may advertise IPs that don't route from the macbook's vantage point, requiring `externalAuthenticationURL` or explicit `publicIPs` in mediamtx config).
- Prototype PTS propagation in the C shim: extract `GST_BUFFER_PTS(buf)` at the probe, add `ptsNs` to `WendyFrameTiming`, include in the `/detections` WebSocket message schema.

These two tasks should happen during the Stage 1 implementation week, not after.

**Architecture:**

1. **`mediamtx`** (already deployed, already doing RTSP→RTSP relay) **adds WebRTC publication** of the same H.264 stream via its built-in `webrtc` protocol. Zero transcode, zero GPU work — it's RTP repacketization. mediamtx ships this out of the box; it's a config change, not new code.

2. **Detector emits only detections** via the `/detections` WebSocket built in Stage 1. No pipeline changes beyond removing the MJPEG branch.

3. **Browser:**
   - `<video id="webrtc-video">` subscribes to `rtsp://10.42.0.2:8889/relayed/whep` (mediamtx's WHEP endpoint).
   - `<canvas>` overlay stack as in Stage 1.
   - Sync mechanism: each detection carries `timestampNs` = PTS of the frame that produced it (extract in the Swift pad probe from `GST_BUFFER_PTS`). Browser uses `requestVideoFrameCallback` (Chrome only initially — document the browser requirement) to get the `<video>` element's current presentation timestamp, matches against the detection-message PTS buffer, and draws the closest match. Sub-frame alignment.

4. **Firefox fallback**: if RVFC is unavailable, draw bboxes on RAF using the latest detection (same degraded sync as Stage 1). Browser-compatibility matrix documented in OPERATIONS.md.

**Stage 2 wins over Stage 1:**
- +8 ms inference headroom (preprocess back to the pure nvinfer cost, ~22 ms, from the current 26-31 ms).
- Sub-100 ms glass-to-glass video latency (vs MJPEG's 500 ms).
- Multi-viewer trivially (WebRTC fan-out via mediamtx).
- Frame-accurate overlay sync via PTS matching.
- Detection stream is reusable for Grafana/alerts/analytics without changing anything.

**Stage 2 effort:** mediamtx WebRTC config **0.5–1 day** (routing over USB-C + SDP/ICE debugging is where it bites). Removing MJPEG from the detector pipeline ~1 hour. Browser WebRTC client + RVFC-based PTS sync ~1 day. Total: **1.5–2.5 days**.

## What gets rejected outright

- **Option 2 (CUDA stream priority)**: 0.5–1 ms savings against an 8 ms Stage 2 target, requires patching DS plugins. All four reviewers agreed. Dead.
- **Option 3 (separate process)**: 2× decode + same side-channel complexity as Option 4, minus the viewer-side wins. Strictly dominated.

## Real open blockers (both Stage 2)

1. **mediamtx WebRTC over USB-C NCM**: viability of WebRTC signaling from the macbook through `monitor_proxy.py` to `10.42.0.2:8889/whep`. SDP/ICE may advertise unreachable addresses. Needs spike NOW, not after Stage 1 ships.
2. **PTS propagation**: extract `GST_BUFFER_PTS(buf)` at the probe, expose via `WendyFrameTiming`, include in the `/detections` message. Mechanical but load-bearing for Stage 2's frame-accurate sync.

## Summary

- **Stage 1 (this week):** valve-gated MJPEG + `/detections` WebSocket. ~1 day server + client. Kills idle GPU waste. Overlay is **semantically wrong under motion** (boxes trail moving cars by ~1 car-length); acceptable only as a liveness/coarse-monitoring dashboard.
- **Stage 2 de-risking (this week, concurrent with Stage 1):** prototype mediamtx WebRTC over the USB-C path and the PTS propagation in the C shim. Cheap if it works, critical-path blocker if it doesn't.
- **Stage 2 implementation (next sprint):** mediamtx WebRTC + remove MJPEG from detector + RVFC/PTS sync. 1.5–2.5 days. Reclaims 5.3 ms measured (~10% of frame budget) and possibly more with VIC-pool effects. Sub-100 ms latency. Frame-accurate overlays. Multi-viewer.
- Stage 2 is committed. Stage 1's code (WebSocket schema, client canvas, subscriber fan-out) survives the Stage 2 transition unchanged; only the video source swaps from `<img src="/stream">` to `<video>` + WebRTC.
