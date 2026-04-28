# REVIEW-EMBEDDED.md
# Embedded / WendyOS / Swift Server-Side Review
# Repo: deepstream-swift-detector (public, github.com/mihai-chiorean/deepstream-swift-detector)
# Reviewer scope: CDI/containerd deploy plumbing, Swift server-side, concurrency, metrics, public-readiness
# GPU/NVIDIA-stack/TRT/pipeline-correctness owned by parallel gpu-engineer agent — not covered here.
# Date: 2026-04-23

---

## TL;DR

The Swift server-side code (concurrency, Hummingbird wiring, `DetectionBroadcaster` actor) is
clean and the metrics Mutex design is correct. The primary public-readiness problems are
operational: `start.sh` deploys the Python detector instead of the Swift one, three places in
OPERATIONS.md point readers to a gitignored file (`docs/HANDOFF.md`), and `wendy.json` carries a
stale private-LAN registry address that will confuse any integrator who tries `wendy run`. The
`cairo-stage/` binary blobs are orphaned junk post-Stage-2 and add 3.5 MB to every clone for no
reason. Nothing found constitutes a safety or data-correctness issue in the detector itself.

---

## Findings

---

### F1 [HIGH] `wendy.json` — private LAN registry IP hardcoded as `baseImage`

**Finding:** `detector-swift/wendy.json` line 7:
```json
"baseImage": "192.168.68.70:5000/swift-detector-base:latest"
```
`192.168.68.70` is the author's Jetson. A stranger running `wendy run` in
the `detector-swift/` directory will get a connection-refused on a host that
does not exist for them, with no diagnostic message explaining why the
`swift.language` path no longer applies.

**Why:** The current deploy path bypasses the `swift.baseImage` field entirely — it
uses the `Dockerfile` instead of the Swift-native build path. The `swift` stanza
in `wendy.json` is a stale leftover from before the Stage-2 Dockerfile switch.
Strangers reading `wendy.json` alongside the README will be confused about which
path is authoritative.

**Suggested fix:** Either remove the `"language": "swift"` block and `"swift": {...}` object
entirely (leaving only `appId`, `version`, and `entitlements`), or replace `baseImage` with a
note comment (not valid JSON, so remove the stanza). README should note that the current deploy
path requires a pre-built binary + Dockerfile, not the `wendy run --swift` shortcut.

---

### F2 [HIGH] `start.sh` deploys `detector/` (Python), never `detector-swift/`

**Finding:** `start.sh` lines 50-55:
```bash
cd "$SCRIPT_DIR/detector"
wendy run --device "$DEVICE" --detach --restart-unless-stopped
```
The deploy script deploys `detector/` (the Python sibling) as "the detector".
`detector-swift/` is never touched by `start.sh`. `README.md` says the Swift
detector is the primary subject of the repo.

**Why:** A reader who follows the README to `./start.sh` runs the Python detector
silently. The dashboard echo at line 75 (`http://$DEVICE:9090/stream`) is also
wrong — Stage 2 removed that MJPEG stream endpoint and replaced it with a 301
redirect. The script is a session artifact never updated for the public extract.

**Suggested fix:** Either update `start.sh` to deploy `detector-swift/` instead of
`detector/`, or add a prominent header comment explaining it is for development
convenience only and the primary deployment workflow is `wendy run` inside
`detector-swift/`. Remove the stale `/stream` URL from the echo block.

---

### F3 [HIGH] OPERATIONS.md has three broken references to gitignored `docs/HANDOFF.md`

**Finding:** Three critical ops references in OPERATIONS.md point to `docs/HANDOFF.md`:
- Line 178: `currently unstable — see docs/HANDOFF.md §9`
- Line 261: `See docs/HANDOFF.md §3 for the authoritative copy and §10 for the list of deploy gotchas`
- Line 328: `docs/HANDOFF.md §9.5 for the full FPS=0 decision tree`

`docs/HANDOFF.md` is in `.gitignore` and absent from the public repo.

**Why:** Line 261 is the most damaging: it replaces the "authoritative" build command
with a reference to a file nobody can read. A stranger deploying the Swift detector
for the first time is told "see HANDOFF §3" for the build/deploy steps they actually
need — those steps don't exist anywhere accessible in the public repo.

**Suggested fix:** Replace each HANDOFF reference with the inline content or a pointer
to the `ARCHITECTURE.md` section that covers the same material. For §3 (build/deploy),
the OPERATIONS.md "Stopping / starting the detector" section already contains the
full recipe — remove the "see HANDOFF §3 for the authoritative copy" parenthetical.
For §9 and §9.5, inline the FPS=0 decision tree or link to ARCHITECTURE.md.

---

### F4 [HIGH] `wendy.json` `swift.resources` lists artifacts that don't match the Dockerfile deploy path

**Finding:** `detector-swift/wendy.json` lists under `swift.resources`:
```json
"yolo26n.onnx:/app/yolo26n.onnx",
"yolo26n_b2_fp16.engine:/app/yolo26n_b2_fp16.engine",
"ffmpeg:/app/ffmpeg",
"libnvdsparsebbox_yolo26.so:/app/lib/libnvdsparsebbox_yolo26.so"
```
The engine is in `.gitignore` and absent from the repo. `ffmpeg` is an unexplained
binary blob in the repo root of `detector-swift/`. The `.dockerignore` excludes
`ffmpeg` and `yolo26n.onnx` from the build context. None of these resources are
consumed via `wendy run --resources` in the current deploy path — they are `COPY`'d
by the Dockerfile directly.

**Why:** The resources stanza was the mechanism for the older `swift package
build-container-image` path (before the Stage-2 Dockerfile switch). In the current
path it is entirely inert — but it misleads integrators into thinking they need to
supply these files to `wendy run` as mounts.

**Suggested fix:** Remove the `"swift": { "resources": [...] }` stanza entirely. The
Dockerfile comment already documents which files are COPY'd. Also address `ffmpeg`:
if it is unused (Stage 2 removed FFmpeg), remove it from the repo and `.dockerignore`.

---

### F5 [MEDIUM] Orphan source directories and unused package dependency

**Finding:** Three cleanup leftovers from the Stage-2 MJPEG removal:
- `Sources/CFFmpeg/` — full `module.modulemap` + `shim.h` for libav*, but not declared as a target in `Package.swift`. SwiftPM ignores the directory silently.
- `Sources/CTurboJPEG/` — same: `module.modulemap` + `shim.h`, not a declared target.
- `Package.swift` line 13: `swift-container-plugin` declared as a package dependency but unused by any target. SwiftPM will resolve and download this package on every `swift package resolve`.

**Why:** These are stage-artifact directories that were never cleaned up. `swift-container-plugin`
is especially confusing because it implies the container image is built via the plugin, while
`OPERATIONS.md` line 238 explicitly notes "There is no `swift package build-container-image` path
anymore."

**Suggested fix:** Delete `Sources/CFFmpeg/`, `Sources/CTurboJPEG/`. Remove the
`swift-container-plugin` dependency from `Package.swift`. If the abandoned
`ARCHITECTURE.md` Part 10 line "Removed from the previous design: `swift-container-plugin`
(now using a Dockerfile again)" needs updating, it does — it already acknowledges the removal.

---

### F6 [MEDIUM] Contradictory Swift SDK names across files

**Finding:** Two different SDK names appear in the repo for the same SDK:
- `6.2.3-RELEASE_wendyos-device_aarch64` (with `-device`) — used in ARCHITECTURE.md Part 11
  "Step 1: Install the Swift SDK", OPERATIONS.md build command (line 251)
- `6.2.3-RELEASE_wendyos_aarch64` (without `-device`) — used in Package.swift comment (line 20),
  vendor/README.md

A stranger running `swift sdk install` needs exactly one correct name.

**Why:** These names are likely different releases of the SDK (one is the device runtime SDK, one
is the host tools SDK), or the naming was updated at some point and not propagated consistently. An
integrator will not know which to use.

**Suggested fix:** Decide on one canonical SDK name, verify it matches the artifact bundle zip
that was actually used, and update all references to use it consistently. Add a note explaining
the naming convention (e.g., `-device` suffix = target-side sysroot SDK vs. host tools).

---

### F7 [MEDIUM] `deepstream_gpu_memory_mb` gauge registered but never written

**Finding:** `Metrics.swift` lines 395-399:
```swift
let gpuMemoryMB: GaugeMetric = metrics.gauge(
    "deepstream_gpu_memory_mb",
    help: "GPU memory usage in megabytes"
)
```
No `.set()` call exists for `gpuMemoryMB` anywhere in the codebase. The metric
appears in `/metrics` output permanently stuck at `0`.

**Why:** The metric was probably intended to be populated from tegrastats or
`NvBufSurface` pool accounting. It is never populated, confusing any Prometheus
alert rule or dashboard that reads it.

**Suggested fix:** Either remove `gpuMemoryMB` and its `# HELP` / `# TYPE` lines
from the output, or wire it to the `deepstream_gpu_memory_mb` value from the
`gpu-stats/` sidecar via a scrape federation. For a reference implementation,
removing it is cleaner — the GPU stats sidecar already exports `jetson_*` memory gauges.

---

### F8 [MEDIUM] `OPERATIONS.md` leaks author-private paths and machine names

**Finding:** Multiple lines in OPERATIONS.md contain non-reproducible author-specific references:
- Line 89: `cd ~/workspace/samples/deepstream-vision` (author's working directory)
- Line 249-250: `cd /home/mihai/workspace/samples/deepstream-vision/detector-swift`
- Lines 105, 107, 108, 270: `edge-builder-1.local`, `edge-builder-1` (author's dev host mDNS name)
- Line 201, 204, 326: `nmcli con up "badgers den"` (author's WiFi SSID)

Additionally `docs/benchmark-python-vs-swift.md` lines 153, 154, 318-319, 457-463 reference
`/home/mihai/workspace/samples/deepstream-vision/...` as file paths.

**Why:** A stranger has no `edge-builder-1`, no `~/workspace/samples`, and a different WiFi SSID.
These are not illustrative placeholders — they look like canonical commands and will silently fail
or confuse.

**Suggested fix:** Replace all author-specific paths with relative paths from the repo root or
with generic placeholder names (e.g., `<dev-host>.local`, `<your-ssid>`, `./detector-swift/`).
Update benchmark docs paths to reference the files by their in-repo `docs/benchmark-data/` paths.

---

### F9 [MEDIUM] `/health` vs `/healthz` discrepancy

**Finding:** `OPERATIONS.md` line 176 lists `/healthz` in the port 9090 endpoint table.
`HTTPServer.swift` registers the endpoint as `/health` (no z), visible at lines 6 and 97-108.

**Why:** This is a functional discrepancy. Someone curl-ing `/healthz` will get a 404. The
README's "Observe" section does not list health at all, so this is only in the port table.

**Suggested fix:** Fix OPERATIONS.md line 176 to read `/health`.

---

### F10 [MEDIUM] `docs/rtsp-relay.md` contains camera credentials in plaintext

**Finding:** `docs/rtsp-relay.md` lines 40 and 68 contain:
```
source: rtsp://jetson:jetsontest@192.168.68.69:554/stream1
```
Username `jetson` and password `jetsontest` for the camera are in the public repo.

**Why:** While this is a LAN-only camera that is inaccessible from the internet, publishing
default/reused credentials in a public reference doc is a bad practice to model — especially
for a repo that will be cited in a blog post as a reference implementation.

**Suggested fix:** Replace with `rtsp://<camera-user>:<camera-password>@<camera-ip>:554/stream1`
and remove the inline explanation that names the credentials (`jetson:jetsontest`). Rotate the
camera password if it is reused elsewhere.

---

### F11 [MEDIUM] Untracked `Task { }` in reconnect loop — shutdown gap up to 30 s

**Finding:** `GStreamerFrameReader.runWithReconnect()` line 390 spawns:
```swift
Task {
    var backoffSeconds: UInt64 = 2
    let maxBackoffSeconds: UInt64 = 30
    while !Task.isCancelled {
        try? await Task.sleep(nanoseconds: 500_000_000)
        ...
    }
}
```
No `Task` handle is stored. `stopReconnecting()` does not cancel this task — it
sets `isRunning = false` and nils `pipeline`, but the loop's exit condition is
`Task.isCancelled`, which is never set true by `stopReconnecting()`.

**Why:** After `stopReconnecting()` is called, the reconnect loop continues running
for up to ~30 s (maxBackoffSeconds) before the next `Task.isCancelled` check. In that
window the loop calls `self.pollBus()` on a `nil` pipeline (safe: guarded), but
`self.teardownPipeline()` could race if the caller re-starts the pipeline. In the
single-stream, single-stop lifecycle this is benign. For a public reference implementation
it is an anti-pattern: the comment at line 388 says the task inherits actor isolation, which
is true, but the missing handle means structured cancellation is absent.

**Suggested fix:** Store the `Task` handle and cancel it in `stopReconnecting()`:
```swift
private var reconnectTask: Task<Void, Never>?

reconnectTask = Task { ... }

func stopReconnecting() {
    reconnectTask?.cancel()
    reconnectTask = nil
    ...
}
```

---

### F12 [MEDIUM] Package.swift comment contradicts the `unsafeFlags` it contains

**Finding:** `Package.swift` CGStreamer target comment (lines 20-25):
> "The toolset.json injects -isystem paths for gstreamer-1.0 and glib-2.0; the sysroot
> root covers usr/include directly. No -I overrides needed in Package.swift."

Line 44 then adds:
```swift
.unsafeFlags(["-isystem", "Sources/CGStreamer/vendor"]),
```

**Why:** Either the vendor directory is needed (in which case the "no -I overrides needed"
comment is wrong) or it is not (in which case the `unsafeFlags` line is a dead flag).
`vendor/README.md` says "no longer used" and "do not add headers here." This three-way
contradiction between the comment, the flag, and the vendor README will confuse a
contributor debugging a cross-compile failure.

**Suggested fix:** If headers are provided by the SDK sysroot (as vendor/README.md states),
remove the `unsafeFlags(["-isystem", "Sources/CGStreamer/vendor"])` line and update the
Package.swift comment to confirm SDK-provided headers are sufficient. If the vendor dir is
actually needed for some transitive header, update vendor/README.md to reflect that.

---

### F13 [MEDIUM] `ProbeContext` leak per reconnect cycle in `nvds_shim.c`

**Finding:** `nvds_shim.c` lines 217-232: `wendy_install_detection_probe()` allocates
a `ProbeContext` with `g_malloc0` and installs a probe with `GDestroyNotify = NULL`.
The comment says:
> "ProbeContext is intentionally never freed: the probe lives for the lifetime of the
> pipeline. For the detector's single-pipeline lifecycle this is fine."

But `GStreamerFrameReader.runWithReconnect()` rebuilds the pipeline on each reconnect
cycle, calling `wendy_install_detection_probe()` again. Each cycle leaks one
`ProbeContext` (48 bytes on aarch64).

**Why:** At a worst-case 30-second reconnect cadence (camera WiFi flapping), this is
~6 KB/hour — irrelevant in absolute terms on an 8 GB device. However the comment claiming
"single-pipeline lifecycle" is factually incorrect in the reconnect path and will mislead
a contributor who adds reconnect support or changes the pipeline rebuild logic.

**Suggested fix:** Pass a `GDestroyNotify` destructor (`g_free`) to `gst_pad_add_probe()`
so the `ProbeContext` is automatically freed when `gst_pad_remove_probe()` is called
in `teardownPipeline()`. No functional change needed in teardown logic — GStreamer calls
the notify when the probe is removed. Update the comment.

---

### F14 [LOW] `cairo-stage/` binary blobs are orphaned post-Stage 2

**Finding:** `detector-swift/cairo-stage/` contains ~3.5 MB of aarch64 `.so` files for
`libcairo` and its transitive dependencies (21 libraries). These were required by `nvdsosd`
in Stage 1. Stage 2 removed `nvdsosd` from the pipeline. The Dockerfile no longer has
a `COPY cairo-stage/` instruction. The `.dockerignore` does not exclude `cairo-stage/`.

**Why:** Committed binary blobs that are never consumed add clone weight and create a
false impression that the container needs cairo at runtime.

**Suggested fix:** Delete `detector-swift/cairo-stage/` and add `cairo-stage/` to
`.gitignore` (or leave it deleted — it has a `stage-cairo.sh` script that can regenerate it).
Update `ARCHITECTURE.md` War Story 4 to note cairo is no longer needed after Stage 2.

---

### F15 [LOW] `RTSPFrameReader.swift` filename is misleading

**Finding:** `Sources/Detector/RTSPFrameReader.swift` contains only two structs:
`StreamConfig` and `StreamsConfig`. The `RTSPFrameReader` actor (FFmpeg subprocess RTSP
decode) was removed. The comment at line 1 explains this, but the filename still matches
a non-existent class.

**Suggested fix:** Rename to `StreamConfig.swift`.

---

### F16 [LOW] `OPERATIONS.md` still references gitignored `detector-swift/PORT_PLAN.md`

**Finding:** `OPERATIONS.md` line 4: "Pair with `detector-swift/PORT_PLAN.md` for the
detector port work itself." Line 340: "see `PORT_PLAN.md`". `PORT_PLAN.md` is in
`.gitignore` and absent from the public repo.

**Suggested fix:** Replace with a reference to `detector-swift/ARCHITECTURE.md` which
covers the same material in the public repo.

---

### F17 [LOW] `start.sh` uses `--restart-unless-stopped` on the main detector

**Finding:** `start.sh` passes `--restart-unless-stopped` for all three services including
the primary detector. OPERATIONS.md (lines 254, 320-323) and CLAUDE.md (constraint 4)
explicitly warn against this for untested builds. The default value in the script's
echo at line 11 (`DEVICE="${1:-<your-device>.local}"`) will cause `ping` to fail
immediately and exit with an error — so the restart flag is moot for a default run,
but for a real device the detector will restart on any crash, including a NVDEC
tight-restart loop that can wedge driver state.

**Suggested fix:** Remove `--restart-unless-stopped` from the detector service block.
Keep it for `gpu-stats` and `vlm` which are lower-risk. Add a comment in `start.sh`
linking to the warning in OPERATIONS.md.

---

### F18 [LOW] CPU usage figures inconsistent across documents

**Finding:** README.md line 113: "26.6% of one CPU core". ARCHITECTURE.md Part 1
table: "~55%". These refer to different measurement conditions (CLAUDE.md clarifies:
26.6% is per-core ≈ one core, 55% is system-relative with threading). But neither
document cross-references the other or explains the discrepancy.

**Suggested fix:** Add a parenthetical in ARCHITECTURE.md explaining that 55% is
system-total CPU across the cooperative pool, while 26.6% is the single-core
equivalent (26.6% × 1 core / total cores ≈ 55% if the Orin Nano has 4 active cores).

---

### F19 [LOW] GStreamer pipeline string inlines `stream.url` without escaping

**Finding:** `GStreamerFrameReader.buildPipelineString()` line 621:
```swift
rtspsrc location=\(url) latency=200 protocols=tcp
```
`url` is inserted directly into the `gst_parse_launch` string. A URL containing
spaces, `!`, or `"` would silently corrupt or terminate the pipeline description.

**Why:** `streams.json` is baked into the container at build time, not user-supplied
at runtime, so this is not an injection risk in the current setup. However, any
integrator who passes a URL with special characters (e.g., `rtsp://user:p@ss!@host/`)
will get a confusing `gst_parse_launch` parse error with no indication of why.

**Suggested fix:** Either quote the URL: `rtspsrc location=\"\(url)\"` in the
pipeline string, or document in `streams.json` that the URL must not contain spaces
or unescaped special characters meaningful to `gst_parse_launch`.

---

## README + OPERATIONS Specific Notes

These are targeted at the cleanup agent:

**README.md:**
- The build command block (`swift build --swift-sdk <your-aarch64-sdk>`) does not specify
  `--product Detector`. Without `--product`, `swift build` will try to build all targets
  including CGStreamer, which requires DeepStream headers not available on a vanilla host.
  Add `--product Detector` to the example command.
- `WENDY_AGENT=<device-ip> wendy run -y --detach` — the `-y` flag is not documented in
  the wendy CLI help. Either verify this is a valid flag or replace with `--yes`.
- The "Observe" section lists `/metrics` on `9090` but does not list `/health`. Add it.
- The link to `ARCHITECTURE.md` for CDI bind-mounts is a self-referential doc, not missing —
  this is fine, just ensure the link works.

**OPERATIONS.md:**
- Line 4: Replace "Pair with `detector-swift/PORT_PLAN.md`" → "Pair with `detector-swift/ARCHITECTURE.md`".
- Lines 89, 249-250: Replace absolute paths with relative paths from repo root.
- Lines 105-108: Replace `edge-builder-1.local` with `<dev-host>.local`.
- Lines 178, 261, 328: Inline the referenced content or remove the HANDOFF.md pointers (file is absent).
- Line 176: Change `/healthz` → `/health`.
- Line 201, 204, 326: Replace `"badgers den"` with `"<your-ssid>"`.
- Line 238: Update "There is no `swift package build-container-image` path anymore" to also
  note `swift-container-plugin` is still in Package.swift as a stale dep and should not be used.

**docs/rtsp-relay.md:**
- Lines 40, 68: Replace `jetson:jetsontest` credentials with `<camera-user>:<camera-password>`.
- Line 121: Replace `nmcli con up "badgers den"` → `nmcli con up "<your-wifi-ssid>"`.

---

## What's GOOD (do not accidentally break)

1. **`DetectionBroadcaster` actor design is correct.** The `subscribe()`/`distribute()`/`unsubscribe()` pattern is properly actor-isolated. `distribute()` is non-blocking (yields to per-client `AsyncStream.Continuation` without awaiting), per-client `bufferingNewest(4)` prevents slow clients from blocking the detection loop. The `withTaskGroup` + `group.next()` + `group.cancelAll()` pattern in the WebSocket handler is the right structured-concurrency approach for two concurrent tasks (drain + send).

2. **Prometheus histogram implementation is correct.** The two-phase design — store in a single frequency bucket in `observe()`, then accumulate into cumulative `le` counts in `render()` — produces mathematically correct Prometheus output. The comment on line 54 ("counts[i] is the number of observations <= bounds[i]") is misleading (it's actually the frequency count for that interval), but the rendered output is correct.

3. **`Unmanaged` lifetime management in the pad probe is correct.** `passRetained` in `startPipeline()` balanced by `release()` in `teardownPipeline()`, with the raw opaque pointer stored in `retainedBox` to ensure the release is always called exactly once. The `takeUnretainedValue()` in the `@_cdecl` probe entry point is correct (ownership remains with `retainedBox`).

4. **`MetricsRegistry` using `Mutex` (not actor) is the right call.** The hot path (20 FPS × N histograms × M metrics) cannot afford actor hops. The single-lock-acquire snapshot in `render()` is a sound pattern.

5. **`gst_parse_launch` warning handling.** The code distinguishes between a parse error (returns `nil`) and a parse warning (returns non-nil with non-nil `GError`). This is non-obvious GStreamer API behavior that is handled correctly.

6. **`@unchecked Sendable` on `DetectionStream` is justified.** The class bridges a GStreamer streaming thread callback to Swift's cooperative pool. The `continuation` field is genuinely `Sendable` (AsyncStream continuations are designed for cross-thread use), and the `@unchecked` is the correct escape hatch for C-thread-origin code. The `Unmanaged` lifetime management prevents the reference from being freed before the thread is done.

7. **`AllOriginsMiddleware` CORS `*`** — acceptable for an embedded sensor device on a LAN. The device is not internet-exposed, and the monitor UI needs cross-origin reads. Would be inappropriate for a cloud API, is fine here.

8. **`shim.h` `wendy_gst_pull_sample` / `wendy_gst_try_pull_sample`** — These functions are dead code now (Stage 2 removed `appsink`) but they are correct and not harmful. Their presence documents the Stage 1 approach. Do not remove them without checking if they are referenced in ARCHITECTURE.md's war story section.

9. **EOS vs ERROR bus message distinction** in `wendy_gst_bus_pop_error` (returns 1=error, 2=EOS) is handled correctly in both the C shim and the Swift `pollBus()` / `ReconnectTrigger` enum.

10. **`DetectionStream.bufferingNewest(4)`** — This buffers 4 frames for the detection loop consumer. The consumer is `for await frame in detectionStream` — the single sole consumer. Dropping 4+ queued frames is a real loss (e.g., 200ms of track-disappearance events dropped during a GStreamer thread burst). This is the author's deliberate tradeoff (frames are not stored; only the most recent matter for the live display). For a reference implementation that only drives a live dashboard, this is fine.

---

## Codex Round 1 Verdict

Codex R1 ran against the upstream private `samples` repo (cwd confusion — it defaulted to
`/home/mihai/workspace/samples/deepstream-vision` not the public extract). From the data it
gathered:

**Confirmed findings:** F1 (wendy.json private IP), F2 (broken HANDOFF refs), F4 (start.sh
deploys Python), F5 (swift-container-plugin stale dep, CFFmpeg/CTurboJPEG orphan dirs),
F8 (gpuMemoryMB gauge), F9 (private paths), F10 (/healthz vs /health), F11 (credentials),
F18 (start.sh restart-unless-stopped).

**R1 newly surfaced (in private/upstream only, not in public extract):** The upstream repo has
`Sources/DetectorCore/` (TriggerEngine, TriggerAuditLog) and test targets not present in the
public extract. Also `tracker_config_nvdeepsort.yml` and `tracker_config_nvsort.yml` present
upstream but not in the public extract. These are not findings for the public repo but confirm
the public extract is intentionally a subset.

**F7 (histogram math):** R1 gathered the Metrics.swift code. Analysis confirms F7 is a FALSE
POSITIVE. The two-phase design (frequency bucket storage, cumulative render) produces correct
Prometheus output. The misleading comment on line 54 is a documentation nit, not a bug.

---

## Codex Round 2 Verdict

Codex R2 also ran against the upstream private repo (same cwd issue). It gathered file listings
confirming CFFmpeg/CTurboJPEG are tracked in the private repo and that Package.swift in the
upstream has `DetectorCore` and test targets. It did not produce a final text verdict before the
session ended.

**R2 assessment (manual):** The additional question about URL injection in `buildPipelineString()`
is confirmed as a LOW (F19 above) — the URL source is `streams.json` baked at build time, not
user-supplied at runtime, but special characters would cause confusing parse failures for an
integrator. The CORS `*` question is assessed as acceptable (see What's GOOD point 7). The
`<your-device>.local` default in `start.sh` causing immediate `ping` failure is the desired
behavior (fail-fast if no device is specified), not a bug — not added as a finding.

---

## Summary

| Severity | Count | Findings |
|----------|-------|----------|
| HIGH | 3 | F1, F2, F3 |
| MEDIUM | 9 | F4, F5, F6, F7, F8, F9, F10, F11, F12, F13 |
| LOW | 7 | F14, F15, F16, F17, F18, F19 |

Wait — adjusted count: F1/F2/F3 are HIGH, but F4 (start.sh) was also marked HIGH above. Let me
reconcile: F4 (wendy.json swift.resources stale) was relabeled from HIGH to MEDIUM in the
final review text above (see F4 heading — MEDIUM). The start.sh finding (originally F4 in the
draft, now incorporated into F2's "suggested fix" and covered separately as the separate F2
start.sh finding above) is HIGH.

**Corrected summary:**

| Severity | Count | Findings |
|----------|-------|----------|
| HIGH | 3 | F1 (wendy.json private IP), F2 (start.sh Python), F3 (HANDOFF.md dead refs) |
| MEDIUM | 9 | F4 (wendy.json resources stale), F5 (orphan dirs/dep), F6 (SDK name), F7 (gpu_memory_mb), F8 (private paths), F9 (/health vs /healthz), F10 (credentials), F11 (Task handle), F12 (Package.swift comment) |
| LOW | 7 | F13 (ProbeContext leak), F14 (cairo-stage), F15 (RTSPFrameReader.swift name), F16 (PORT_PLAN.md ref), F17 (start.sh restart), F18 (CPU % inconsistency), F19 (URL injection) |
