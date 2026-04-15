# Port Plan: Swift detector → DeepStream-native pipeline

Status: **stable software-decode baseline running. Port is next milestone.**
Designed so a fresh session can pick this up without needing to re-run the
research that produced it.

## Why this port exists

The `appsink + gst_buffer_map` path leaks NVMM surfaces at 45–80 MB/s on
JetPack 6 / L4T r36 regardless of converter plugin or properties tried:
default `nvvideoconvert`, `nvvidconv`, `compute-hw=1`, `copy-hw=2`,
`num-extra-surfaces=1`, NV12 intermediate, `queue leaky=downstream` — all
leaked.

Root cause: pulling frames via `gst_buffer_map(READ)` on NVMM buffers pins
`NvBufSurface` pool entries per mapped frame when no CUDA context is present
in the consumer process, and `gst_buffer_unmap` at the GStreamer layer does
not propagate down to `NvBufSurfaceUnMap` for DeepStream pool buffers. The
Python detector documents this in a comment near line 701 of `detector/
detector.py`: *"CRITICAL FOR JETSON: You MUST call unmap_nvds_buf_surface()
after get_nvds_buf_surface(). Failure causes severe memory leaks (memory
grows to 100%+ within minutes)."*

Python's DeepStream pipeline avoids all of this by staying in NVMM
end-to-end: `nvinfer` runs TensorRT in-pipeline on NVMM, attaches detection
results as GstMeta, and the app reads them via a pad probe. No `gst_buffer
_map`, no leak.

## Target architecture (mirrors Python's)

```
Main pipeline (detections only, no frame pulls):
    rtspsrc → rtph264depay → h264parse
      → nvv4l2decoder
      → nvstreammux (batch-size=1, required by nvinfer)
      → nvinfer    (loads yolo26n_b2_fp16.engine via config file)
      → nvtracker  (NvDCF, config via tracker_config.yml)
      → fakesink

    Pad probe on nvtracker src pad:
      - gst_buffer_get_nvds_batch_meta(buf) — returns NvDsBatchMeta*
      - walk frame_meta_list → obj_meta_list, each NvDsObjectMeta has:
          .classId, .confidence, .rect_params (left/top/width/height in
          source-frame pixel space), .object_id (from nvtracker)
      - flatten to Swift Detection[], hand off via Unmanaged + Task.detached

MJPEG sidecar (only when an MJPEG client is connected):
    tee off the decoded NVMM buffer →
      nvvideoconvert → nvosd (in-pipeline overlay from batch meta) →
      nvjpegenc → appsink
    nvjpegenc output is system-memory JPEG (~20 KB/frame); gst_buffer_map
    on system memory is free, no NVMM pinning.

VLM crop path (on track finalization, occasional):
    Explicit NvBufSurfaceMap / memcpy / NvBufSurfaceUnMap on the track's
    best frame. Mirrors Python's approach — map only when absolutely
    needed, unmap immediately.
```

## What's done in this repo

- [x] Stable baseline: `Sources/Detector/GStreamerFrameReader.swift`
      pipeline is `avdec_h264` software decode, deployed with auto-restart.
      Zero memory leak; verified flat at ~400 MB over 9+ min.
- [x] Agent scratch-buffer + 8 fps MJPEG throttle retained in `Detector
      .swift` / `FrameRenderer.swift`.
- [x] Research confirmations (summarized in this file; no other artifacts
      needed).
- [x] `nvinfer_config.txt` — configured for YOLO26n, `cluster-mode=4`
      (no clustering, because the one-to-one head already emits final
      detections), `maintain-aspect-ratio=1 symmetric-padding=1` (matches
      how the engine was trained).
- [x] `tracker_config.yml` — copy of Python detector's NvDCF config.
- [x] `Sources/CNvdsParser/nvdsparsebbox_yolo26.cpp` — custom bbox parser
      (~120 lines) for the YOLO26 `[1, 300, 6]` output. Uses `detection
      Params.perClassPreclusterThreshold` for confidence gate, skips
      pad slots with `class_id < 0` or `x2 ≤ x1`.
- [x] `Sources/CNvdsParser/Makefile` — build recipe.

## Tasks left (in execution order)

### 1. Build `libnvdsparsebbox_yolo26.so`

**Decision**: container build using the `dustynv/l4t-deepstream:r36.4.0`
image, copy the resulting `.so` into the detector container's `/app/lib/`.

- Add a build stage to `Dockerfile.bak` (rename/revive):
  ```Dockerfile
  FROM dustynv/l4t-deepstream:r36.4.0 AS nvparser
  COPY Sources/CNvdsParser /src
  WORKDIR /src
  RUN make DEEPSTREAM_INCLUDES=/opt/nvidia/deepstream/deepstream/sources/includes
  ```
- Copy `libnvdsparsebbox_yolo26.so` into the runtime image at
  `/app/lib/libnvdsparsebbox_yolo26.so` — the path referenced in
  `nvinfer_config.txt` line `custom-lib-path`.
- Alternative: one-shot build on the Jetson itself and mount the .so as a
  resource via `wendy run ... --resources libnvdsparsebbox_yolo26.so:/app/
  lib/libnvdsparsebbox_yolo26.so` (current `wendy run` in
  `project_current_state.md` already uses `--resources` for the engine).

### 2. Verify the pipeline runs outside Swift (smoke test)

Before any Swift code changes, verify nvinfer + the custom parser on the
device with `gst-launch-1.0`:

```bash
ssh root@10.42.0.2 'GST_DEBUG=3 LD_LIBRARY_PATH=/app/lib \
  gst-launch-1.0 -v \
    rtspsrc location=rtsp://jetson:jetsontest@192.168.68.69:554/stream1 \
      latency=200 protocols=tcp \
    ! rtph264depay ! h264parse \
    ! nvv4l2decoder \
    ! m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 \
    ! nvinfer config-file-path=/app/nvinfer_config.txt \
    ! nvtracker ll-config-file=/app/tracker_config.yml \
        ll-lib-file=/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so \
    ! fakesink'
```

If this runs without crash and emits `GST_DEBUG=3` lines from nvinfer, the
engine + custom parser + tracker load correctly. Measure VmRSS — it should
be flat (no `gst_buffer_map` anywhere, so no leak even from this test).

### 3. C shim for pad probe + metadata extraction

Extend `Sources/CGStreamer/` (not a new module — keep one place for all
C interop). Add a sibling header or new file.

**Headers needed at build time** (from DeepStream SDK):
  - `nvdsmeta.h`
  - `gstnvdsmeta.h`

Both at `/opt/nvidia/deepstream/deepstream/sources/includes` in the
`dustynv/l4t-deepstream` image. Options:
  - Copy them into `Sources/CGStreamer/vendor/` (simple, self-contained).
  - Or pass via `-Xcc -I/path` in `Package.swift` unsafeFlags and keep
    the host toolchain wired to find them (fragile for cross-compile).

**The flat C struct + API** (add to `shim.h` or new `nvds_shim.h`):

```c
typedef struct {
    int      classId;
    float    confidence;
    float    left;          // source-frame pixel space
    float    top;
    float    width;
    float    height;
    uint64_t trackerId;     // 0 if not yet tracker-confirmed
    int      frameNum;
} WendyDetection;

// Walk the NvDsBatchMeta linked lists and fill out[]. Returns count.
// Never calls gst_buffer_map. Caller provides a stack buffer sized
// to the expected max (300 for YOLO26, so 16 KB is plenty).
int wendy_nvds_flatten(GstBuffer *buf,
                       WendyDetection *out,
                       int maxOut);

// Install a pad probe that calls `callback(count, ptr, userData)` each
// time the pipeline emits a buffer. The callback runs on the GStreamer
// streaming thread — it MUST NOT touch Swift actor state directly;
// copy into a Sendable value and bounce via Task.detached.
typedef void (*WendyDetectionCallback)(int count,
                                       const WendyDetection *dets,
                                       void *userData);

gulong wendy_install_detection_probe(GstElement *element,
                                     const char *padName,     // "src"
                                     WendyDetectionCallback cb,
                                     void *userData);
```

Internally the shim does GList walks so Swift never touches `GList`:

```c
// in the shim .c
#include <nvdsmeta.h>
#include <gstnvdsmeta.h>

int wendy_nvds_flatten(GstBuffer *buf, WendyDetection *out, int maxOut) {
    NvDsBatchMeta *bm = gst_buffer_get_nvds_batch_meta(buf);
    if (!bm) return 0;
    int count = 0;
    for (NvDsMetaList *l = bm->frame_meta_list; l && count < maxOut; l = l->next) {
        NvDsFrameMeta *fm = (NvDsFrameMeta *)l->data;
        for (NvDsMetaList *o = fm->obj_meta_list; o && count < maxOut; o = o->next) {
            NvDsObjectMeta *om = (NvDsObjectMeta *)o->data;
            out[count].classId    = om->class_id;
            out[count].confidence = om->confidence;
            out[count].left       = om->rect_params.left;
            out[count].top        = om->rect_params.top;
            out[count].width      = om->rect_params.width;
            out[count].height     = om->rect_params.height;
            out[count].trackerId  = om->object_id;
            out[count].frameNum   = fm->frame_num;
            count++;
        }
    }
    return count;
}
```

### 4. New `GStreamerFrameReader` (pad-probe-driven)

Replace `AsyncStream<Frame>` with `AsyncStream<[Detection]>`. The probe
callback is `@convention(c)` — cannot capture a Swift closure.

Pattern (from agent research):

```swift
final class DetectionStream {
    private let continuation: AsyncStream<[Detection]>.Continuation
    let stream: AsyncStream<[Detection]>

    init() {
        var c: AsyncStream<[Detection]>.Continuation!
        self.stream = AsyncStream { c = $0 }
        self.continuation = c
    }

    func install(on element: OpaquePointer) {
        let box = Unmanaged.passRetained(self).toOpaque()
        wendy_install_detection_probe(element, "src", probeCallback, box)
    }

    func ingest(_ detections: [Detection]) {
        continuation.yield(detections)
    }
}

@_cdecl("wendy_swift_detection_callback_entry")
func probeCallback(count: Int32,
                   dets: UnsafePointer<WendyDetection>?,
                   userData: UnsafeMutableRawPointer?) {
    guard let dets, let userData, count > 0 else { return }
    let ds = Unmanaged<DetectionStream>.fromOpaque(userData).takeUnretainedValue()
    // Copy into Sendable Swift struct array — don't leak C pointers across threads.
    var copy = [Detection]()
    copy.reserveCapacity(Int(count))
    for i in 0..<Int(count) {
        let d = dets[i]
        copy.append(Detection(classId: Int(d.classId),
                              confidence: d.confidence,
                              x: d.left, y: d.top, w: d.width, h: d.height,
                              trackId: d.trackerId == 0 ? nil : Int(d.trackerId)))
    }
    ds.ingest(copy)  // actor-isolated — fine because ingest is an actor method
}
```

The probe callback is fire-and-forget at ~20 Hz — one `continuation.yield`
per frame.

### 5. Gut outdated Swift code

Delete:
- `Sources/Detector/DetectorEngine.swift` — TensorRT binding.
- `Sources/Detector/YOLOPreprocessor.swift` — letterbox / normalize /
  HWC→CHW. `nvinfer` does this now via config.
- `Sources/Detector/YOLOPostprocessor.swift` — sigmoid / confidence filter.
  Our custom parser does the filter; `cluster-mode=4` skips NMS.
- `Sources/Detector/Tracker.swift` + `KalmanFilter2D.swift` — `nvtracker`
  replaces with NvDCF. `trackerId` arrives attached to each `Detection`.

`Package.swift`:
- Remove TensorRT / CUDA linker settings.
- Add `-Xcc -I` to DeepStream headers (or point at vendored copy).
- Add `.linkedLibrary("nvdsgst_meta")` and `.linkedLibrary("nvds_meta")`.

### 6. Update `Detector.swift`

`runStreamDetectionLoop` becomes an `AsyncStream<[Detection]>` consumer.
No TensorRT invocation, no `FrameRenderer.renderFrame`, no tracker update —
all of that was either deleted or moved in-pipeline.

What remains in the loop:
- Log fps via `Metrics.fps.set(...)`.
- Update `deepstream_frames_processed_total`, detection counters.
- Drive `TrackFinalizer.submit(...)` when a tracker ID disappears
  (detect the disappearance by diffing current IDs against the last
  seen set — `nvtracker` doesn't emit an explicit "track ended" event,
  so this is a Swift-side bookkeeping pass).

### 7. MJPEG sidecar pipeline

Second pipeline launched only when `state.shouldExtractFrames == true`.
Teed off the *same* decoded stream (requires a `tee` element before
`nvstreammux`, and a queue before each branch):

```
rtspsrc → rtph264depay → h264parse → nvv4l2decoder → tee name=t
  t. → queue ! nvstreammux ! nvinfer ! nvtracker ! fakesink  (detection)
  t. → queue ! nvvideoconvert ! nvosd ! nvjpegenc ! appsink   (MJPEG)
```

The MJPEG appsink pulls JPEG-encoded `GstBuffer`s in system memory.
`gst_buffer_map` on these is cheap (malloc'd pages, not NVMM pool).
Rate-limit to 5 fps via `videorate` or appsink buffer caps.

nvosd reads the same `NvDsBatchMeta` and draws overlays directly — no
Swift-side rendering. We can delete `FrameRenderer.drawDetections` too.

### 8. VLM crop path

`TrackFinalizer` currently receives a `Frame` (full RGB bytes). That pixel
access goes away in the new design — until VLM actually needs the crop.

Two options:
- (a) **Simpler**: `TrackFinalizer` accepts only `Detection` metadata,
  and a periodic pad probe samples *one* frame per finalized track via
  explicit `NvBufSurfaceMap` (following Python's pattern at
  `detector.py:695` — the `get_frame_data_from_buffer` helper). This is
  the minimum NVMM mapping we can do.
- (b) **Simplest for now**: keep VLM disabled; ship NVDEC-without-VLM,
  add VLM back once (a) is implemented.

## Verification protocol

Each milestone:
1. Deploy single-shot to the Jetson (no `--restart-unless-stopped` — we
   don't want an auto-restart loop if the new code misbehaves).
2. Apply cgroup cap (`ssh root@10.42.0.2 'CONTAINER=detector-swift
   /usr/local/bin/detector-cap'`) — 5 GiB default is fine.
3. Sample `/proc/$PID/status` VmRSS every 15 s for 5 minutes.
4. **Pass criterion**: RSS plateaus under **2 GB** (Swift runtime + nvinfer
   internals + YOLO26 engine context ≈ 1.5 GB). No cgroup OOM, no global
   OOM, no "Detection loop ended" in detector log.
5. **Independent check** (the leak canary): monitor
   `/sys/kernel/debug/nvmap/iovmm/allocations` size on the Jetson. Should
   be flat after warmup. **Previously it grew 65 MB → 3,176 MB in 30 s on
   the leaking pipeline** — anything similar means the pad probe is
   accidentally touching NVMM.
6. Confirm detection flow: `curl http://10.42.0.2:9090/metrics` shows
   `deepstream_frames_processed_total` incrementing and
   `deepstream_detections_total{class_=...}` growing.

## Operational notes (session gotchas to preserve)

- **`wendy run` does NOT rebuild Swift code.** You MUST `swift build` +
  `swift package build-container-image` + push BEFORE `wendy run`. `wendy
  run` only creates and starts a task from the existing image.
- **Always apply the cgroup cap** immediately after `wendy run`. The
  inherited `oom_score_adj=-998` from containerd means OOM kills
  networking instead of the detector, locking up the Jetson.
- **Do not use `--restart-unless-stopped`** with an unstable build — a
  crashing container will be restarted so fast the NVDEC driver state
  can't recover, CPU pegs, SSH banner times out, and you're forced to
  power-cycle the Jetson. (Happened twice in the debug session this file
  commits against.)
- **SSH to device via USB-C**: `ssh -o StrictHostKeyChecking=no -o
  UserKnownHostsFile=/dev/null -o LogLevel=ERROR root@10.42.0.2`. USB-C
  is 10.42.0.1 ↔ 10.42.0.2. WiFi path is 192.168.68.70 but unreliable.
- **WendyOS Jetson WiFi tends to not reconnect after reboot.** After any
  `sudo reboot`, run `ssh root@10.42.0.2 'nmcli con up "badgers den"'` to
  bring WiFi back (required for RTSP camera at 192.168.68.69).
- **VLM (llama-server)** runs in Docker under the name `llama-vlm`.
  `docker stop llama-vlm` to keep it down during detector memory tests —
  it holds ~2 GB unified memory which steals room from the detector.
- **Registry** at `10.42.0.2:5000` lives in a containerd task. If `swift
  package build-container-image` fails with `Failed to connect ... port
  5000`, restart it:
  `ssh root@10.42.0.2 'ctr -n default containers ls | grep registry'`
  then `ctr -n default tasks start -d <container-id>`.

## File map (what you'll touch)

| File | Action |
|---|---|
| `nvinfer_config.txt` | already written; review batch-size vs. pipeline |
| `tracker_config.yml` | already written; leave as-is |
| `Sources/CNvdsParser/nvdsparsebbox_yolo26.cpp` | already written |
| `Sources/CNvdsParser/Makefile` | already written |
| `Sources/CGStreamer/shim.h` + new `nvds_shim.h` | add probe install + flatten |
| `Sources/CGStreamer/module.modulemap` | add DeepStream header include + link libs |
| `Sources/Detector/GStreamerFrameReader.swift` | rewrite pipeline + reader |
| `Sources/Detector/Detector.swift` | simplify loop (no inference call) |
| `Sources/Detector/DetectorEngine.swift` | **delete** |
| `Sources/Detector/YOLOPreprocessor.swift` | **delete** |
| `Sources/Detector/YOLOPostprocessor.swift` | **delete** |
| `Sources/Detector/Tracker.swift` + `KalmanFilter2D.swift` | **delete** |
| `Sources/Detector/FrameRenderer.swift` | reduce to JPEG-only helper, or delete if using nvosd+nvjpegenc |
| `Sources/Detector/TrackFinalizer.swift` | strip frame-bytes path; either pad-probe-map or disable VLM |
| `Package.swift` | drop TensorRT, add `-Xcc -I` and DS link libs |
| `Dockerfile` (revive from `Dockerfile.bak`) | add parser build stage |
| `wendy.json` | verify `--resources` list covers new files |

## What to do first in a fresh session

1. Open `PORT_PLAN.md`, read the "Why this port exists" + "Target architecture" sections.
2. Execute **Task 1**: build `libnvdsparsebbox_yolo26.so` in a container.
3. Execute **Task 2**: run the `gst-launch-1.0` smoke test. This is the single
   highest-leverage moment — it confirms the engine + parser + tracker
   chain works *before* you touch any Swift code. If this crashes, fix it
   on the command line, not in Swift.
4. Only then start the C shim + Swift refactor. Those are straightforward
   once the pipeline runs.
