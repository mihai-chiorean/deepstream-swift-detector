# Port Plan: Swift detector → DeepStream-native pipeline

Status: **in progress**.
Reason: the current `appsink + gst_buffer_map` path leaks NVMM surfaces at
45–80 MB/s on JetPack 6 / L4T r36 because Swift has no CUDA context in-process,
and `gst_buffer_unmap` at the GStreamer layer does not propagate down to
`NvBufSurfaceUnMap` for DeepStream pool buffers. See inline comment in
`Sources/Detector/GStreamerFrameReader.swift` (`buildPipelineString`) for
the full history. Python + DeepStream does not hit this because it uses
`nvinfer` in-pipeline and never pulls RGB frames to CPU.

## Target architecture

Mirror Python's pipeline:

```
rtspsrc → h264parse → nvv4l2decoder → nvstreammux → nvinfer → nvtracker → fakesink
                                                                   │
                                                                   └─ pad-probe on src pad
                                                                         reads NvDsBatchMeta
                                                                         attaches no CPU mapping
                                                                         hands detections to Swift
```

MJPEG sidecar (when a stream client is connected):
```
tee → nvvideoconvert → nvosd (overlays from in-pipeline meta) → nvjpegenc → appsink
```
`nvjpegenc` emits system-memory JPEGs (~20 KB) — safe to `gst_buffer_map`.

## What's done

- [x] Research (gpu-engineer + embedded-linux + swift-backend) — confirmed this
      is ~weekend scope, not a rewrite. Findings folded into comments.
- [x] Stable baseline via software decode (`avdec_h264`) committed and running.
      Zero memory leak; ~400 MB + 1.5 GB TensorRT workspace.
- [x] `nvinfer_config.txt` — YOLO26n end-to-end, cluster-mode=4 (no NMS,
      the one-to-one head already does it), custom parser reference.
- [x] `Sources/CNvdsParser/nvdsparsebbox_yolo26.cpp` — ~120 lines C++,
      reads the `[1, 300, 6]` output tensor and emits
      `NvDsInferParseObjectInfo` list. No NMS, no anchor math.
- [x] `Sources/CNvdsParser/Makefile` — build recipe (needs DeepStream
      headers at `$DEEPSTREAM_INCLUDES`).
- [x] `tracker_config.yml` — NvDCF, copied from the Python detector.

## What's left

- [ ] **Build `libnvdsparsebbox_yolo26.so`** and drop it at
      `/app/lib/libnvdsparsebbox_yolo26.so` inside the container. Options:
        1. Container build step in `Dockerfile` using `dustynv/l4t-deepstream`
           builder image (simplest).
        2. GitHub Actions job cross-building against JetPack SDK.
        3. On-device build (one-shot during initial setup).
- [ ] **C shim for DeepStream metadata** — extend
      `Sources/CGStreamer/shim.h` (or new module `CDeepstream`) with:
        * `wendy_install_detection_probe(GstElement*, callback, user_data)`
        * `wendy_nvds_flatten(NvDsBatchMeta*, WendyDetection*, int max)`
          returning count — hides the GList walk behind a flat-struct API.
        * `WendyDetection` struct: `{ int class_id; float conf; float x, y,
          width, height; uint64_t track_id; int frame_num; }`.
- [ ] **New `GStreamerFrameReader`** — `AsyncStream<[Detection]>` backed by a
      pad probe instead of an appsink pull loop. The probe runs on the
      GStreamer streaming thread; it copies metadata to a Swift-owned
      `Sendable` array then `Task.detached { await actor.ingest(...) }`.
- [ ] **Gut outdated Swift code**:
        * Delete `Sources/Detector/DetectorEngine.swift` (TensorRT binding)
        * Delete `Sources/Detector/YOLOPreprocessor.swift`,
          `YOLOPostprocessor.swift` — nvinfer does all of this.
        * Delete `Sources/Detector/Tracker.swift`,
          `KalmanFilter2D.swift` — nvtracker replaces.
        * `Package.swift`: drop TensorRT dependency, drop CUDA framework
          dependency; add DeepStream include path.
- [ ] **MJPEG sidecar pipeline** — second pipeline launched only when a
      client is actually connected (driven by `DetectorState.shouldExtract
      Frames`). Encodes via `nvjpegenc` so the map-back into Swift is
      system-memory JPEGs, not NVMM RGB.
- [ ] **VLM crop path** — the one place we still need pixel access. Use
      the Python approach: `NvBufSurfaceMap` explicitly, `memcpy` the crop
      region, `NvBufSurfaceUnMap` *immediately*. Bypass `gst_buffer_map`
      entirely for this.

## Notes

- We keep `TrackFinalizer`, `VLMClient`, `HTTPServer`, `Metrics`,
  `FrameRenderer` unchanged. They already consume `Detection` + frame
  bytes at the API boundary.
- The `Detection` Swift struct shape stays the same; only the *source*
  changes (nvtracker metadata vs. our Swift IOU tracker).
- `tracker_config.yml` uses NvDCF with visual similarity — we get richer
  tracking than our old IOUTracker + Kalman essentially for free.
- `nvjpegenc` does JPEG encoding on the VIC hardware, so our `FrameRenderer`
  software JPEG path is no longer needed. Saves CPU too.

## Verification protocol (after each milestone)

1. Deploy single-shot (no `--restart-unless-stopped`) to the Jetson.
2. Sample `/proc/$PID/status` VmRSS every 15 s for 5 minutes.
3. **Pass criterion**: RSS plateaus under 2 GB. No cgroup OOM, no global OOM.
4. Check `/sys/kernel/debug/nvmap/iovmm/allocations` growth — should be flat.
5. Confirm detections flowing via `/detector/metrics` — `deepstream_frames_processed_total` incrementing.

## Reference

- Python implementation: `../detector/detector.py` — especially `osd_sink_pad_buffer_probe` at line 768, `get_frame_data_from_buffer` at line 695 (the critical "you MUST call unmap_nvds_buf_surface" comment).
- `Sources/CGStreamer/shim.h` — existing C shim pattern.
- `Sources/Detector/GStreamerFrameReader.swift` — existing GStreamer wrapper; the pad-probe variant will live here (or alongside).
