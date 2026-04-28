# REVIEW-GPU.md — DeepStream Swift Detector (public extract)

**Reviewer:** GPU / NVIDIA edge-inference lens
**Repo:** `github.com/mihai-chiorean/deepstream-swift-detector`
**Commit reviewed:** `07b32c4` (last public push 2026-04-20)
**Date:** 2026-04-23

---

## TL;DR

The detection plumbing is mostly sound (pad probe on `nvtracker.src` reading `NvDsBatchMeta` via a C shim, no `gst_buffer_map`, custom YOLO26 [1,300,6] parser with sane bounds), but several **HIGH** issues need fixing before public eyes land on this:

- **Engine artifact contract is broken.** Config + Dockerfile reference `yolo26n_b2_fp16.engine` but only the ONNX is in the repo, no `onnx-file=` fallback, no documented `trtexec` build command, and the `b2` filename hints at batch-size-2 against `batch-size=1` in config. Reproduction story doesn't work.
- **`USE_NEW_NVSTREAMMUX` is not pinned.** Pipeline depends on legacy mux behaviour but the env that selects it is unset; CDI runtime could flip it.
- **Camera RTSP credentials `jetson:jetsontest` in plain text** in `docs/rtsp-relay.md`.
- **Pad-probe lifecycle has unproven concurrency invariants** (`g_free`-via-`GDestroyNotify` + `Unmanaged.release()` happen together but neither is shown to drain in-flight callbacks).
- **Failed probe install is non-fatal** — the detector can transition to PLAYING with no detections and no startup failure.

Plus public-readiness wince: a 51 MB statically-linked aarch64 `ffmpeg` binary, a 4.9 MB cairo-stage of dead vendored x11 `.so`s, an empty `vendor/` `-isystem` path that breaks first-clone build, and a `192.168.68.70:5000` private-registry reference.

After two codex rounds: **2 net-new HIGH (engine artifact contract, streammux v1/v2)** plus **1 net-new MEDIUM (parser tensor-ABI validation)** were added in R1; R2 caught a polarity bug in my Finding 10 (filter list) that is a publication blocker, a wrong premise in Finding 16 (`fakesink` default), and a missing tracker-width/height regression vs the Python sibling. R2 verdict: **R3 needed before publication; major findings are right but specific fixes/wording must be corrected.**

**R3 (publication-readiness pass, applied inline below):**
- Severity demotions for the public-repo / showcase context (not a deployed production binary): Findings 3, 4, 5, 6, 8 demoted from HIGH → MEDIUM. The reproducer story is gated by Findings 1, 2, 7, not by these. Finding 27 promoted LOW → MEDIUM (bbox correctness for non-16:9 sources).
- Factual fixes: Finding 13 "78-class list" → 68 filtered classes (8 IDs before 20, plus 20-79 inclusive); Finding 9 wording reworded so the `DetectionStream` retain is **not** leaked every cycle (Swift `release()`s `retainedBox` in teardown), and the `g_free` UAF is a *future-fix* hazard, not present behaviour (current code passes `NULL`); Finding 4 strong case is `ptsNs`/overlay alignment, not component-latency buckets (the C shim reads system timestamps).
- Net publication-blocker HIGH set after R3: **Findings 1 (creds), 2 (engine artifact contract), 7 (vendor `-isystem` empty dir).**

---

## Findings

### HIGH

**1. Camera RTSP credentials committed in plain text.**
**Why:** `docs/rtsp-relay.md:40` and `:68` show `rtsp://jetson:jetsontest@192.168.68.69:554/stream1`. Even if the camera is on a private LAN, the public repo now states the username, password, port, and that the device is a TP-LINK camera. Standard credential-rotation hygiene says: rotate this password and remove from history (or make this an obvious placeholder). The pre-cleanup-20260420 tag preserves the leak in history regardless.
**Suggested fix:** Rotate the camera password, then `git filter-repo` to redact, replace with `<camera-user>:<camera-pass>` placeholder. (file: `docs/rtsp-relay.md:40,68`)

**2. Engine artifact contract is broken: config + Dockerfile reference `yolo26n_b2_fp16.engine`; repo has only ONNX; no documented build command. (Added per codex R1.)**
**Why:** `nvinfer_config.txt:12` sets `model-engine-file=/app/yolo26n_b2_fp16.engine`; `Dockerfile:42` does `COPY yolo26n_b2_fp16.engine /app/...`. `.gitignore` excludes `*.engine` so the engine is not in the public repo. There is **no `onnx-file=` setting in `nvinfer_config.txt`**, so nvinfer cannot rebuild the engine on first run; it expects a pre-built plan. The `b2` filename (likely batch-size-2 build) vs `batch-size=1` in both nvinfer and nvstreammux configs is suggestive of stale-or-mismatched batch contract — not proof (a dynamic explicit-batch engine with maxBatch=2 may still accept batch=1), but worth verifying the engine's binding profile. TensorRT plans are version+device sensitive (JetPack r36.4.4 ships TRT 10.3); a plan built on a different Jetson generation or different TRT minor will not load.
**Suggested fix:** Either (a) add `onnx-file=/app/yolo26n.onnx` so nvinfer auto-builds, or (b) ship `docs/engine-build.md` with the exact `trtexec` invocation, rename the engine file to reflect its actual batch profile, and document the JetPack/TRT version it was built against. Verify the plan's binding shapes with `polygraphy inspect model yolo26n_b*_fp16.engine` and put the output in the doc. (Source: TensorRT compatibility docs.) (file: `detector-swift/nvinfer_config.txt:12`, `detector-swift/Dockerfile:42`, `detector-swift/yolo26n.onnx`)

**3. Streammux v1/v2 behavior is not pinned (`USE_NEW_NVSTREAMMUX`). (Added per codex R1; demoted HIGH → MEDIUM per R3 — environment-sensitive, but not broken by default unless the Wendy/CDI runtime exports `USE_NEW_NVSTREAMMUX=yes`.)**
**Why:** DS 7.1 ships both legacy `nvstreammux` and the new mux. Defaults flip when the env var `USE_NEW_NVSTREAMMUX=yes` is set. The pipeline depends on legacy properties: `width`, `height`, and `live-source` semantics are different (or unavailable) on the new mux. If the runtime environment flips, parsing may fail or — worse — silently apply different defaults (e.g. no scaling). `batched-push-timeout` does still exist on new mux, contrary to my R1 absorption — codex R2 corrected that detail; the property-set difference is real but partial. Nothing in the Swift code, Dockerfile, or runtime env pins this.
**Suggested fix:** Set `setenv("USE_NEW_NVSTREAMMUX", "no", 1)` in `setupPluginEnvironment()` alongside the other env vars there, and document. If there's a long-term plan to migrate to mux v2, add a TODO citing what changes (drop width/height, drop live-source, switch to batch-policy properties). (Source: nvstreammux2 docs.) (file: `detector-swift/Sources/Detector/GStreamerFrameReader.swift:634-651`)

**4. `nvstreammux` is missing `live-source=1` (and `batched-push-timeout`) for an RTSP source. (Demoted HIGH → MEDIUM per R3 — real but not clone-and-run blocking; the strong case is `ptsNs`/overlay alignment, not component-latency buckets.)**
**Why:** `GStreamerFrameReader.swift:624` builds `nvstreammux name=m batch-size=1 width=1920 height=1080`. With `rtspsrc latency=200 protocols=tcp` upstream, the source is live. With `live-source=0` (default), legacy mux **synthesizes the batched buffer's PTS from the negotiated source FPS** rather than copying input PTS through. With `live-source=1` it copies input PTS — material for the overlay-sync (`ptsNs` in JSON) story. The component-latency-meta story is **not** primarily affected: the C shim reads component system timestamps, not PTS. **At `batch-size=1` this is not a batch-fill liveness bug** (one frame is a full batch), so the cost is sub-frame timestamp correctness, not stalls.
**Suggested fix:** `nvstreammux name=m batch-size=1 width=1920 height=1080 live-source=1 batched-push-timeout=40000`. The `batched-push-timeout` is defensive against future multi-stream batching. (Source: NVIDIA `nvstreammux` docs.) (file: `detector-swift/Sources/Detector/GStreamerFrameReader.swift:618-631`)

**5. No `queue` elements as back-pressure boundaries in the pipeline. (Demoted HIGH → MEDIUM per R3 — production engineering call; absence does not usually break a reproducer.)**
**Why:** "Single thread" was the wrong framing — GStreamer does spawn task threads for sources, decoders, sinks, and async plugins (codex R1 corrected this). The real concern is back-pressure isolation: without a `queue` between mux and nvinfer (or decoder and mux), a stall in nvinfer propagates upstream to the decoder and rtspsrc jitter buffer, where buffer drops appear as stream stutters rather than deferred work. There is also no leaky-downstream queue protecting against transient inference slowdowns.
**Suggested fix:** Insert `queue max-size-buffers=4 max-size-bytes=0 max-size-time=0 leaky=downstream` between (i) `nvv4l2decoder` and `nvstreammux`, and (ii) `nvstreammux` and `nvinfer`. Downstream-leaky on the inference queue says "drop oldest if inference can't keep up" — appropriate for live detection. (Source: GStreamer threading docs.) (file: `detector-swift/Sources/Detector/GStreamerFrameReader.swift:618-631`)

**6. Failed probe install is logged but not fatal — detector can run with zero detections. (Added per codex R2; demoted HIGH → MEDIUM per R3 — robustness bug, not a clone-and-run blocker on the static-pad path.)**
**Why:** `GStreamerFrameReader.swift:526-528` logs "Failed to install detection probe on nvtracker src pad" when `id == 0`, then **still transitions to PLAYING** at line 530. The detector becomes a service that publishes nothing, with no startup failure to alert on. For a detection service, "I am running but I am blind" is the worst observable state — Prometheus shows `deepstream_fps=0` and a human has to notice.
**Suggested fix:** Throw `GStreamerError.probeInstallFailed("nvtracker.src")` from `startPipeline` if `wendy_install_detection_probe` returns 0, after releasing acquired resources (the retained box, the tracker ref, the parsed pipeline). The reconnect loop will then retry with backoff — appropriate behaviour. (file: `detector-swift/Sources/Detector/GStreamerFrameReader.swift:510-528`)

**7. `unsafeFlags(["-isystem", "Sources/CGStreamer/vendor"])` points at an empty directory; reproduction story is broken.**
**Why:** `Package.swift:44` adds `Sources/CGStreamer/vendor` as `-isystem`, but the directory contains only `.gitkeep` and a `README.md`. The header at `nvds_shim.c:13-15` says vendor is "populated by the Dockerfile builder stage" — but no Dockerfile in this repo populates it. A first-time reader following README "Build" steps cannot compile because `nvdsmeta.h` is not on the include path unless the WendyOS SDK happens to have it. The two `Package.swift` comment blocks (lines 16-32 vs 38-45) also contradict each other on whether SDK sysroot or vendor takes precedence.
**Suggested fix:** Either (a) check in the four required DeepStream headers under `vendor/` with an NVIDIA EULA note (verify license), or (b) delete the `unsafeFlags`, document the SDK-sysroot path explicitly in README, and add a one-script `scripts/fetch-headers.sh` that pulls them from a Jetson DeepStream install. (file: `detector-swift/Package.swift:33-57`)

**8. Custom YOLO26 parser silently accepts non-finite values. (Demoted HIGH → MEDIUM per R3 — good engineering, not publication-blocking for a showcase.)**
**Why:** `nvdsparsebbox_yolo26.cpp:74-98` reads `x1, y1, x2, y2, conf, cls` and gates on `conf < threshold` and `x2 <= x1 || y2 <= y1`. There is no check for `NaN` (which compares false to anything, so it slips past `conf < threshold`) or `Inf`. Pathological FP16 underflow at the head of an export can produce `NaN` confidence that propagates into `obj.detectionConfidence = NaN`, breaking Prometheus histogram bucketing. `cls = static_cast<int>(slot[5])` on NaN is C++ undefined behaviour.
**Suggested fix:** Add `if (!std::isfinite(conf) || !std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2) || !std::isfinite(slot[5])) continue;` at the top of the per-slot loop. (file: `detector-swift/Sources/CNvdsParser/nvdsparsebbox_yolo26.cpp:71-98`)

### MEDIUM

**9. Pad-probe lifecycle: `gst_pad_remove_probe` + `Unmanaged.release()` ordering relative to in-flight callbacks is unproven. (Added per codex R2; wording corrected per R3.)**
**Why:** `teardownPipeline()` at `GStreamerFrameReader.swift:554-581` does, in order: `gst_pad_remove_probe(pad, probeId)` then `Unmanaged.fromOpaque(box).release()`. GStreamer documents `gst_pad_remove_probe` as MT-safe, but the contract is that the probe will not fire **after** the call returns; it does not block waiting for a callback already mid-execution to drain. If the streaming thread is mid-callback when teardown runs (unlikely on a state change to NULL but possible during a graceful EOS-driven teardown), `wendyDetectionProbeEntry` calls `Unmanaged<DetectionStream>.fromOpaque(box).takeUnretainedValue()` — a use-after-free if `release()` has already run. **Future-fix hazard for Finding 10's `ProbeContext` leak**: if a future patch passes `g_free` as `GDestroyNotify`, `g_free(ctx)` will run on probe-remove; if the streaming thread is still in `probe_callback` reading `ctx->cb`, that's a UAF too. (Current code passes `NULL`, so no `g_free` race today — only the `Unmanaged.release()` race.) The "GOOD" section of this review previously called the retain-balance "correct"; that is too strong without proof.
**Suggested fix:** Two-phase teardown: (i) atomically swap a `cb` pointer in `ProbeContext` to NULL (signalling "drop callbacks"); (ii) call `gst_pad_remove_probe`; (iii) **block on a "no-active-callback" condition** (atomic counter or per-probe G_LOCK) before any future `g_free(ctx)` and the `Unmanaged.release()`. Alternative: keep leaking `ProbeContext` (sub-100 KB/day, see Finding 10) and introduce active-callback draining only when a confirmed UAF is observed. The `DetectionStream` retain is *not* leaked every cycle — Swift `release()`s `retainedBox` in teardown today; the question is whether that release races with an in-flight callback. Cite the GStreamer pad-probe MT semantics in the comment. (file: `detector-swift/Sources/Detector/GStreamerFrameReader.swift:554-581`, `detector-swift/Sources/CGStreamer/nvds_shim.c:217-237`)

**10. `ProbeContext` is leaked on every reconnect cycle (real, but small) — fix needs Finding 9 lifecycle guarantee.**
**Why:** `nvds_shim.c:217-233` `g_malloc0`s a `ProbeContext` and passes `NULL` as `GDestroyNotify`. Each reconnect leaks ~32 bytes. Codex R1 corrected my "megabytes per day" framing — at backoff-capped reconnect rates this is sub-100 KB/day. Real, bounded.
**Suggested fix:** Pass `g_free` as the GDestroyNotify in `gst_pad_add_probe(...)`, **but only after Finding 9's two-phase teardown is in place** — otherwise you trade a leak for a UAF. Document the ordering. (Source: GStreamer pad probe docs.) (file: `detector-swift/Sources/CGStreamer/nvds_shim.c:217-233`)

**11. Custom parser does not strictly validate tensor ABI (name, dtype, dim order, output count). (Added per codex R1; corrected priority + suggested fix per R2.)**
**Why:** `nvdsparsebbox_yolo26.cpp:43-67` only checks `outputLayersInfo.empty()` + `dataType == FLOAT` + `actual >= expected`. It does not check the layer name (a future model with a different output binding order silently picks the wrong tensor), it does not require `actual == expected` (a `[1,6,300]` transpose with the same element count would silently misread), and it does not check the binding count (an extra confidence head slips past). **Codex R2: this is MEDIUM, not HIGH** — only becomes HIGH if model/output churn is expected. The fix as I originally drafted (`numDims == 2 && d[0] == 300 && d[1] == 6`) is too brittle: DS may expose `[300,6]` (rank 2) or `[1,300,6]` (rank 3) depending on how `NvDsInferLayerInfo.inferDims` reports explicit batch — verify on target before encoding.
**Suggested fix:** On first frame: log `layer.layerName`, `layer.dataType`, `layer.inferDims.numDims`, and `layer.inferDims.d[0..numDims]`. Add a `static bool s_logged = false;` guard. After verifying on the actual deployed engine, encode the *observed* exact ABI: `outputLayersInfo.size() == 1`, `strcmp(layer.layerName, "<observed-name>") == 0`, `layer.dataType == FLOAT`, exact dim shape. Fail the parse loudly with a one-time logger if observed ≠ expected. (file: `detector-swift/Sources/CNvdsParser/nvdsparsebbox_yolo26.cpp:43-67`)

**12. `nvinfer_config.txt` has both `cluster-mode=4` and `topk=300`/`pre-cluster-threshold=0.45` — confusing, not "dead". (Corrected per codex R1.)**
**Why:** `topk` is consumed by `filterTopKOutputs` (a per-class cap), independent of clustering. With `cluster-mode=4` the clustering pass is skipped, but `topk` still acts as a cap on emitted detections. `topk=300` is a no-op only because the model emits exactly 300 slots and the parser yields ≤300. `pre-cluster-threshold` is read by my custom parser via `detectionParams.perClassPreclusterThreshold[cls]`, so it's also live, just consumed elsewhere.
**Suggested fix:** Add a comment block to `[class-attrs-all]`: `# topk caps emitted detections per class (filterTopKOutputs, not a clustering knob); 300 = engine slot count, effectively no-op. # pre-cluster-threshold is consumed by the custom parser via detectionParams.perClassPreclusterThreshold[cls], not by nvinfer's clustering — see nvdsparsebbox_yolo26.cpp.` (file: `detector-swift/nvinfer_config.txt:30,47-50`)

**13. `filter-out-class-ids`: hardcoded inline 68-class suppress list. CRITICAL polarity correction per codex R2; class count corrected per R3.**
**Why:** `GStreamerFrameReader.swift:626` lists 68 classes to *suppress* (8 explicit IDs in `4;6;8;9;10;11;12;13;` plus 20-79 inclusive = 60 more). **Current behaviour KEEPS the COMPLEMENT: 12 classes — `0,1,2,3,5,7,14,15,16,17,18,19`** — i.e., person/bicycle/car/motorcycle/bus/truck/bird/cat/dog/horse/sheep/cow. **My R1 absorption text said "current behaviour keeps 9, 10, 11" — that was wrong; 9, 10, 11 (traffic light, fire hydrant, stop sign) are FILTERED OUT.** This is the kind of bug this finding was warning about, and I almost shipped it. The list is unreadable and this is the reason for refactoring.
**Suggested fix:** Move the inversion to startup code in Swift: define `let allowedClassNames: Set<String> = ["person","bicycle","car","motorcycle","bus","truck","bird","cat","dog","horse","sheep","cow"]`, read `labels.txt`, compute the inverse against indices at construction time, emit the result into the gst-launch string. **Test the round-trip against the current filter list verbatim before merging** — assert the produced semicolon-list equals the existing `4;6;8;9;10;11;12;13;20;...;79` so you don't drift behaviour. (file: `detector-swift/Sources/Detector/GStreamerFrameReader.swift:626`)

**14. `tracker_config.yml` header still says "Optimized for Jetson Orin with YOLO11n detection".**
**Why:** Stale comment in both `detector-swift/tracker_config.yml:5` and `detector/tracker_config.yml:5`. Detector now runs YOLO26n. NvDCF tuning may still be reasonable but the header lies.
**Suggested fix:** Update comment; note whether tuning was re-validated for YOLO26's detection density. (file: `detector-swift/tracker_config.yml:5`)

**15. Tracker resolution regression vs Python sibling: `tracker-width`/`tracker-height` not set on Swift's nvtracker. (Added per codex R2.)**
**Why:** `detector/detector.py:1056-1057` sets `tracker.set_property('tracker-width', 640); tracker.set_property('tracker-height', 384)`. The Swift gst-launch string at `GStreamerFrameReader.swift:627-629` sets only `ll-config-file` and `ll-lib-file` — **no tracker-width or tracker-height**, so the Swift detector falls back to NvDCF defaults (typically 960×544). The Python and Swift detectors are therefore tracking at different resolutions, which means CPU/GPU cost and accuracy are different — invalidating the "same tracker, head-to-head" framing in `docs/benchmark-python-vs-swift.md`. Also: 960×544 is meaningfully more correlation-filter compute than 640×384 (1.97× more pixels per template).
**Suggested fix:** Add `tracker-width=640 tracker-height=384` to the nvtracker line in the Swift pipeline string. Re-run the head-to-head benchmark with matched tracker resolution and update `docs/benchmark-python-vs-swift.md` accordingly. (file: `detector-swift/Sources/Detector/GStreamerFrameReader.swift:627-629`)

**16. Tracker `compute-hw` not deliberately set; `nvinfer` uses `scaling-compute-hw` (different property). (Corrected split per codex R2.)**
**Why:** R1 introduced this finding but conflated two distinct properties. `nvinfer` exposes `scaling-compute-hw` (which engine handles input-tensor scaling: GPU vs VIC). `nvtracker` and `nvvideoconvert` expose `compute-hw`. On Jetson, defaults often pick VIC. VIC is energy-efficient but lower throughput than GPU for some color-conversion patterns; GPU is faster but contends with inference. For a pipeline with detection on the same GPU, leaving these as default means contention is implicit, not chosen.
**Suggested fix:** (a) Profile `scaling-compute-hw=1` (GPU) vs `=2` (VIC) for nvinfer pre-processing on a representative clip; set explicitly in `nvinfer_config.txt`. (b) Profile `compute-hw=1` vs `=2` on `nvtracker`; set explicitly on the gst-launch line. Document the choices. (file: `detector-swift/nvinfer_config.txt`, `detector-swift/Sources/Detector/GStreamerFrameReader.swift:627-629`)

**17. `wendy_extract_frame_timing` strict-less-than guard.**
**Why:** `nvds_shim.c:120` skips entries where `out <= in` as a clock-anomaly guard. With ms-quantized timestamps in DS 7.1, fast components can produce `out == in` exactly. This silently drops fast frames from histograms rather than counting them as ~0 ms.
**Suggested fix:** Change to `< in_system_timestamp` strictly. (file: `detector-swift/Sources/CGStreamer/nvds_shim.c:120`)

**18. Latency-meta wiring should be validated in a smoke test. (Added per codex R2.)**
**Why:** `Dockerfile:73-74` sets `NVDS_ENABLE_LATENCY_MEASUREMENT=1` and `NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1`, which is the documented path. But there's no startup assertion that nonzero decoder/mux/infer/tracker component metas actually arrive on this DS 7.1 install. If decoder latency stays zero (a known DS quirk in some configs), the code currently just silently reports zeros and the histogram looks "fine but missing one bucket". The "What's GOOD: latency-meta wiring" claim assumes this works on the target.
**Suggested fix:** Add a one-shot startup assertion: after the first 30 frames, check that each of `decode_ms / streammux_ms / inference_ms / postprocess_ms` has at least one non-zero observation; log a `WARN` if any bucket is permanently zero. Keeps the production code from claiming a latency story it isn't actually telling. (file: `detector-swift/Sources/Detector/Detector.swift:212-219`, `detector-swift/Sources/CGStreamer/nvds_shim.c:95-148`)

**19. Inverse-letterbox is settled: nvinfer DOES rescale parser output to source-frame coords. (Resolved per codex R1.)**
**Why:** I left this as "verify and document". Codex R1 settled it positively: DS 7.1 `attach_metadata_detector()` (in `gstnvinfer_meta_utils.cpp`) rescales parser boxes using `offset_left/top` and `scale_ratio_x/y` before attaching `NvDsObjectMeta`, **independent of cluster-mode**. The parser's existing comment is correct.
**Suggested fix:** Update the parser comment with a citation to `gstnvinfer_meta_utils.cpp::attach_metadata_detector()`. No code change. (file: `detector-swift/Sources/CNvdsParser/nvdsparsebbox_yolo26.cpp:11-13`)

### LOW

**20. `fakesink` defaults to `sync=false` — Finding was wrong as drafted. (Removed per codex R2.)**
**Why:** I claimed `fakesink sync=true` was the default and that it forced PTS pacing. Codex R2 caught: GStreamer `fakesink` defaults `sync=false` (verified against [GStreamer fakesink docs](https://gstreamer.freedesktop.org/documentation/coreelements/fakesink.html)). The pipeline as written does not have the rate-limiter I described. Keeping a stub here as a research item only.
**Suggested fix (if anything):** Set `fakesink sync=false async=false enable-last-sample=false` explicitly to (i) document the intent for any future maintainer who reads it, and (ii) make `enable-last-sample=false` save a small reference per buffer. Not load-bearing. (file: `detector-swift/Sources/Detector/GStreamerFrameReader.swift:630`)

**21. 51 MB `ffmpeg` aarch64 static binary committed at `detector-swift/ffmpeg`.**
**Why:** Gitignored from Docker context but in the repo. No reference in any source file. Looks like leftover debug. Bloats clones, looks unmaintained.
**Suggested fix:** `git rm`; consider `git filter-repo` if historical clone size matters. Add to `.gitignore` after removal. (file: `detector-swift/ffmpeg`)

**22. `cairo-stage/` (4.9 MB of vendored x11/cairo `.so`s) committed but Stage 2 declared dead.**
**Why:** `ARCHITECTURE.md:277,540` say Stage 2 removed the dependency. 28 `.so` files are dead weight.
**Suggested fix:** Delete the directory. Optionally keep `scripts/stage-cairo.sh`. (file: `detector-swift/cairo-stage/*`)

**23. `yolo26n.onnx` committed without license/attribution/export-command documentation.**
**Why:** Ultralytics releases under AGPLv3 by default. `detector-swift/LICENSE.txt` exists; check compatibility. Today there is no `docs/yolo26-export.md`, no SHA, no `yolo export` command. Combined with Finding 2 (engine artifact contract), the model provenance story is broken.
**Suggested fix:** Add `docs/model-provenance.md` with: source URL/version, license, export command, engine build command, and a SHA256 for `yolo26n.onnx`. (file: `detector-swift/yolo26n.onnx`)

**24. `force-implicit-batch-dim=0` documentation is muddled. (Tempered per codex R1.)**
**Why:** Codex R1 corrected my framing. On TRT 10.3 / JetPack 6.2.1, implicit batch is essentially legacy. `0` is the safe default.
**Suggested fix:** Remove the line entirely or trim the comment to one neutral sentence. (file: `detector-swift/nvinfer_config.txt:44-45`)

**25. `featureFocusOffsetFactor_y: -0.2` ("focus slightly above center, better for people") is asymmetric for mixed people+vehicles workload.**
**Why:** Biases DCF feature window upward; optimal for people, marginal for vehicles. Worth documenting.
**Suggested fix:** Note the trade-off in a comment, or split tracker configs per dominant class. (file: `detector-swift/tracker_config.yml:58`)

**26. `192.168.68.70:5000/swift-detector-base:latest` private registry reference in `wendy.json`.**
**Why:** Reveals internal infrastructure to a public reader. Stale cruft. The Dockerfile already documents it as unreachable.
**Suggested fix:** Remove or replace with `ubuntu:24.04`. (file: `detector-swift/wendy.json:6`)

**27. `symmetric-padding=1` correctness assumption is undocumented; `nvstreammux enable-padding=` is also unset. (Re-graded per codex R1; sequenced with Finding 3 per R2; promoted LOW → MEDIUM per R3 — bbox-correctness-affecting for non-16:9 sources, belongs near the engine/preprocessing contract; not HIGH because the documented camera path is 1080p.)**
**Why:** Codex R1 elevated this — padding compatibility between engine training and DS preprocessing is bbox-correctness-critical. Combined with `maintain-aspect-ratio=1`, this is an assumption about engine-export hyper-params. Legacy `nvstreammux` has its own `enable-padding`; if the source is not 16:9 1920×1080, mux may distort before nvinfer letterboxes. **R2 sequencing note:** adding `enable-padding=1` is valid only if the repo pins legacy mux per Finding 3. New mux does not have this property.
**Suggested fix:** After Finding 3 lands (`USE_NEW_NVSTREAMMUX=no` pinned), add `enable-padding=1` to `nvstreammux` properties. Document in `docs/engine-build.md` (alongside Finding 2) that `symmetric-padding=1` requires the engine was exported with centered letterbox (Ultralytics default). (file: `detector-swift/nvinfer_config.txt:33`, `detector-swift/Sources/Detector/GStreamerFrameReader.swift:624`)

**28. Trailing `;` line-comments in `nvinfer_config.txt` are bad style, not a parse-blocker. (Tempered per codex R1.)**
**Why:** GLib `GKeyFile` does not strip trailing `;` from string values, but the integer getter parses leading digits and ignores trailing garbage, so `model-color-format=0          ; RGB ...` resolves to `0`. Any future string-valued key with a trailing comment would corrupt the value.
**Suggested fix:** Convert all trailing `;` to standalone-line `#` comments. (file: `detector-swift/nvinfer_config.txt:18,25,26,27`)

**29. Pad-probe is `GST_PAD_PROBE_TYPE_BUFFER` only — no buffer-list, no event-side handling. (Added per codex R2.)**
**Why:** `gst_pad_add_probe` at `nvds_shim.c:227` registers `GST_PAD_PROBE_TYPE_BUFFER`. It does not fire for events (incl. EOS), queries, or buffer-lists. EOS is handled via bus polling in Swift, so this isn't a correctness bug today, but: (a) if upstream ever emits a `GstBufferList` (unusual for nvtracker, but not impossible), frames are silently lost; (b) there's no opportunity to emit a "final empty detection frame" on EOS for downstream cleanup; (c) track-finalization on EOS depends on the bus path, not the probe.
**Suggested fix:** Document in the probe header that buffer-lists and events are intentionally not handled (state the assumption that nvtracker emits only `GstBuffer`). Optionally add `GST_PAD_PROBE_TYPE_BUFFER_LIST` with a forwarding path that iterates the list — defensive, low cost. (file: `detector-swift/Sources/CGStreamer/nvds_shim.c:227-233`)

**30. `GST_PLUGIN_SCANNER=""` is a workaround that should be scoped/documented. (Added per codex R2.)**
**Why:** `setupPluginEnvironment()` at `GStreamerFrameReader.swift:649` sets `GST_PLUGIN_SCANNER=""` to avoid the Ubuntu 24.04 scanner ABI mismatch with CDI-injected JetPack GStreamer (documented in `ARCHITECTURE.md:545`). Disabling the external scanner means plugin discovery happens in-process, so a malformed plugin can crash or poison the app process during scan. Not a blocker — the workaround is plausible and documented — but the in-process risk should be acknowledged.
**Suggested fix:** Add a comment in `setupPluginEnvironment()` citing the ABI-mismatch reason and the in-process-scan risk; consider gating with an env-detection check (only blank the scanner if running inside the WendyOS / JetPack CDI environment). (file: `detector-swift/Sources/Detector/GStreamerFrameReader.swift:634-651`)

---

## What's GOOD (don't lose)

- **NVMM correctness via `gst_buffer_get_nvds_batch_meta`.** No `gst_buffer_map`, no NVMM pinning, no leak. The architecture (pad probe on `nvtracker.src` reading GstMeta) is genuinely clean **once Findings 6, 9, 10 are addressed** (the lifecycle gaps don't undo the design — they qualify the implementation).
- **C-shim boundary design.** `WendyDetection`/`WendyFrameTiming` POD structs let Swift never touch GLib types. POD copy in `wendy_nvds_flatten` means Swift gets a snapshot; no use-after-free risk on `obj_label` because the shim does not touch it.
- **Reconnect logic shape.** Bus polling at 500 ms with exponential backoff capped at 30 s, FPS gauge reset on disconnect, preserved subscriber stream across reconnects.
- **Concurrency contract documentation.** The 70-line header on `GStreamerFrameReader.swift` describing thread isolation, retain semantics, and teardown order is the kind of thing senior reviewers want to see — it's the documentation that exposed Finding 9.
- **Custom parser scope.** ~80 lines that does exactly one thing.
- **Latency-meta wiring** (subject to Finding 18's smoke test). Picking up `NVDS_LATENCY_MEASUREMENT_META` and bucketing by component name with documented substring matching is the right way.
- **Honest caveat in the blog.** Stating "pad-probe histogram averages include nvstreammux queueing time" is a trust signal.

---

## Codex Round 1 verdict

**Source:** `/tmp/codex-gpu-r1-output.md`.

**What R1 corrected (applied to my findings):**
- Finding 4 (live-source): I overstated the failure mode. Real cost is wrong PTS propagation, not "mux waits for a non-arriving batch". Reworded.
- Finding 5 (queues): "Single thread" was sloppy — GStreamer does spawn task threads. Reframed as back-pressure / latency-isolation.
- Finding 12 (topk dead): Wrong reason. `topk` is a `filterTopKOutputs` cap, not a clustering knob.
- Finding 13 (filter list): My proposed allowlist would have changed behavior. Caught and corrected (then re-corrected in R2 — see below).
- `maxShadowTrackingAge` finding (R1 demoted as overconfident research item; not carried into R2's renumbered finding list — noted here so the reference is not orphaned).
- Finding 10 (ProbeContext leak): Real, but "MB/day" was overstated; sub-100 KB/day. Reworded.
- Finding 19 (inverse letterbox): Settled positively. No code change needed.
- Finding 20 (fakesink sync — original): I implied this caused the 50 ms intervals. Wrong — those are source-pacing.
- Finding 24 (force-implicit-batch-dim): Muddled framing; demoted to LOW.
- Finding 27 (symmetric-padding): Codex R1 said HIGH; kept MEDIUM-coupled to engine provenance. Also added the upstream-`enable-padding` half I missed.
- Finding 28 (trailing comments): Tempered.

**What R1 added (new findings):**
- HIGH: engine artifact contract (now Finding 2). Adopted.
- HIGH: streammux v1/v2 environment dependency (now Finding 3). Adopted.
- MEDIUM: parser tensor-ABI validation (now Finding 11). Adopted.
- MEDIUM: `compute-hw` / `scaling-compute-hw` (now Finding 16, split per R2). Adopted.

**R1 verdict for me:** Conflated "looks wrong" with "is wrong" in 4-5 findings. The new HIGH findings codex added are exactly what a senior NVIDIA engineer flags first.

## Codex Round 2 verdict

**Source:** `/tmp/codex-gpu-r2-output.md`.

**What R2 corrected (applied to the revised review):**
- **Finding 13 (filter-out-class-ids polarity bug — publication blocker):** I had absorbed R1's correction with the wrong polarity. The list **filters out** 9, 10, 11 (traffic light, fire hydrant, stop sign); current behaviour **keeps** 0,1,2,3,5,7,14-19. **Rewritten** with polarity-correct text and a "test the round-trip" mandate.
- **Finding 16 (fakesink sync=true):** Plain `fakesink` defaults to `sync=false` (verified against [GStreamer fakesink docs](https://gstreamer.freedesktop.org/documentation/coreelements/fakesink.html)). The whole finding rested on a false premise. **Demoted to LOW (Finding 20)** with a stub explaining the correction.
- **Finding 11 (parser ABI exact-shape check):** My proposed `numDims == 2 && d[0] == 300 && d[1] == 6` was too brittle — DS may report `[300,6]` or `[1,300,6]` depending on explicit-batch reporting. **Rewritten** to "log first, then encode the observed exact ABI".
- **Finding 16 (compute-hw):** Split into `scaling-compute-hw` (nvinfer) vs `compute-hw` (nvtracker / nvvideoconvert). Two distinct properties, one finding addressed both wrong.
- **TL;DR:** I had said "3 net-new HIGH findings" but parser-ABI is MEDIUM, not HIGH. Corrected to "2 HIGH + 1 MEDIUM".

**What R2 added (new findings):**
- **HIGH 6: Failed probe install is logged but not fatal.** `startPipeline` continues to PLAYING with `id == 0`. Adopted as Finding 6.
- **MEDIUM 9: Pad-probe lifecycle UAF risk.** `gst_pad_remove_probe` does not block in-flight callbacks; `Unmanaged.release()` and (proposed) `g_free` race with the streaming thread. Adopted as Finding 9, **changing my "GOOD: retain-balance discipline is correct" claim**.
- **MEDIUM 15: Tracker-width/height regression vs Python sibling.** Swift falls back to NvDCF defaults (~960×544); Python uses 640×384. Real benchmark-validity issue. Adopted.
- **MEDIUM 18: Latency-meta needs smoke test on target.** Env vars are set but no validation. Adopted.
- **LOW 29: Probe type only covers buffers** (no buffer-list, no event-side handling). Adopted as documentation/defensive note.
- **LOW 30: `GST_PLUGIN_SCANNER=""` workaround should be scoped.** In-process plugin scan has its own risk surface. Adopted as comment/scoping fix.

**What R2 confirmed unchanged:**
- HIGH 2 (engine artifact contract), HIGH 3 (streammux v1/v2), HIGH 4 (live-source — kept HIGH due to overlay-sync/SLO tie-in, with R2's caveat that MEDIUM is also defensible).
- The polarity-corrected Finding 13 + lifecycle-aware Finding 9 are the publication-readiness blockers.

**R2 verdict for me:** R3 needed before publication. The big shape is right; the specific fix code/wording in Findings 13, 16, and 11 was wrong. R2 also surfaced the lifecycle/MT-safety concerns I should have caught in the first pass — that's the "different-engineer perspective" R1 asked about made concrete.

**Counts:** 8 HIGH + 11 MEDIUM + 11 LOW = 30 findings (post R1+R2 reorganization).

## Codex Round 3 verdict

**Source:** `/tmp/codex-gpu-r3-output.md`.

**What R3 corrected (applied inline above):**
- **Finding 13 (filter list count):** I called it a "78-class list." It is **68 filtered classes** — 8 explicit IDs before 20, plus 20-79 inclusive (60 more). The kept-complement count of 12 was correct; only the suppress-count was wrong.
- **Finding 9 (lifecycle UAF wording):** I said the alternative was to "leak `ctx` and `DetectionStream` retain on every cycle." Wrong — Swift releases `retainedBox` in teardown today; the `DetectionStream` retain is not leaked. Reworded to "keep leaking `ProbeContext` (sub-100 KB/day) and introduce active-callback draining only when a confirmed UAF is observed."
- **Finding 9/TL;DR (`g_free` framing):** I implied `g_free` via `GDestroyNotify` happens with `Unmanaged.release()` in present code. Current code passes `NULL`. Reworded as a future-fix hazard, not present behaviour.
- **Finding 4 (live-source rationale):** I tied `live-source=1` to `NVDS_LATENCY_MEASUREMENT_META` correctness. Wrong — the C shim reads component system timestamps, not PTS. The strong case is `ptsNs` / overlay alignment only.

**What R3 demoted (applied inline above):**
- Finding 3 (streammux v1/v2): HIGH → MEDIUM. Environment-sensitive, not broken by default. Promote back to HIGH only if the Wendy/CDI runtime is shown to plausibly export `USE_NEW_NVSTREAMMUX=yes`.
- Finding 4 (live-source): HIGH → MEDIUM. Real, not clone-and-run blocking.
- Finding 5 (queues): HIGH → MEDIUM. Production engineering call.
- Finding 6 (probe-install non-fatal): HIGH → MEDIUM. Robustness bug, not a blocker on the static-pad path.
- Finding 8 (parser non-finite): HIGH → MEDIUM. Good engineering, not publication-blocking for a showcase.

**What R3 promoted (applied inline above):**
- Finding 27 (symmetric-padding / `enable-padding`): LOW → MEDIUM. Bbox-correctness-affecting for non-16:9 sources.

**What R3 noted as fall-out:**
- `maxShadowTrackingAge` referenced in the R1 verdict as an "original number" but no finding by that name existed in R2's renumbered list. Reworded the R1 verdict note so the reference is not orphaned.

**R3 bottom line:** "R3 is much closer and the main blockers are right: engine artifact/build contract, vendored-header build story, credentials, and filter-list polarity. I would publish after fixing the small factual errors and demoting the real-but-not-blocking HIGHs."

**Counts post-R3:** 3 HIGH (Findings 1, 2, 7) + 16 MEDIUM (3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 27) + 10 LOW (19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30 — 11 entries; Finding 27 was promoted out of LOW).

**R3 verdict for me:** Fixable. Publication-ready after the inline edits above.

Sources:
- [GStreamer fakesink docs](https://gstreamer.freedesktop.org/documentation/coreelements/fakesink.html)
