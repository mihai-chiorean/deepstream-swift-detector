# Tracker tuning + frame-bundled metadata transport — research notes

**Status:** research notes, 2026-04-16. No code/config changes were applied based on this — Mihai's call on which (if any) to ship.

**Prompted by:** Mihai's questions after the bbox-lead UX work — (1) is there room to make the tracker more accurate or efficient, (2) is there a way to send box data with the frame so the timing residual goes away.

---

## Q1 — Tracker accuracy & efficiency

Current tracker is NvDCF via `libnvds_nvmultiobjecttracker.so`, configured at `detector-swift/tracker_config.yml`. Detection input is unfiltered (all 80 COCO classes pass through to the tracker).

### NvDCF knobs worth touching

| Knob | Now | Suggested | Tradeoff |
|---|---|---|---|
| `minTrackerConfidence` (L17) | 0.2 | 0.3-0.35 | Less shadow-mode noise; risk dropping tracks during brief occlusions |
| `featureImgSizeLevel` (L57) | 2 | 1 | ~30% less feature-extraction CPU; weaker color-name discrimination on small objects |
| `filterLr` (L61) | 0.075 | 0.1 | Faster adaptation to appearance change; more drift on still subjects |
| `useColorNames` / `useHog` (L55-56) | 1 / 0 | keep | Already tuned for Jetson |
| `maxShadowTrackingAge` (L19) | 30 | 15 | Faster reclaim of stale tracks at 21 fps (1.4s vs 0.7s of prediction); risk re-IDing across long occlusions |
| `maxTargetsPerStream` (L13) | 150 | 40 | Hard cap on per-frame work; only matters in extreme crowds |
| `trackerWidth/Height` | unset (defaults 960x544) | 640x384 | Lower visual-tracker compute; reduced sub-pixel precision |

ReID is off (NvDCF visual-tracker only). Leave it off.

### Class filter — biggest quick win

`gst-nvtracker` exposes **`operate-on-class-ids`** as an element property — a semicolon-separated list of class IDs the tracker will process. Non-matching detections pass through without tracker metadata. Set on the `wendy_tracker` element in `GStreamerFrameReader.swift:359`:

```
operate-on-class-ids=0;1;2;3;5;7
```

(person, bicycle, car, motorcycle, bus, truck)

This is a real CPU win — visual feature extraction is skipped for filtered classes. YOLO26's 80 COCO outputs include couches, toasters, refrigerators etc. that we will never see and never want tracks for; today the tracker burns CPU on probationary tracks for them.

Source: [NVIDIA gst-nvtracker docs](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html).

### Detection confidence floor

Current `pre-cluster-threshold=0.4` (`detector-swift/nvinfer_config.txt:48`, gate enforced in `nvdsparsebbox_yolo26.cpp:86`). For person/car YOLO26n at 1080p, **0.45-0.5** kills the long tail of low-confidence false positives that the tracker would otherwise spend probation frames (`probationAge: 3`, L18) confirming.

### Alternative trackers on DeepStream 7.1

Per NVIDIA: IOU and NvSORT need **no pixel data** (no RGBA/NV12 buffers). NvDCF and NvDeepSORT do.

- **IOU** — bbox-only, baseline; ID switches galore on crowd occlusions. Skip.
- **NvSORT** — Kalman + cascaded association, no pixel processing. **Biggest CPU win for this workload**, decent ID stability for cars/people moving smoothly. **The switch worth making.**
- **NvDeepSORT** — ReID CNN, best accuracy, heaviest. Overkill at 21 fps single-stream on Orin Nano.

### Measurement

Cheapest credible metric: **ID-switch count per minute** on a fixed clip (count distinct `trackId`s assigned to the same physical object over time). The pipeline already exposes `postprocessMs` per-component via the latency probe (`GStreamerFrameReader.swift:71`); diff that across configs for a CPU number. `track-continuity = mean(track lifetime in frames) / mean(detection lifetime)` is one log-grep away.

---

## Q2 — Frame-bundled metadata transport

Mihai's intuition: WebRTC media + WebSocket detections on separate transports is what forces PTS-based client-side alignment with residual lead/lag. If bbox metadata could travel **in-band** with the video, decode and render would be lockstep.

| Path | Feasibility on our stack | Effort | Timing residual | Right for |
|---|---|---|---|---|
| **A. H.264 SEI user_data_unregistered NAL** | DeepStream encoder supports `sei-payload` on `nvv4l2h264enc`; mediamtx passes NALs through. Browser-side: WebCodecs `VideoDecoder` exposes encoded chunks but **does not surface SEI** — needs a JS Annex-B parser before handing chunks to the decoder. Chrome 94+, Safari 16.4+, Firefox flag-gated. | 4-7 days | **Pixel-perfect** | Production, demo with WebCodecs path |
| **B. WebRTC DataChannel + RVFC `rtpTimestamp`** | mediamtx WebRTC already negotiates SCTP; add a server-side data-channel publisher. Browser: `requestVideoFrameCallback`'s `metadata.rtpTimestamp` is Baseline 2024 ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback)) and works for remote tracks. Correlate by RTP ts (90 kHz). | **2-3 days** | **Sub-frame** (same clock domain, jitter-correlated) | Production, debug |
| **C. `nvdsosd` server-side burn-in** | Already done in Stage 1; we know it costs ~23 MB nvmap and reintroduces the dropped branch. | <1 day to revert | **Pixel-perfect** | Investor demo, low-end clients with no JS |
| **D. RTP header extensions** | mediamtx does not expose extension-header injection — fork required. ~255-byte cap forces fragmentation across packets within a frame; ordering nightmare. | 7-10+ days | Pixel-perfect-ish | Don't |

### Recommendation

For a sharp blog in days, the honest answer is **B** — WebRTC data channel keyed to `rtpTimestamp` collapses the alignment problem to one clock domain without re-architecting the encoder. Days of work, browser-universal, sub-frame in practice — close enough to feel locked at 21 fps.

SEI (A) is the technically prettiest answer and the right long-term play, but the WebCodecs detour is a week of work whose payoff is invisible to readers.

For a blog narrative, the strongest story is option B shipped + a closing section that explains why A is the eventual destination and C is what we'd run at an investor demo today. "PTS-aligned client-side overlay was the right call given the constraints" is also defensible — but only if the post quantifies the residual lead/lag we actually measured. Without a number on the page, readers will assume we never tried.

---

## Combined recommendation

**Tracker:** switch to **NvSORT**, add `operate-on-class-ids=0;1;2;3;5;7` on the nvtracker element, raise `pre-cluster-threshold` to 0.45. Measurable CPU drop and a clean blog graph.

**Transport:** ship **DataChannel + rtpTimestamp**. Document SEI as future work. Don't burn nvmap on burn-in unless the demo requires it.

---

## Sources

- Gst-nvtracker — [DeepStream documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html)
- [config_tracker_NvDCF_perf.yml (NVIDIA-AI-IOT)](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/apps/deepstream-nvdsanalytics/config_tracker_NvDCF_perf.yml)
- [NVIDIA DeepStream Tracker Deep Dive (Edge AI Vision)](https://www.edge-ai-vision.com/2022/06/nvidia-deepstream-technical-deep-dive-multi-object-tracker/)
- [HTMLVideoElement.requestVideoFrameCallback (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback)
- [GstVideo SEI Unregistered User Data](https://gstreamer.freedesktop.org/documentation/video/gstvideosei.html)
- [WebCodecs SEI/Recovery Point issue #650](https://github.com/w3c/webcodecs/issues/650)
