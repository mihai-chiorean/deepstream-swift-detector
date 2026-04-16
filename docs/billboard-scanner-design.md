# Billboard Scanner: Mobile Detection & Inventory System

## Design Document v4.0

**Project**: Automated billboard detection, OCR, and geo-tagged inventory from a moving vehicle
**Platform**: Jetson Orin Nano 8GB + WendyOS + USB3 industrial camera + continuous video
**Author**: Mihai
**Date**: 2026-03-31

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-03-31 | Initial design with DSLR tethered capture |
| v2.0 | 2026-03-31 | Continuous video pipeline, staged inference, automotive hardening, calibration |
| v3.0 | 2026-03-31 | USB-only V4L2 (cut CSI), GPU memory budget, billboard-specific tracker FSM, corrected durability semantics, PaddleOCR feasibility spike, frame buffer memory design, deployment profiles |
| v3.1 | 2026-03-31 | Fixed GPU arbitration (chunked OCR), corrected frame-ring memory math, explicit low-light operating envelope, single-side coverage scope constraint |
| v4.0 | 2026-04-01 | Replaced PaddleOCR with Qwen3.5-2B VLM via llama.cpp. Natively multimodal (no separate -VL model). Added Appendix A (OCR vs VLM comparison) with TRT-LLM vs llama.cpp runtime analysis. |

---

## 1. Problem Statement

Outdoor advertising inventory (billboards, signs, posters) is surveyed manually today -- people drive around, take photos, and log them in spreadsheets. This is slow, expensive, and error-prone. We want an automated system that:

1. Continuously captures video from a car-mounted camera while driving
2. Detects billboards in each frame using on-device ML, tracking them across frames
3. Selects the best frame per billboard (sharpest, most fronto-parallel) for OCR
4. Reads text on billboards (company name, URL, phone number) via OCR on selected frames
5. Tags each detection with **observation point** GPS coordinates, heading, and timestamp
6. Uploads structured results (image crop, text, location, metadata) to a web dashboard

The entire pipeline runs on a Jetson Orin Nano in the vehicle -- no cloud inference required.

**Foundational design decisions**:
- **Continuous video** with tracked objects and best-frame selection (not tethered stills)
- **USB3 V4L2 camera only** for v3.0 (CSI/libargus is a future option, explicitly out of scope)
- **One OCR job at a time**, serialized behind YOLO inference via a GPU semaphore

---

## 2. System Architecture

```
                                    ┌─────────────────────────────┐
                                    │         Web Dashboard       │
                                    │  (Billboard Inventory Map)  │
                                    └──────────────▲──────────────┘
                                                   │ HTTPS POST
                                                   │ (WiFi/LTE hotspot)
┌──────────────────────────────────────────────────┴───────────────────┐
│                        Jetson Orin Nano (WendyOS)                    │
│                                                                      │
│  ┌──────────┐   USB3   ┌──────────────────────────────────────────┐  │
│  │ e-CAM80  │─────────▶│            Billboard Scanner App         │  │
│  │ (V4L2)   │          │                                          │  │
│  └──────────┘          │  Stage 1: Continuous (every frame)       │  │
│                        │  ┌────────┐  ┌──────────┐  ┌─────────┐  │  │
│  ┌──────────┐   USB    │  │ V4L2   │─▶│ YOLO Det │─▶│Billboard│  │  │
│  │ GPS      │─────────▶│  │Capture │  │(TensorRT)│  │Tracker  │  │  │
│  │(ZED-F9R) │          │  │(MMAP)  │  │ FP16     │  │(FSM)    │  │  │
│  └──────────┘          │  └────────┘  └──────────┘  └────┬────┘  │  │
│                        │                                  │       │  │
│                        │  Stage 2: Per-track (on finalize)│       │  │
│  ┌──────────┐          │  ┌──────────────────────────┐    │       │  │
│  │ USB SSD  │◀─────────│  │ Best-crop VLM pipeline   │◀───┘       │  │
│  │(storage) │          │  │ (GPU semaphore: 1 job)   │            │  │
│  └──────────┘          │  │ Qwen3-VL via llama.cpp   │            │  │
│                        │  └────────────┬─────────────┘            │  │
│                        │               ▼                          │  │
│                        │  ┌─────────────────────────────────────┐ │  │
│                        │  │  Storage (SQLite FULL + fsync chain)│ │  │
│                        │  └─────────────────────────────────────┘ │  │
│                        └──────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.1 Two-Stage Pipeline

**Stage 1 -- Continuous Detection + Tracking** (every frame, ~10 FPS)
1. V4L2 delivers NV12 frames via `VIDIOC_REQBUFS` + `MMAP` at 10 FPS, 1920x1080
2. NV12 -> RGB conversion + resize to 640x640 on CPU (~5ms)
3. YOLO billboard detector (TensorRT FP16) -- ~35ms
4. Detections fed into billboard-specific tracker (FSM, see Section 4.5)
5. For each active track: compute Laplacian sharpness of the detection crop at capture resolution. If sharper than current best, replace the stored best-crop JPEG.
6. GPS position interpolated to frame timestamp from gpsd

**Stage 2 -- Best-Crop OCR** (once per billboard, on track finalization)
1. When a track transitions to `finalized` state, submit its best crop to the OCR queue
2. VLM queue is serialized (max 1 concurrent GPU job) via async semaphore
3. Qwen3-VL extracts text, company name, URL, phone, and description in one inference call (~1-3s)
4. Parse structured JSON response
5. Write observation record: fsync image file, rename, fsync directory, then INSERT into SQLite

**GPU arbitration**: TensorRT kernels are not preemptible. A semaphore alone cannot guarantee YOLO priority over an in-flight OCR job. The actual scheduling model is **chunked OCR with frame-level admission control**:

1. The main loop runs YOLO on each frame at ~10 FPS (~35ms per inference)
2. Between YOLO calls, there is ~65ms of idle GPU time
3. OCR is broken into **individual TensorRT calls** (one det invocation OR one rec invocation per slot), not a monolithic pipeline
4. Before each OCR chunk, the scheduler checks if a new frame is ready for YOLO. If yes, YOLO runs first.
5. A 5-region OCR job takes ~7 chunks (1 det + 5 rec + 1 postprocess), spread across ~7 inter-frame gaps (~700ms wall time)

This means OCR runs opportunistically in YOLO's idle slots but **never delays** a YOLO frame. The worst case is OCR latency increases under high FPS, not that detection drops frames. If billboard density causes OCR backlog (>10 queued jobs), the oldest queued jobs are dropped with a log warning.

---

## 3. Hardware

### 3.1 Bill of Materials

| # | Component | Specific Product | Purpose | Price (USD) |
|---|-----------|-----------------|---------|-------------|
| 1 | **Compute** | NVIDIA Jetson Orin Nano 8GB Dev Kit | On-device ML inference | $200 |
| 2 | **Camera** | e-con Systems e-CAM80_CUNANO (8MP, USB3, global shutter) | V4L2 native, industrial grade, no rolling shutter skew | $150 |
| 3 | **GPS** | u-blox ZED-F9R evaluation kit (USB) | 20 Hz GNSS + IMU dead reckoning | $220 |
| 4 | **GPS (budget alt)** | u-blox NEO-M9N USB breakout | 25 Hz GNSS, no dead reckoning | $50 |
| 5 | **Storage** | Samsung T7 500GB USB-C SSD | Local capture + result storage | $50 |
| 6 | **Camera housing** | IP67 project box + optical glass window + CPL filter | Weatherproof exterior mount | $40 |
| 7 | **Camera mount** | RAM Mounts RAM-B-166U suction cup + 1" ball arm | Automotive-grade, vibration dampened | $60 |
| 8 | **Safety tether** | Steel cable lanyard + adhesive anchor | Secondary camera retention | $10 |
| 9 | **Power: DC-DC** | Victron Orion-Tr 12/12-9 isolated converter | Surge/transient/reverse-polarity protection | $45 |
| 10 | **Power: hold-up** | 10F supercapacitor module (12V rated) | ~3s hold-up for graceful shutdown | $25 |
| 11 | **Enclosure** | Pelican 1200 + 80mm filtered fan + dust filter | Forced airflow for Jetson + SSD | $55 |
| 12 | **USB hub** | Industrial USB 3.0 hub (4-port, powered) | Camera + GPS + SSD | $30 |
| 13 | **CPL filter** | Circular polarizer sized to camera lens | Reduce billboard surface glare | $15 |
| 14 | **Connectivity** | Phone hotspot or Quectel RM520N LTE modem | Upload when available | $0-80 |
| 15 | **Misc** | Strain-relief cable glands, vibration pads, thermal paste, zip ties | Automotive hardening | $25 |
| | **Total (with ZED-F9R)** | | | **~$925 - $1,005** |
| | **Total (budget, NEO-M9N)** | | | **~$755 - $835** |

### 3.2 Camera: USB3 V4L2 Only (v3.0 Scope)

CSI/libargus is explicitly **out of scope** for v3.0. Rationale:
- CSI frames live in Argus/EGL/NVMM surfaces, requiring a completely different capture pipeline
- Forcing CSI data through CPU copies for the V4L2-style `[UInt8]` interface eliminates zero-copy benefits
- V4L2 over USB3 is simpler, well-tested, and sufficient for 8MP @ 10 FPS
- CSI support can be added as a v4.0 feature if USB3 bandwidth or latency becomes a bottleneck

**V4L2 capture specifics**:
- Buffer mode: `MMAP` with 4 kernel-side buffers (`VIDIOC_REQBUFS`)
- Pixel format: `V4L2_PIX_FMT_NV12` (native to most USB3 cameras, avoids in-camera conversion)
- Timestamp source: `V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC` from sensor SOF (start-of-frame) when available, else `CLOCK_MONOTONIC` at dequeue time
- Queue depth: 4 buffers. If processing falls behind, oldest buffer is dropped (not accumulated)

### 3.3 Power Architecture

```
Car 12V ──┬── Cigarette lighter / accessory circuit
           │
           ▼
  ┌──────────────────────────────┐
  │ Victron Orion-Tr 12/12-9    │
  │ Input: 9-18V (load dumps,   │
  │ cranking dips handled)       │
  │ Galvanic isolation           │
  │ Output: 12.0V / 9A          │
  │ Remote ON/OFF ← ignition    │
  └──────────┬───────────────────┘
             │
    ┌────────┴────────┐
    │ 10F supercap    │ ← 3s hold-up for graceful shutdown
    └────────┬────────┘
             │
    Jetson barrel jack (12V / 1.25A max)
    Camera + peripherals via USB power
```

**Shutdown sequence**: Victron remote-on falls when ignition off. Supercap holds power for ~3 seconds. Jetson detects power-loss via voltage monitoring (ADC or USB watchdog) and initiates: flush SQLite WAL checkpoint, close camera, sync filesystem, then halt.

### 3.4 Thermal Design

| Parameter | Specification |
|-----------|--------------|
| Ambient assumption | Up to 50C cabin (parked car in sun, worst case) |
| Jetson Tj max | 95C (throttles at 85C) |
| Enclosure | Pelican 1200, two 80mm filtered openings (intake low, exhaust high) |
| Airflow | Noctua NF-A8 80mm, 50 CFM, pulling air through case |
| SSD placement | In airflow path, Samsung T7 operates 0-60C |
| Monitoring | `tegrastats` polled every 5s: GPU temp, CPU temp, throttle flags, memory |
| Throttle policy | GPU > 80C: reduce to 5 FPS. > 85C: detection only, skip OCR. > 90C: pause inference, alert. |

Camera is exterior-mounted in IP67 housing -- not affected by cabin temperature.

---

## 4. Software Architecture

### 4.1 WendyOS App Structure

```
billboard-scanner/
├── wendy.json
├── Dockerfile
├── Package.swift
├── Sources/
│   └── BillboardScanner/
│       ├── BillboardScanner.swift    # @main entry point
│       ├── V4L2Capture.swift         # V4L2 MMAP capture, NV12 frames
│       ├── GPSReader.swift           # gpsd JSON client
│       ├── BillboardDetector.swift   # YOLO TensorRT engine
│       ├── BillboardTracker.swift    # Billboard-specific FSM tracker
│       ├── FramePool.swift           # Global refcounted frame ring
│       ├── BestCropStore.swift       # Per-track best JPEG crop
│       ├── VLMEngine.swift            # Qwen3-VL text/brand extraction
│       ├── ResultAggregator.swift    # Merge detection + OCR + GPS
│       ├── StorageManager.swift      # Durable write chain
│       ├── UploadQueue.swift         # HTTPS sync with backpressure
│       ├── ThermalMonitor.swift      # tegrastats, throttle policy
│       ├── HTTPServer.swift          # Hummingbird status dashboard
│       └── Metrics.swift             # Prometheus metrics
├── models/
│   └── billboard_yolo11s.onnx        # VLM (Qwen3-VL) loaded via llama.cpp at runtime
├── config/
│   ├── scanner.json
│   ├── calibration.json
│   └── labels.txt                    # "billboard"
└── scripts/
    ├── train_billboard_model.py
    └── calibrate_camera.py
```

### 4.2 wendy.json (USB Deployment Profile)

```json
{
    "appId": "sh.wendy.billboard-scanner",
    "version": "1.0.0",
    "language": "swift",
    "entitlements": [
        { "type": "gpu" },
        { "type": "network", "mode": "host" },
        { "type": "device", "path": "/dev/video0", "description": "USB3 camera (V4L2)" },
        { "type": "device", "path": "/dev/ttyACM0", "description": "GPS serial device" },
        { "type": "volume", "source": "/mnt/ssd", "target": "/data", "description": "USB SSD for captures" }
    ]
}
```

**What's NOT in v3.0 entitlements** (and why):
- No CSI/libargus device mounts -- USB V4L2 only
- No GPIO access for shutdown signaling -- using USB voltage monitoring instead
- No PPS device -- timestamp sync via gpsd NTP discipline, not hardware PPS

If a CSI profile is needed later, it would require additional entitlements for `/dev/video*` (Argus-managed), and potentially socket access to `nvargus-daemon`.

### 4.3 GPU Memory Budget

**Critical constraint**: Orin Nano 8GB has **shared** CPU/GPU memory. All allocations compete.

| Component | GPU Memory (MB) | Notes |
|-----------|----------------|-------|
| YOLO11s TensorRT engine | ~80 | FP16, batch=1, 640x640 input |
| YOLO execution context + workspace | ~120 | `maxWorkspaceSize = 128MB` |
| YOLO input/output CUDA buffers | ~5 | 640x640x3 float + 84x8400 float |
| Qwen3.5-2B Q4_K_M (llama.cpp) | ~1,500-2,000 | Natively multimodal VLM for text/brand extraction |
| V4L2 frame buffers (4x NV12 1080p) | ~12 | Kernel-mapped, 3.1MB each |
| Global frame ring (see 4.6) | ~50 | 16 frames x 3.1MB NV12 @ 1080p. Hard cap at 200MB for future 8MP upgrade. |
| **Total estimated peak** | **~1,767** | |
| **Available (8GB - OS/runtime)** | **~6,500** | Comfortable headroom (~4.7GB free) |

**Hard caps enforced in code**:
- TensorRT workspace: 128MB per engine (set via `IBuilderConfig`)
- Global frame ring: 200MB max, 16 slots, refcounted
- VLM queue: max 1 concurrent job (async semaphore), YOLO frames processed first
- Track crop store: per-track best crop stored as compressed JPEG (50-200KB), not raw pixels

**Validation milestone**: Run `tegrastats` on target device with YOLO engine + Qwen3.5-2B loaded simultaneously, confirm peak usage stays under 3GB, leaving 5GB for OS + buffers + headroom.

### 4.4 ML Pipeline

#### Billboard Detection

- **Model**: YOLO11s, fine-tuned on billboard data only
- **Single class**: `billboard` (V1). Street signs and storefronts are out of scope.
- **Input**: 640x640 RGB, batch=1
- **Output**: [1, 84, 8400] -> NMS -> bboxes with confidence
- **TensorRT**: FP16, workspace 128MB, engine auto-built from ONNX on first launch

#### Text & Brand Extraction (VLM)

Stage 2 uses **Qwen3.5-2B** (Q4_K_M GGUF, ~1.5-2GB GPU) via llama.cpp to extract text, brands, and context from billboard crops in a single inference call. This replaces the traditional OCR pipeline (PaddleOCR) that was evaluated and rejected -- see Appendix A for the full comparison.

Qwen3.5-2B is a natively multimodal model (vision + language built into every variant, no separate "-VL" model). It scores 64.2 on MMMU and has native tool-calling and structured JSON output support.

**Why VLM instead of OCR**:
- Answers the actual question ("what company?") rather than just reading characters
- Handles stylized fonts, logos, partial occlusion, and non-Latin scripts
- No ONNX/TensorRT export pain -- llama.cpp runs natively on aarch64 CUDA
- Already integrated in the deepstream-vision stack (vlm service on port 8090)
- Eliminates the highest integration risk from prior design versions

**Why llama.cpp, not TensorRT-LLM**:
TRT-LLM generates hardware-optimized kernels that fully exploit tensor cores (SM87 on Orin), and would likely be 30-50% faster for decode. However, TRT-LLM's Jetson branch (`v0.12.0-jetson`) is only tested on AGX Orin 64GB, and its runtime overhead (paged KV cache manager, inflight batching) is designed for datacenter GPUs with memory to spare. On 8GB shared memory, the integration risk is high. Since we run VLM only ~1-5 times per minute, the llama.cpp speed is more than sufficient. TRT-LLM is a future upgrade if NVIDIA ships proper Orin Nano support.

**VLM prompt**:
```
You are analyzing a billboard photograph. Extract:
- company_name: The advertiser's name
- website: Any URL visible (null if none)
- phone: Any phone number visible (null if none)
- description: What is being advertised (1 sentence)

Respond in JSON only.
```

**Performance**: ~1-3 seconds per crop. Acceptable because:
- OCR runs once per physical billboard (on track finalization), not per frame
- Crops are queued and processed in YOLO's inter-frame GPU gaps (chunked scheduling)
- A 2-second VLM call is spread across ~20 inter-frame slots

**Fallback**: If Qwen3.5-2B cannot run on the target device (memory pressure, build issues), defer text extraction to the upload server where a larger VLM or cloud API can process the crops.

### 4.5 Billboard Tracker (FSM Design)

Standard MOT trackers (IoU + Kalman) are designed for small, fast-moving objects. Billboards are large, slow-moving (relative to camera), and occluded by poles/trees. The tracker uses a billboard-specific state machine:

```
                ┌──────────┐
     new det ──▶│ TENTATIVE│──── 3+ consecutive matches ────┐
                └────┬─────┘                                │
                     │ miss for 5+ frames                   │
                     ▼                                      ▼
                 [discarded]                          ┌──────────┐
                                                     │  ACTIVE  │
                                                     └────┬─────┘
                                                          │
                              ┌────────────────────────── │ ──────────────┐
                              │ bbox area decreasing      │               │
                              │ for 5+ frames             │ miss for      │
                              │ (receding)                │ 30 frames     │
                              ▼                           │ (long miss    │
                        ┌──────────┐                      │  tolerance)   │
                        │ EXITING  │                      │               │
                        └────┬─────┘                      │               │
                             │ area < min_threshold       │               │
                             │ OR miss for 10 frames      │               │
                             ▼                            ▼               │
                        ┌───────────┐                                     │
                        │ FINALIZED │◄────────────────────────────────────┘
                        └─────┬─────┘
                              │
                              ▼
                     Submit best crop to OCR queue
```

**State transitions**:
- **TENTATIVE -> ACTIVE**: 3+ consecutive frame matches (prevents spurious single-frame detections)
- **ACTIVE -> EXITING**: Bounding box area has been monotonically decreasing for 5+ frames (billboard is receding in rear-view). Uses hysteresis: area must decrease by >5% cumulative.
- **ACTIVE -> FINALIZED**: Track lost for 30 frames without entering EXITING (sudden occlusion or lane change). Long miss tolerance prevents fragmentation from momentary occlusions (poles, trees, trucks).
- **EXITING -> FINALIZED**: Either bbox area drops below minimum OCR-viable size, or track lost for 10 frames
- **TENTATIVE -> discarded**: No matches within 5 frames (noise)

**Why geometry-first, not appearance-first**: Billboards are visually repetitive (white rectangles with text). Appearance embeddings have low discriminative power between different billboards. Position trajectory and size trend are much more reliable signals.

### 4.6 Frame Buffer Memory Management

**Problem**: At 8MP (3264x2448), one NV12 frame is ~12MB. Buffering 30 frames per track consumes 360MB per track, which is unacceptable on 8GB shared memory.

**Solution**: Two-tier storage.

**Tier 1: Global frame ring** (for detection/tracking at capture resolution)
- Fixed-size ring buffer of 16 NV12 frames at **1920x1080** (not full 8MP)
- Total memory: 16 x 3.1MB = ~50MB (hard cap at 200MB for safety margin)
- Refcounted: frames are released when no active track references them
- V4L2 capture writes into this ring; if ring is full, oldest unreferenced frame is evicted

**Tier 2: Per-track best crop** (for OCR, stored as compressed JPEG)
- Each active track stores ONE best crop as a compressed JPEG blob (~50-200KB)
- When a new frame scores higher on sharpness/size, the old crop is replaced
- At 1920x1080 capture, the crop is taken directly from the frame ring
- If full 8MP is needed for OCR, **capture at 1080p for tracking but trigger a single full-res capture on track finalization** (camera supports resolution switching per-frame via V4L2 `VIDIOC_S_FMT`). This is a v4.0 optimization.

**Memory accounting**:
- Frame ring: 50-200MB (configurable, hard cap)
- Per-track crops: ~200KB x 10 active tracks = 2MB (negligible)
- No per-track frame history. Metadata (sharpness, bbox, GPS) is stored, not pixels.

### 4.7 Storage Durability

**Design choice**: We accept losing **the most recent 1-2 observations** on sudden power loss, but guarantee that all previously acknowledged observations (returned 200 OK to the pipeline) are durable.

**Write sequence for each observation**:

```
1. Write JPEG crop to /data/surveys/{session}/{id}.jpg.tmp
2. fsync(fd) on the temp file
3. rename() to /data/surveys/{session}/{id}.jpg
4. fsync(dirfd) on the containing directory
5. INSERT INTO observations (...) in SQLite
   (SQLite configured: PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL)
```

`synchronous=FULL` ensures the WAL is fsynced on every commit. Combined with the file write sequence above, this guarantees that if a record appears in the DB, its image file exists and is complete.

**On startup recovery**:
- Scan for orphaned `.tmp` files -> delete (incomplete writes)
- Reconcile DB rows against image files -> mark rows with missing images as `damaged`
- Resume upload queue from last un-uploaded record

**Backpressure**: If SSD write latency exceeds 100ms (thermal throttling), reduce capture FPS rather than silently dropping observations. Alert via Prometheus metric.

### 4.8 Performance Budget

| Stage | Time (ms) | Frequency | GPU? |
|-------|----------|-----------|------|
| V4L2 MMAP dequeue | <1 | 10 FPS | No |
| NV12 -> RGB + resize 640x640 | 3-5 | Every frame | CPU |
| YOLO inference (FP16) | 30-40 | Every frame | Yes |
| Tracker FSM update | <1 | Every frame | No |
| Laplacian sharpness (crop) | 2-3 | Per detection | CPU |
| GPS interpolation | <1 | Every frame | No |
| **Stage 1 total** | **~38-50** | **~10 FPS** | |
| | | | |
| JPEG crop from ring | 1-2 | Per billboard | No |
| Qwen3-VL inference (llama.cpp) | 1,000-3,000 | Per billboard | Yes |
| JSON response parsing | <1 | Per billboard | No |
| Durable write (fsync chain) | 10-30 | Per billboard | No |
| **Stage 2 wall time** | **~1-3s** | **Spread across ~10-30 inter-frame gaps** | |

Stage 1 uses ~40ms of each 100ms frame budget. Stage 2 VLM inference is chunked into inter-frame GPU gaps via the scheduler. A single VLM call takes 1-3s wall time but does not block YOLO detection frames. In dense areas with many finalizing tracks, the VLM queue may grow -- jobs older than 10 entries are dropped (with log warning) to prevent unbounded backlog.

---

## 5. Camera Calibration

### 5.1 Intrinsic Calibration

Performed once per camera+lens using checkerboard:

```bash
python scripts/calibrate_camera.py \
    --device /dev/video0 \
    --pattern 9x6 \
    --square-size 25mm \
    --output config/calibration.json
```

Produces: fx, fy, cx, cy, k1, k2, p1, p2, k3.

### 5.2 Extrinsic Calibration

Camera-to-vehicle transform:

```json
{
    "intrinsics": { "fx": 1200.0, "fy": 1200.0, "cx": 960, "cy": 540, "distortion": [...] },
    "extrinsics": {
        "mount_position": "roof_right",
        "height_above_ground_m": 1.6,
        "yaw_offset_deg": 90.0,
        "pitch_offset_deg": -5.0
    }
}
```

### 5.3 Timestamp Sync

V4L2 provides `CLOCK_MONOTONIC` timestamps. gpsd provides GPS time. Offset calibrated at startup by comparing `CLOCK_MONOTONIC` to gpsd's system-clock-adjusted timestamps. Drift is typically <1ms over a survey run. No hardware PPS required for v3.0.

---

## 6. Camera Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| Resolution | 1920x1080 | Sufficient for billboard OCR at typical urban distances. Full 8MP deferred to v4.0. |
| Frame Rate | 10 FPS | 1.7m between frames at 60 km/h. Sufficient for tracking. |
| Exposure | Fixed 1/2000s | Freeze motion, prevent auto-exposure hunting |
| Gain | Auto, bounded 0-24 dB | Compensate for lighting |
| Focus | Fixed at hyperfocal distance | Everything from 2.5m to infinity sharp. No AF hunting. |
| White Balance | Fixed 5500K | Consistent. Adjust in post if needed. |

### 6.1 CPL Filter

Circular polarizer reduces billboard surface glare and specular reflections. Loses ~1.5 stops. Acceptable given gain headroom.

### 6.2 Operating Envelope

**Daylight profile (default)**: 1/2000s exposure, gain 0-24 dB, CPL on. Covers bright sun through overcast.

**Low-light profile** (switchable via config): 1/500s exposure, gain 0-36 dB, CPL off. For dusk and illuminated billboards at night. Increased motion blur (~4x vs daylight) partially mitigated by selecting sharpest crop. Not suitable for unlit signs.

**Hard scope limit for v3.0**: This system is designed for **daylight and dusk operation** with illuminated billboards. Unlit signs at night are explicitly out of scope. Night detection targets (>50%) in the test matrix apply only to LED/illuminated billboards.

### 6.3 Coverage Scope

**v3.0 covers one side of the road per survey pass.** The single camera is mounted on the right side, aimed at the right-side roadside. To inventory both sides, drive the route in both directions (or make a return pass on the opposite side).

This is a hard scope constraint of the single-camera prototype. Dual-camera support (left + right) is a v4.0 feature.

---

## 7. Environmental Test Matrix

| Condition | Pass Criteria |
|-----------|---------------|
| Daylight, clear, 40 km/h | >80% billboard detection rate |
| Daylight, overcast | >80% detection |
| Dusk (golden hour) | >70% detection |
| Night (illuminated/LED signs only) | >50% detection with low-light profile, OCR on lit signs. Unlit signs out of scope. |
| Light rain | >60% detection, note water droplet impact |
| Direct sun glare | Document failure modes, test CPL effectiveness |
| Highway, 100 km/h | >70% detection, no blur in crops |
| Urban canyon (GPS-challenged) | Dead reckoning maintains fix, >70% detection |
| Truck/tree occlusion | Tracker maintains identity through momentary occlusion |

---

## 8. Training the Billboard Model

### 8.1 Data Sources

| Source | Images | Notes |
|--------|--------|-------|
| Mapillary Vistas | ~5,000 | Billboard class only, global street-level |
| Open Images v7 | ~15,000 | Billboard annotations |
| Custom drive-by | 2,000+ | From own rig, representative of deployment domain |

### 8.2 Domain-Representative Sampling

Custom data sampled across: distance (10m/30m/80m+), angle (fronto-parallel/30/60 oblique), speed (0/30/60/100 km/h), lighting (morning/midday/dusk/night), weather (clear/overcast/rain), billboard type (highway/urban/poster/digital).

Evaluation sets split **by scenario**, not random split.

### 8.3 Iteration Strategy

1. **V1**: Public datasets. Deploy, collect real data. Expect mediocre recall.
2. **V2**: Add 1K+ annotated domain-specific frames from V1 false negatives. Target >80% recall.
3. **V3**: Active learning loop (flag low-confidence detections for review).

---

## 9. Geolocation Strategy

### V1: Observation Points

Stores the **car's GPS position and heading** at capture time. This is the observer location, NOT the billboard location. Explicitly documented in schema and dashboard.

### V2 (future): Billboard Position Estimation

Triangulate from multiple frames using camera intrinsics + extrinsics + bearing intersection. Cluster on server via PostGIS.

### Cross-Frame Dedup

Within a survey run: tracker assigns consistent IDs. Across runs: server-side PostGIS clustering (30m radius) + appearance similarity of best crops.

---

## 10. Web Dashboard

### API

```
POST   /api/v1/observations         # Upload batch from device
GET    /api/v1/observations         # List with spatial/temporal filters
GET    /api/v1/assets               # Clustered billboard assets
GET    /api/v1/assets/:id           # Asset with all observations
PUT    /api/v1/assets/:id           # Manual correction/merge
DELETE /api/v1/assets/:id           # Remove false positive
GET    /api/v1/surveys              # Survey run list
GET    /api/v1/export?format=geojson
```

### Tech Stack

- **Backend**: Hummingbird 2 (Swift)
- **Database**: PostgreSQL + PostGIS
- **Storage**: S3-compatible for crops
- **Frontend**: Leaflet + OpenStreetMap SPA

---

## 11. Deployment

### Operational Workflow

1. **Before survey**: Start car. System auto-boots. Check `http://jetson.local:9090/health`: camera feed, GPS fix, storage, temperature.
2. **During**: Drive. Dashboard shows detection count, FPS, GPS track, thermal status.
3. **After**: Connect WiFi/LTE. Upload queue drains. Review on web dashboard.

### WendyOS Commands

```bash
cd billboard-scanner/
wendy run --device jetson.local --detach
wendy device logs --app billboard-scanner --device jetson.local
```

---

## 12. Risk Register

| Risk | Mitigation |
|------|------------|
| Camera USB disconnect | V4L2 auto-reconnect, USB strain relief, watchdog restart |
| GPS signal loss | ZED-F9R dead reckoning via IMU. Log fix mode per observation. |
| Motion blur | 1/2000s fixed exposure, global shutter camera, quality gate in frame selector |
| Billboard glare | Exterior mount (no windshield), CPL filter |
| Rain on lens | IP67 housing, hydrophobic coating. Quality gate skips degraded frames. |
| Thermal throttling | Forced airflow, staged throttle policy (FPS -> OCR-off -> inference-off) |
| Power transient | Isolated DC-DC, supercap hold-up, fsync write chain, SQLite FULL sync |
| Model domain shift | Domain-representative training, active learning, per-scenario eval |
| SSD fills up | 500GB = ~50K observations. Alert at 80%. Auto-prune uploaded raw frames. |
| VLM too slow or OOM | Qwen3-VL-2B INT4 fits in ~1.5GB. If too slow on device, defer extraction to upload server. |
| Tracker fragmentation | Billboard FSM with 30-frame miss tolerance, geometry-first matching |
| GPU OOM | Measured memory budget, hard caps per engine, 1-at-a-time OCR semaphore |

---

## 13. Milestones

### Prototype Track

| Phase | Scope | Duration |
|-------|-------|----------|
| **P0: VLM Validation** | Qwen3.5-2B Q4_K_M via llama.cpp on Orin Nano: load time, inference latency, memory with YOLO co-resident, JSON output quality on sample billboard crops. | 2 days |
| **P1: Hardware Assembly** | Mount camera + Jetson + GPS in car, verify connections, power | 1 week |
| **P2: Capture Pipeline** | V4L2 MMAP capture, GPS integration, frame ring, status dashboard | 1 week |
| **P3: Detection** | Collect training data, annotate, train YOLO11s V1, TensorRT integration | 2 weeks |
| **P4: Tracker + Selection** | Billboard FSM tracker, best-crop store, per-track finalization | 1 week |
| **P5: VLM + Upload** | Qwen3-VL integration, durable storage, upload queue | 1 week |
| **P6: Web Dashboard** | Map, observation list, CRUD, export | 1 week |
| **P7: Field Testing** | 500+ km survey, tune thresholds, retrain V2 | 2 weeks |

### Production Track

| Phase | Scope | Duration |
|-------|-------|----------|
| **R1: Environmental Testing** | Full test matrix, document failure modes | 2 weeks |
| **R2: Reliability Hardening** | Power transient testing, thermal soak, 8-hour continuous run | 2 weeks |
| **R3: Model V3** | Active learning, per-scenario eval | 2 weeks |
| **R4: Server Clustering** | PostGIS dedup, multi-run asset management | 1 week |
| **R5: Fleet Support** | Multi-vehicle provisioning, OTA model updates | 2 weeks |

---

## 14. Resolved Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Still capture vs video | **Continuous video** | Tracking, best-frame selection, occlusion handling |
| DSLR vs industrial camera | **Industrial USB3** | Reliable, no tethering, global shutter, fixed focus |
| CSI vs USB | **USB3 V4L2 only (v3.0)** | Simpler pipeline, no Argus/NVMM complexity |
| Traditional OCR vs VLM | **VLM (Qwen3.5-2B)** | Answers "what company?" directly, handles logos/stylized text, no ONNX export risk. See Appendix A. |
| VLM runtime | **llama.cpp (GGUF Q4_K_M)** | Proven on aarch64 CUDA, bounded memory, simple deployment. TRT-LLM is faster but untested on Orin Nano 8GB. |
| OCR every frame vs best-frame | **Best-crop per track** | Less GPU, better results from sharpest frame |
| Observer vs billboard location | **Observer point (V1)** | Billboard triangulation deferred to server-side V2 |
| Multi-class vs single-class | **Single class: billboard** | Reduces confusion, higher precision |
| Exterior vs interior mount | **Exterior IP67** | No windshield artifacts |
| Frame buffer design | **Global ring + per-track JPEG crop** | Bounded memory, no per-track frame accumulation |
| SQLite sync mode | **FULL** | True durability on power loss, accept ~1ms per write overhead |
| Tracker design | **Billboard FSM** | Geometry-first, long miss tolerance, area-trend finalization |
| GPU arbitration | **Chunked OCR with frame-level admission** | Non-preemptible TensorRT calls require explicit chunking, not just semaphore |
| Coverage per pass | **One side of road** | Single camera on right side. Drive both directions for full coverage. |
| Night operation | **Illuminated signs only (v3.0)** | Unlit signs out of scope. Low-light profile for dusk/LED boards. |

---

## 15. Open Questions

1. **Dual cameras**: Left + right for complete coverage? Doubles bandwidth + inference.
2. **Night viability**: LED billboards easy, unlit signs need longer exposure.
3. **Billboard face association**: Front/back of same structure -- associate via opposite headings on server?
4. **Offline batch refinement**: Re-run heavier OCR on stored crops using desktop GPU?
5. **Logo recognition**: When to add dedicated model for non-text brand identification?
6. **Full 8MP capture**: Trigger full-res capture on track finalization for better OCR? (future version)

---

## Appendix A: Traditional OCR vs VLM -- Evaluation

We evaluated three approaches for extracting text and brand information from billboard crops. This appendix documents the tradeoffs that led to choosing Qwen3.5-2B (VLM) over traditional OCR.

### Options Evaluated

| | PaddleOCR PP-OCRv4 | EasyOCR | Qwen3.5-2B (Q4_K_M GGUF) |
|---|---|---|---|
| **Architecture** | Det (DB) + Rec (SVTR), two-stage | CRAFT det + CRNN rec | Natively multimodal VLM (vision + language fused), single-stage |
| **Latency per crop** | ~80-120ms (TensorRT) | ~200-500ms (PyTorch) | ~1-3s (llama.cpp) |
| **GPU memory** | ~100MB (det + rec engines) | ~300MB (PyTorch) | ~1,500-2,000MB (Q4 quantized) |
| **Integration effort** | High: ONNX export, static shapes, TensorRT build on aarch64, separate det/rec/postprocess | Medium: pip install, PyTorch CUDA | Low: llama.cpp binary, GGUF download, already in deepstream-vision stack |
| **aarch64 TensorRT risk** | **HIGH**: Dynamic shapes, gather ops, reshape nodes often fail TensorRT export. Requires mandatory feasibility spike. | Low (PyTorch, not TensorRT) | None (llama.cpp, not TensorRT) |
| **Stylized text** | Poor: OCR pipelines struggle with artistic fonts, gradient fills, drop shadows | Poor: same limitation | **Good**: VLMs understand visual context, not just character shapes |
| **Logo/brand recognition** | None: only reads text | None | **Built-in**: "What company is this?" works even for logo-only billboards |
| **Contextual understanding** | None: returns raw character sequences | None | **Yes**: understands "this is a Coca-Cola ad" even if text is partially occluded |
| **Multi-language** | Good (PP-OCRv4 supports 80+ languages) | Good (80+ languages) | Excellent (Qwen3.5 supports 201+ languages) |
| **Output format** | Raw text lines + bounding boxes. Requires separate regex/brand matching pipeline. | Raw text lines | Structured JSON with company, URL, phone, description. One call replaces det + rec + regex + brand matching. |
| **Tool calling / structured output** | N/A | N/A | Native support (Qwen3.5 has built-in tool calling and JSON mode) |

### Decision Matrix

| Criterion | Weight | PaddleOCR | EasyOCR | Qwen3.5-2B |
|-----------|--------|-----------|---------|----------|
| Integration risk on Jetson aarch64 | Critical | FAIL (TensorRT export) | OK | OK |
| Answers the actual question (company, URL) | High | Partial (needs post-processing) | Partial | Full |
| Handles logos and stylized text | High | Poor | Poor | Good |
| Latency (acceptable: <5s per billboard) | Medium | Best (~100ms) | Good (~300ms) | Acceptable (~2s) |
| GPU memory on 8GB shared | Medium | Best (~100MB) | OK (~300MB) | Acceptable (~1.5-2GB) |
| Already in our stack | Low | No | No | Yes (vlm service) |

### Why VLM Wins

1. **Eliminates the #1 risk**: PaddleOCR on aarch64 TensorRT was flagged as the highest integration risk across all 3 external review rounds. VLM via llama.cpp sidesteps TensorRT export entirely.

2. **Answers the real question**: We don't want "raw characters on a billboard." We want "what company is advertising, and how to find them." A VLM answers this directly. PaddleOCR requires OCR + regex + brand database + fuzzy matching -- four components vs. one.

3. **Handles the hard cases**: Billboards are designed to be visually distinctive, not OCR-friendly. Stylized fonts, logos, gradients, and perspective distortion are the norm. VLMs handle these; traditional OCR pipelines don't.

4. **Latency is fine**: The design runs text extraction once per physical billboard (on track finalization), not per frame. 1-3 seconds is well within the latency budget.

5. **Memory is fine**: ~2GB for the VLM + ~200MB for YOLO = ~2.2GB peak on an 8GB device. Comfortable.

### Why Qwen3.5-2B Specifically

- **Natively multimodal**: No separate "-VL" variant needed. Vision is fused into the base model.
- **MMMU 64.2**: Strong vision-language understanding at only 2B parameters.
- **Native tool calling + JSON output**: Built into the model, not bolted on.
- **262K context window**: More than enough for billboard crop + prompt.
- **Gated DeltaNet architecture**: More memory-efficient attention, important on 8GB shared memory.
- **GGUF widely available**: unsloth and bartowski publish Q4_K_M quantizations with 293K+ downloads.
- **Apache 2.0 license**: No restrictions on commercial use.

### VLM Runtime Options

| Runtime | Tensor core usage | Memory overhead | Jetson Orin Nano 8GB | Status |
|---|---|---|---|---|
| **llama.cpp** | Partial (cuBLAS GEMMs, not fused kernels) | Low, predictable | **Works** | Chosen for v1 |
| **SGLang / vLLM** | Better (custom CUDA kernels) | Medium (Python runtime + scheduler) | Possible, tight | Alternative if llama.cpp too slow |
| **TensorRT-LLM** | **Best** (hardware-specific fused kernels, full tensor core utilization) | High (paged KV cache manager, inflight batching system) | **Not tested** -- Jetson branch only validated on AGX Orin 64GB | Future upgrade path |

llama.cpp does not fully exploit tensor cores (uses generic cuBLAS, not SM87-optimized fused kernels). On datacenter GPUs, TRT-LLM is typically 2-3x faster. On Orin Nano, the gap would be smaller (memory-bandwidth-bound at 102 GB/s) but still meaningful -- estimated 30-50% faster decode. However, TRT-LLM's runtime memory overhead and untested Orin Nano support make it a poor fit for v1. Since we only run VLM ~1-5 times per minute, llama.cpp performance is sufficient. TRT-LLM becomes the upgrade path if NVIDIA ships proper Orin Nano support or if billboard density requires faster extraction.

### When to Reconsider

- **TRT-LLM on Orin Nano**: If NVIDIA ships a validated Orin Nano TRT-LLM build with Qwen3.5 support, switch for the speed improvement.
- **Hybrid OCR + VLM**: If billboard density exceeds ~20 finalizations per minute, a fast OCR pre-filter (for easy text) + VLM (for logos/ambiguous cases) could reduce average latency.
- **Larger VLM**: On Orin NX 16GB, Qwen3.5-4B or even 9B would significantly improve extraction quality.
- **Qwen3.5-0.8B**: If memory pressure becomes an issue, the 0.8B variant fits in ~1GB but with noticeably worse vision benchmarks (MMMU 49 vs 64). Only suitable as an emergency fallback.
