// nvds_shim.h
// C shim for DeepStream NvDsBatchMeta pad-probe extraction.
//
// Exposes flat structs and two functions so Swift never touches GList,
// NvDsBatchMeta, NvDsFrameMeta, or NvDsObjectMeta directly.
//
// DeepStream headers (nvdsmeta.h, gstnvdsmeta.h) are provided by the
// WendyOS SDK sysroot (wendyos-device.sdk/usr/include/) and resolved
// automatically via the sysroot path. No vendor/ copy is required.

#pragma once

#include <gst/gst.h>
#include <nvdsmeta.h>
#include <gstnvdsmeta.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// WendyDetection — flat struct that crosses the Swift/C boundary.
//
// Coordinates are in source-frame pixel space as reported by nvtracker.
// trackerId is 0 when the object has not yet been assigned a tracker ID
// (can happen on the first frame a new object appears before nvtracker
// confirms it — rare in practice but defensive).
// ---------------------------------------------------------------------------

typedef struct {
    int      classId;
    float    confidence;
    float    left;        // source-frame pixel space (top-left origin)
    float    top;
    float    width;
    float    height;
    uint64_t trackerId;   // 0 if not yet tracker-confirmed
    int      frameNum;
} WendyDetection;

// ---------------------------------------------------------------------------
// WendyFrameTiming — per-frame component latency extracted from DeepStream's
// NVDS_LATENCY_MEASUREMENT_META user metas on the batch.
//
// Requires NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1 in the environment.
// Fields are 0.0 when the corresponding component did not emit a meta for
// this frame (e.g., NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT not set, or
// the component was skipped).
//
// Bucket mapping:
//   decode_ms      — nvv4l2decoder: decoder input → decoder output.
//                    Attached by nvstreammux when it reads the decoder's
//                    GstReferenceTimestampMeta.
//   streammux_ms   — nvstreammux only: mux enter → mux exit (buffer batching,
//                    waiting for batch to fill, pad synchronisation).
//   inference_ms   — nvinfer only: nvinfer enter → nvinfer exit (letterbox
//                    resize + YOLO26n forward pass + custom bbox-parser).
//   preprocess_ms  — nvstreammux + nvinfer sum (= streammux_ms + inference_ms).
//                    Kept for backwards compatibility; monitor.html reads this
//                    field for "Inference latency" until the new names are live.
//   postprocess_ms — nvtracker: tracker enter → tracker exit.
//                    The custom bbox parser .so runs inside nvinfer's
//                    postprocess step, but DS does not break that out as a
//                    separate component meta. nvtracker is the closest
//                    downstream component and captures the final association
//                    + track-management cost.
//   ptsNs          — GStreamer buffer PTS (presentation timestamp) in
//                    nanoseconds, as reported by GST_BUFFER_PTS(buf).
//                    GST_CLOCK_TIME_NONE (all bits set) means no PTS —
//                    reported as -1 so Swift can check ptsNs < 0.
//                    Used by Stage 2 for frame-accurate bbox overlay sync.
// ---------------------------------------------------------------------------

typedef struct {
    double  decode_ms;       // nvv4l2decoder latency (ms); 0 if unavailable
    double  streammux_ms;    // nvstreammux latency (ms); 0 if unavailable
    double  inference_ms;    // nvinfer latency (ms); 0 if unavailable
    double  preprocess_ms;   // nvstreammux + nvinfer sum (ms); kept for compat
    double  postprocess_ms;  // nvtracker latency (ms); 0 if unavailable
    int64_t ptsNs;           // buffer PTS in ns; -1 if GST_CLOCK_TIME_NONE
} WendyFrameTiming;

// ---------------------------------------------------------------------------
// wendy_nvds_flatten
//
// Walk NvDsBatchMeta from a GstBuffer produced by the nvtracker src pad.
// Fill `out[0..count-1]` with one entry per object meta. Returns the number
// of detections written. Never calls gst_buffer_map. Thread-safe (reads only
// immutable GstMeta attached to the buffer).
//
// Typical max: 300 for YOLO26n (matches `max-objects` in nvinfer_config.txt).
// A stack buffer of 300 * sizeof(WendyDetection) ≈ 16 KB is sufficient.
// ---------------------------------------------------------------------------

int wendy_nvds_flatten(GstBuffer *buf, WendyDetection *out, int maxOut);

// ---------------------------------------------------------------------------
// wendy_extract_frame_timing
//
// Walk the batch-level user meta list of the NvDsBatchMeta attached to `buf`
// and accumulate per-component latencies into `timing`. Requires the env var
// NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1; returns with all fields 0.0
// if the metas are absent (env not set or DS version does not support it).
//
// Also extracts GST_BUFFER_PTS(buf) and stores it as ptsNs. Returns -1 if
// the buffer has no PTS (GST_CLOCK_TIME_NONE).
//
// The function never calls gst_buffer_map and is safe to call from the
// GStreamer streaming thread alongside wendy_nvds_flatten.
// ---------------------------------------------------------------------------

WendyFrameTiming wendy_extract_frame_timing(GstBuffer *buf);

// ---------------------------------------------------------------------------
// WendyDetectionCallback
//
// Called from the GStreamer streaming thread — must NOT block and must NOT
// touch Swift actor state directly. Copy the detections into a Swift-owned
// value type before bouncing via Task.detached or continuation.yield.
//
// timing carries per-component latency data. All fields may be 0 if
// NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT is not set. timing.ptsNs is
// -1 if the buffer carried no PTS.
// ---------------------------------------------------------------------------

typedef void (*WendyDetectionCallback)(int count,
                                       const WendyDetection *dets,
                                       uint64_t frameLatencyNs,
                                       WendyFrameTiming timing,
                                       void *userData);

// ---------------------------------------------------------------------------
// wendy_install_detection_probe
//
// Install a GST_PAD_PROBE_TYPE_BUFFER probe on `padName` of `element`.
// The probe calls `cb(count, dets, frameLatencyNs, timing, userData)` for
// every buffer that carries NvDsBatchMeta. Returns the probe ID (pass to
// gst_pad_remove_probe to uninstall). Returns 0 on failure (pad not found).
// ---------------------------------------------------------------------------

gulong wendy_install_detection_probe(GstElement *element,
                                     const char *padName,
                                     WendyDetectionCallback cb,
                                     void *userData);
