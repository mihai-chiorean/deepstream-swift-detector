// nvds_shim.c
// Implementation of the DeepStream NvDsBatchMeta extraction shim.
//
// Compiled as a regular C file — NOT a header-only inline implementation —
// so that DeepStream types are resolved at build time against the vendored
// headers in Sources/CGStreamer/vendor/ (or the -I path set in Package.swift).
//
// Build note: nvdsgst_meta and nvds_meta are linked via the CGStreamer
// modulemap; see Package.swift for the .linkedLibrary directives.

#include "nvds_shim.h"

// DeepStream headers live in vendor/ (populated by the Dockerfile builder
// stage — see vendor/README.md). The -I flag pointing to this vendor dir
// is set via unsafeFlags in Package.swift.
#include <nvdsmeta.h>
#include <gstnvdsmeta.h>
#include <nvds_latency_meta.h>

#include <string.h>   // strncmp, strstr

// ---------------------------------------------------------------------------
// wendy_nvds_flatten
// ---------------------------------------------------------------------------

int wendy_nvds_flatten(GstBuffer *buf, WendyDetection *out, int maxOut) {
    if (!buf || !out || maxOut <= 0) return 0;

    NvDsBatchMeta *bm = gst_buffer_get_nvds_batch_meta(buf);
    if (!bm) return 0;

    int count = 0;

    for (NvDsMetaList *fl = bm->frame_meta_list;
         fl != NULL && count < maxOut;
         fl = fl->next)
    {
        NvDsFrameMeta *fm = (NvDsFrameMeta *)fl->data;
        if (!fm) continue;

        for (NvDsMetaList *ol = fm->obj_meta_list;
             ol != NULL && count < maxOut;
             ol = ol->next)
        {
            NvDsObjectMeta *om = (NvDsObjectMeta *)ol->data;
            if (!om) continue;

            out[count].classId    = om->class_id;
            out[count].confidence = om->confidence;
            out[count].left       = om->rect_params.left;
            out[count].top        = om->rect_params.top;
            out[count].width      = om->rect_params.width;
            out[count].height     = om->rect_params.height;
            // object_id is UINT64_MAX when the tracker has not yet assigned an ID.
            // Normalise to 0 so Swift can test trackerId == 0 for "untracked".
            out[count].trackerId  = (om->object_id == G_MAXUINT64) ? 0 : om->object_id;
            out[count].frameNum   = fm->frame_num;
            count++;
        }
    }

    return count;
}

// ---------------------------------------------------------------------------
// wendy_extract_frame_timing
//
// Walk batch_user_meta_list looking for NVDS_LATENCY_MEASUREMENT_META entries.
// Each component that enabled latency measurement attaches one such entry with
// its component_name and in/out_system_timestamp (wall-clock seconds as double).
//
// Component name conventions observed in DS 7.1 source:
//   - Decoder:      whatever was passed to nvds_add_reference_timestamp_meta()
//                   by nvv4l2decoder — typically "nvv4l2decoder" or the GstElement
//                   name. We match any name containing "decoder" (case-sensitive
//                   substring, which covers "nvv4l2decoder").
//   - nvstreammux:  "nvstreammux-<gst-element-name>", e.g. "nvstreammux-m"
//   - nvinfer:      GST_ELEMENT_NAME — typically "nvinfer0" or "nvinfer"
//   - nvtracker:    GST_ELEMENT_NAME — "wendy_tracker" in our pipeline
//
// Latency for each component = (out_system_timestamp - in_system_timestamp) * 1000
// (timestamps are in seconds; result in milliseconds).
//
// Bucket mapping:
//   decode_ms      → component_name contains "decoder"
//   streammux_ms   → component_name starts with "nvstreammux"
//   inference_ms   → component_name starts with "nvinfer"
//   preprocess_ms  → nvstreammux + nvinfer sum (backwards compat)
//   postprocess_ms → component_name starts with "wendy_tracker" OR starts with "nvtracker"
//
// PTS extraction:
//   ptsNs          → GST_BUFFER_PTS(buf); -1 if GST_CLOCK_TIME_NONE
// ---------------------------------------------------------------------------

WendyFrameTiming wendy_extract_frame_timing(GstBuffer *buf) {
    WendyFrameTiming timing = { 0.0, 0.0, 0.0, 0.0, 0.0, -1 };

    if (!buf) return timing;

    // Extract buffer PTS (nanoseconds since stream start).
    // GST_CLOCK_TIME_NONE is all-bits-set (UINT64_MAX); report as -1.
    GstClockTime pts = GST_BUFFER_PTS(buf);
    timing.ptsNs = GST_CLOCK_TIME_IS_VALID(pts) ? (int64_t)pts : -1;

    NvDsBatchMeta *bm = gst_buffer_get_nvds_batch_meta(buf);
    if (!bm) return timing;

    for (NvDsMetaList *ul = bm->batch_user_meta_list;
         ul != NULL;
         ul = ul->next)
    {
        NvDsUserMeta *um = (NvDsUserMeta *)ul->data;
        if (!um) continue;
        if (um->base_meta.meta_type != NVDS_LATENCY_MEASUREMENT_META) continue;

        NvDsMetaCompLatency *cl = (NvDsMetaCompLatency *)um->user_meta_data;
        if (!cl) continue;

        // Guard: out must be >= in (clock anomaly / wrap check).
        if (cl->out_system_timestamp <= cl->in_system_timestamp) continue;

        // DS 7.1 stores in/out_system_timestamp as milliseconds-since-epoch
        // encoded in a gdouble (g_get_real_time() returns µs; DS converts to
        // ms before storing). Diff is already in ms — no scaling needed.
        double latency_ms = cl->out_system_timestamp - cl->in_system_timestamp;
        const char *name = cl->component_name;

        if (strstr(name, "decoder") != NULL) {
            // Covers "nvv4l2decoder", "nvv4l2decoder0", etc.
            timing.decode_ms += latency_ms;
        } else if (strncmp(name, "nvstreammux", 11) == 0) {
            // nvstreammux-<name>: buffer batching + pad synchronisation.
            timing.streammux_ms += latency_ms;
            timing.preprocess_ms += latency_ms;
        } else if (strncmp(name, "nvinfer", 7) == 0) {
            // nvinfer / nvinfer0: letterbox resize + YOLO26n forward pass.
            timing.inference_ms += latency_ms;
            timing.preprocess_ms += latency_ms;
        } else if (strncmp(name, "wendy_tracker", 13) == 0 ||
                   strncmp(name, "nvtracker", 9) == 0) {
            // nvtracker runs track management + the custom bbox-parser output.
            timing.postprocess_ms += latency_ms;
        }
        // Other components (nvdsosd, nvjpegenc, etc.) are intentionally ignored.
    }

    return timing;
}

// ---------------------------------------------------------------------------
// Probe callback (called on the GStreamer streaming thread)
// ---------------------------------------------------------------------------

typedef struct {
    WendyDetectionCallback  cb;
    void                   *userData;
    GstElement             *element;          // reserved for future use
    uint64_t                lastProbeMonoNs;  // monotonic timestamp of last probe
} ProbeContext;

static GstPadProbeReturn probe_callback(GstPad        *pad,
                                         GstPadProbeInfo *info,
                                         gpointer        user_data)
{
    (void)pad;

    ProbeContext *ctx = (ProbeContext *)user_data;
    if (!ctx || !ctx->cb) return GST_PAD_PROBE_OK;

    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    // Inter-frame interval: monotonic time since the last probe callback.
    // Preferred over (running_time - buffer.PTS) because rtspsrc tags live
    // buffers with RTP-derived PTS that's typically offset from the pipeline
    // base_time by several seconds (the stream may have started before the
    // pipeline). At 20 fps this reads ~50 ms steady-state; spikes reveal
    // pipeline stalls or decode jitter.
    uint64_t nowMonoNs = (uint64_t)gst_util_get_timestamp();
    uint64_t latencyNs = 0;
    if (ctx->lastProbeMonoNs != 0 && nowMonoNs > ctx->lastProbeMonoNs) {
        latencyNs = nowMonoNs - ctx->lastProbeMonoNs;
    }
    ctx->lastProbeMonoNs = nowMonoNs;

    // Extract per-component latency from DS batch user metas.
    // Returns all-zeros if NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT is not set.
    // Also extracts GST_BUFFER_PTS into timing.ptsNs (-1 if no PTS).
    WendyFrameTiming timing = wendy_extract_frame_timing(buf);

    // Stack-allocate enough room for 300 detections (~16 KB — safe on any
    // platform stack). YOLO26n emits at most 300 objects per batch.
    WendyDetection dets[300];
    int n = wendy_nvds_flatten(buf, dets, 300);

    // Always invoke the callback, even when n == 0. The Swift side tracks
    // ID disappearance frame-to-frame; a zero-detection frame is meaningful.
    ctx->cb(n, dets, latencyNs, timing, ctx->userData);

    return GST_PAD_PROBE_OK;
}

// ---------------------------------------------------------------------------
// wendy_install_detection_probe
// ---------------------------------------------------------------------------

gulong wendy_install_detection_probe(GstElement *element,
                                      const char *padName,
                                      WendyDetectionCallback cb,
                                      void *userData)
{
    if (!element || !padName || !cb) return 0;

    GstPad *pad = gst_element_get_static_pad(element, padName);
    if (!pad) return 0;

    ProbeContext *ctx = (ProbeContext *)g_malloc0(sizeof(ProbeContext));
    ctx->cb       = cb;
    ctx->userData = userData;
    ctx->element  = element;  // not ref'd — pipeline owns it, probe tied to pipeline lifetime
    ctx->lastProbeMonoNs = 0;

    // ProbeContext is intentionally never freed: the probe lives for the
    // lifetime of the pipeline. If callers need cleanup, they should remove
    // the probe via gst_pad_remove_probe and call g_free(ctx) afterward.
    // For the detector's single-pipeline lifecycle this is fine.
    gulong id = gst_pad_add_probe(
        pad,
        GST_PAD_PROBE_TYPE_BUFFER,
        probe_callback,
        ctx,
        NULL   // GDestroyNotify — intentionally NULL, see above
    );

    gst_object_unref(pad);
    return id;
}
