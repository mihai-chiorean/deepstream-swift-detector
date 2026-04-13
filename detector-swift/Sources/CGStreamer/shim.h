// CGStreamer/shim.h
// C helper functions for the parts of GStreamer that don't import to Swift cleanly:
//   - GLib's GST_BIN/GST_APP_SINK casts (macros + void* casts)
//   - GstSample/GstBuffer extraction (returns through out-params for ergonomic Swift)
//
// Swift can call gst_init, gst_parse_launch, gst_element_set_state, etc. directly
// via the imported GStreamer headers — this shim only fills the gaps.

#pragma once

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

// Cast helpers — GStreamer's GST_BIN(x), GST_APP_SINK(x) etc. are C macros
// that expand to G_TYPE_CHECK_INSTANCE_CAST which Swift can't import.

static inline GstBin *wendy_gst_bin_cast(GstElement *element) {
    return GST_BIN(element);
}

static inline GstAppSink *wendy_gst_app_sink_cast(GstElement *element) {
    return GST_APP_SINK(element);
}

static inline GstElement *wendy_gst_app_sink_to_element(GstAppSink *sink) {
    return GST_ELEMENT(sink);
}

// Add an extra directory to the GStreamer plugin registry and immediately scan
// it for plugins.  Call this after gst_init() to ensure plugins installed in
// non-default paths (e.g. Ubuntu aarch64 packages) are discovered alongside the
// JetPack/deepstream plugins that the container runtime injects via GST_PLUGIN_PATH.
//
// Returns the number of plugins found in `path` (may be 0 if already cached).
static inline gint wendy_gst_registry_scan_path(const gchar *path) {
    GstRegistry *registry = gst_registry_get();
    return gst_registry_scan_path(registry, path);
}

// Pull a sample, extract its raw byte buffer, and return it through out-params.
//
// On success returns 1 and fills `out_data`, `out_size`. The caller must call
// `wendy_gst_release_sample` with the returned `out_sample` handle to free it.
// On EOS or error returns 0; out-params are zeroed.
//
// The returned data pointer is only valid until `wendy_gst_release_sample` is
// called — copy the bytes into Swift-owned memory before releasing.
static inline int wendy_gst_pull_sample(
    GstAppSink *sink,
    void **out_sample,
    void **out_data,
    size_t *out_size
) {
    *out_sample = NULL;
    *out_data = NULL;
    *out_size = 0;

    GstSample *sample = gst_app_sink_pull_sample(sink);
    if (!sample) return 0;

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        gst_sample_unref(sample);
        return 0;
    }

    // Map the buffer for reading. Allocate a small struct to hold the
    // GstSample + GstMapInfo until release.
    typedef struct {
        GstSample *sample;
        GstMapInfo map;
    } SampleHandle;

    SampleHandle *handle = (SampleHandle *)g_malloc(sizeof(SampleHandle));
    handle->sample = sample;

    if (!gst_buffer_map(buffer, &handle->map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        g_free(handle);
        return 0;
    }

    *out_sample = handle;
    *out_data = handle->map.data;
    *out_size = handle->map.size;
    return 1;
}

// Release a sample handle returned by wendy_gst_pull_sample.
static inline void wendy_gst_release_sample(void *sample_handle) {
    if (!sample_handle) return;
    typedef struct {
        GstSample *sample;
        GstMapInfo map;
    } SampleHandle;

    SampleHandle *handle = (SampleHandle *)sample_handle;
    GstBuffer *buffer = gst_sample_get_buffer(handle->sample);
    gst_buffer_unmap(buffer, &handle->map);
    gst_sample_unref(handle->sample);
    g_free(handle);
}

// Get the negotiated caps width/height from a sample.
//
// Returns 1 on success, 0 if caps are missing or malformed.
static inline int wendy_gst_get_sample_dimensions(
    GstAppSink *sink,
    int *out_width,
    int *out_height
) {
    *out_width = 0;
    *out_height = 0;

    GstCaps *caps = gst_app_sink_get_caps(sink);
    if (!caps) return 0;
    if (gst_caps_get_size(caps) == 0) {
        gst_caps_unref(caps);
        return 0;
    }

    GstStructure *structure = gst_caps_get_structure(caps, 0);
    int width = 0, height = 0;
    gboolean okw = gst_structure_get_int(structure, "width", &width);
    gboolean okh = gst_structure_get_int(structure, "height", &height);

    gst_caps_unref(caps);

    if (!okw || !okh) return 0;
    *out_width = width;
    *out_height = height;
    return 1;
}
