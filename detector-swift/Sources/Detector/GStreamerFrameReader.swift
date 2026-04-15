// GStreamerFrameReader.swift
// DeepStream-native RTSP detection pipeline using nvinfer + nvtracker.
//
// Pipeline (single rtspsrc, tee-split):
//   rtspsrc → rtph264depay → h264parse
//     → nvv4l2decoder
//     → tee name=t
//       t. → queue ! nvstreammux ! nvinfer ! nvtracker ! fakesink   (detection)
//       t. → queue max-size-buffers=2 leaky=downstream              (MJPEG sidecar)
//             ! nvvideoconvert ! nvjpegenc quality=85
//             ! appsink name=mjpeg_sink emit-signals=false sync=false
//                        max-buffers=1 drop=true
//
// Detection branch: pad probe on nvtracker src pad. No gst_buffer_map —
// no NVMM pinning, no memory leak.
//
// MJPEG branch: nvjpegenc produces system-memory JPEG buffers (~20 KB).
// gst_buffer_map on system-memory buffers is cheap (malloc pages, not
// NVMM pool). The branch always runs but the appsink drops frames when
// no client is pulling — effectively idle when /stream has no viewers.
//
// Single rtspsrc: avoids the camera's 1-connection limit. The tee splits
// the decoded NVMM stream after nvv4l2decoder so both branches share one
// RTSP/H.264 decode path.
//
// Concurrency contract:
//   The GStreamer streaming thread calls the @convention(c) probe callback.
//   The callback captures a DetectionStream via Unmanaged (retained at
//   install, released in stop()). AsyncStream.Continuation.yield is safe
//   from any thread.

import CGStreamer
import Logging

#if canImport(Glibc)
    import Glibc
#elseif canImport(Musl)
    import Musl
#endif

// MARK: - Detection

/// A single object detection produced by nvinfer + nvtracker.
///
/// Coordinates are in source-frame pixel space (top-left origin), as reported
/// by nvtracker. trackId is nil when the object has not yet been confirmed by
/// the tracker (trackerId == 0 from the C shim).
struct Detection: Sendable {
    /// Top-left corner x coordinate (source-frame pixels).
    var x: Float
    /// Top-left corner y coordinate (source-frame pixels).
    var y: Float
    /// Bounding box width (pixels).
    var width: Float
    /// Bounding box height (pixels).
    var height: Float
    /// COCO class index (0-based), as reported by nvinfer + custom bbox parser.
    var classId: Int
    /// Detection confidence score (0–1), from the YOLO26 custom bbox parser.
    var confidence: Float
    /// Optional tracker-assigned identity; nil when trackerId == 0 from C shim.
    var trackId: Int?
    /// Pipeline frame number (monotonically increasing per stream).
    var frameNum: Int
}

// MARK: - DetectionStream

/// Bridges the C pad-probe callback to an AsyncStream<[Detection]>.
///
/// The instance is heap-allocated and retained for the lifetime of the
/// GStreamer pipeline via `Unmanaged.passRetained`. Calling `finish()` ends
/// the stream; `ingest` feeds detections from the streaming thread.
///
/// Swift 6 note: `continuation` is Sendable; `ingest` can be called safely
/// from the GStreamer streaming thread without actor hops. The class is
/// marked @unchecked Sendable to cross the isolation boundary.
final class DetectionStream: @unchecked Sendable {

    private let continuation: AsyncStream<[Detection]>.Continuation
    let stream: AsyncStream<[Detection]>

    init() {
        var cap: AsyncStream<[Detection]>.Continuation!
        self.stream = AsyncStream(bufferingPolicy: .bufferingNewest(4)) { cap = $0 }
        self.continuation = cap
    }

    /// Called from the C probe callback on the GStreamer streaming thread.
    /// Safe to call from any thread — AsyncStream.Continuation is Sendable.
    func ingest(_ detections: [Detection]) {
        continuation.yield(detections)
    }

    /// Finish the stream (called on pipeline teardown).
    func finish() {
        continuation.finish()
    }
}

// MARK: - GstElementRef

/// @unchecked Sendable wrapper for a GstElement* returned from the pipeline.
///
/// Safety: the pointee is a GStreamer ref-counted C object owned by the pipeline
/// (GStreamerFrameReader holds a gst_object_ref'd copy). The pointer is valid
/// for the lifetime of the pipeline and is released in GStreamerFrameReader.cleanup().
/// Callers MUST NOT call gst_object_unref on the pointer they receive — it is owned
/// by GStreamerFrameReader.
struct GstElementRef: @unchecked Sendable {
    let element: UnsafeMutablePointer<GstElement>
}

// MARK: - @convention(c) probe entry point

/// GStreamer streaming-thread callback. Must be a free function with C linkage.
///
/// userData is a retained `DetectionStream` pointer (Unmanaged.passRetained
/// in `GStreamerFrameReader.start()`). We use `takeUnretainedValue()` because
/// the object is retained until `stop()` calls `release()` on the stored box.
@_cdecl("wendy_detection_probe_entry")
func wendyDetectionProbeEntry(
    count: Int32,
    dets: UnsafePointer<WendyDetection>?,
    userData: UnsafeMutableRawPointer?
) {
    guard let userData else { return }
    let ds = Unmanaged<DetectionStream>.fromOpaque(userData).takeUnretainedValue()

    // Build a Swift-owned copy of the detection array so no C pointer escapes
    // this function. The C probe callback stack frame is about to be reclaimed.
    let n = Int(count)
    var copy = [Detection]()
    copy.reserveCapacity(n)

    if let dets {
        for i in 0 ..< n {
            let d = dets[i]
            copy.append(Detection(
                x: d.left,
                y: d.top,
                width: d.width,
                height: d.height,
                classId: Int(d.classId),
                confidence: d.confidence,
                trackId: d.trackerId == 0 ? nil : Int(d.trackerId),
                frameNum: Int(d.frameNum)
            ))
        }
    }

    ds.ingest(copy)
}

// MARK: - GStreamerFrameReader

/// Actor that owns a DeepStream GStreamer pipeline and exposes detections as
/// an `AsyncStream<[Detection]>`.
///
/// The pipeline includes a tee that feeds both the detection branch (nvinfer +
/// nvtracker + fakesink) and the MJPEG branch (nvjpegenc + appsink). Both
/// branches share a single rtspsrc/nvv4l2decoder, avoiding the camera's
/// one-connection limit.
actor GStreamerFrameReader {

    // MARK: Configuration

    let stream: StreamConfig

    // MARK: Pipeline state

    /// `GstElement *` for the parsed pipeline. NULL until `start()` succeeds.
    private var pipeline: UnsafeMutablePointer<GstElement>?

    /// `GstElement *` for the nvtracker element (probe target).
    private var nvtrackerElement: UnsafeMutablePointer<GstElement>?

    /// `GstElement *` for the MJPEG appsink.
    private var mjpegSinkElement: UnsafeMutablePointer<GstElement>?

    /// The live DetectionStream instance feeding the AsyncStream.
    private var detectionStream: DetectionStream?

    /// Raw opaque pointer from Unmanaged.passRetained — stored so stop() can
    /// call fromOpaque(...).release() to balance the retain taken in start().
    private var retainedBox: UnsafeMutableRawPointer?

    /// Probe ID returned by wendy_install_detection_probe.
    private var probeId: gulong = 0

    private var isRunning = false
    private let logger: Logger

    // MARK: Init

    init(stream: StreamConfig) {
        self.stream = stream
        self.logger = Logger(label: "GStreamerFrameReader[\(stream.name)]")
    }

    // MARK: Lifecycle

    /// Initialise GStreamer, parse the DeepStream pipeline (with tee + MJPEG
    /// branch), install the pad probe, and transition to PLAYING.
    ///
    /// Returns a tuple of the detection AsyncStream and an optional GstElementRef
    /// for the MJPEG branch appsink. The ref is valid for the lifetime of the
    /// pipeline; the caller MUST NOT release the underlying element.
    func start() throws -> (detections: AsyncStream<[Detection]>, mjpegSink: GstElementRef?) {
        guard !isRunning else {
            guard let ds = detectionStream else {
                throw GStreamerError.notStarted
            }
            let sinkRef = mjpegSinkElement.map { GstElementRef(element: $0) }
            return (ds.stream, sinkRef)
        }

        setupPluginEnvironment()
        gst_init(nil, nil)
        scanPluginPaths()

        let pipelineString = buildPipelineString()
        logger.info("Launching DeepStream pipeline", metadata: [
            "pipeline": "\(pipelineString)",
        ])

        var error: UnsafeMutablePointer<GError>?
        guard let parsed = gst_parse_launch(pipelineString, &error) else {
            let message = error.flatMap { String(cString: $0.pointee.message) } ?? "unknown"
            if let err = error { g_error_free(err) }
            throw GStreamerError.parseFailed(message)
        }
        if let err = error {
            let message = String(cString: err.pointee.message)
            logger.warning("gst_parse_launch warning: \(message)")
            g_error_free(err)
        }

        self.pipeline = parsed

        let bin = wendy_gst_bin_cast(parsed)

        // Find the nvtracker element by name to attach the probe.
        guard let tracker = gst_bin_get_by_name(bin, "wendy_tracker") else {
            cleanup()
            throw GStreamerError.elementNotFound("wendy_tracker")
        }
        self.nvtrackerElement = tracker

        // Find the MJPEG appsink (optional — log if not found but don't fail).
        // gst_bin_get_by_name returns a ref-counted element; we store it and
        // release it in cleanup(). The caller receives the raw pointer via
        // GstElementRef and MUST NOT release it — it's owned by GStreamerFrameReader.
        if let sink = gst_bin_get_by_name(bin, "mjpeg_sink") {
            self.mjpegSinkElement = sink
            logger.info("MJPEG appsink element found in pipeline")
        } else {
            logger.warning("MJPEG appsink element 'mjpeg_sink' not found — /stream will be unavailable")
        }

        // Create the DetectionStream and retain it for the lifetime of the probe.
        let ds = DetectionStream()
        self.detectionStream = ds
        let box = Unmanaged.passRetained(ds).toOpaque()
        self.retainedBox = box

        let id = wendy_install_detection_probe(
            tracker,
            "src",
            { count, dets, userData in
                wendyDetectionProbeEntry(count: count, dets: dets, userData: userData)
            },
            box
        )
        self.probeId = id

        if id == 0 {
            logger.error("Failed to install detection probe on nvtracker src pad")
        }

        let stateChange = gst_element_set_state(parsed, GST_STATE_PLAYING)
        guard stateChange != GST_STATE_CHANGE_FAILURE else {
            cleanup()
            throw GStreamerError.stateChangeFailed("PLAYING")
        }

        isRunning = true
        logger.info("DeepStream pipeline started", metadata: [
            "stream": "\(stream.name)",
            "mjpegBranch": "\(mjpegSinkElement != nil ? "enabled" : "disabled")",
        ])

        let sinkRef = mjpegSinkElement.map { GstElementRef(element: $0) }
        return (ds.stream, sinkRef)
    }

    /// Transition the pipeline to NULL and release all GStreamer resources.
    func stop() {
        isRunning = false

        detectionStream?.finish()
        detectionStream = nil

        if let box = retainedBox {
            Unmanaged<DetectionStream>.fromOpaque(box).release()
            retainedBox = nil
        }

        cleanup()
        logger.info("DeepStream pipeline stopped", metadata: [
            "stream": "\(stream.name)",
        ])
    }

    // MARK: Private

    private func buildPipelineString() -> String {
        let url = stream.url
        // Single rtspsrc → nvv4l2decoder → tee.
        // Branch 1 (detection): nvstreammux → nvinfer → nvtracker → fakesink.
        // Branch 2 (MJPEG):     nvvideoconvert → nvjpegenc → appsink.
        //
        // queue max-size-buffers=2 leaky=downstream on the MJPEG branch: natural
        // frame dropper. When nvjpegenc is slower than the source, older frames
        // are dropped to keep latency low.
        //
        // appsink max-buffers=1 drop=true: when no client is pulling, GStreamer
        // discards frames at the appsink rather than building up a backlog. This
        // keeps memory flat when /stream has no viewers.
        // Pipeline: tee AFTER nvtracker so both branches carry NvDsBatchMeta
        // (required by nvdsosd to draw boxes). Detection branch is just a
        // fakesink; the pad probe is on nvtracker's src pad, unchanged.
        // MJPEG branch: nvvideoconvert (NV12 NVMM → RGBA NVMM) → nvdsosd
        // draws boxes in-place → nvvideoconvert back → nvjpegenc → appsink.
        return """
        rtspsrc location=\(url) latency=200 protocols=tcp \
        ! rtph264depay ! h264parse \
        ! nvv4l2decoder \
        ! m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 \
        ! nvinfer config-file-path=/app/nvinfer_config.txt \
        ! nvtracker name=wendy_tracker \
            ll-config-file=/app/tracker_config.yml \
            ll-lib-file=/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so \
        ! tee name=t \
        t. ! queue ! fakesink \
        t. ! queue max-size-buffers=2 leaky=downstream \
        ! nvvideoconvert \
        ! nvdsosd \
        ! nvvideoconvert \
        ! nvjpegenc quality=85 \
        ! appsink name=mjpeg_sink emit-signals=false sync=false max-buffers=1 drop=true
        """
    }

    private func setupPluginEnvironment() {
        let pluginPaths = [
            "/usr/lib/gstreamer-1.0",
            "/usr/share/nvidia-container-passthrough/usr/lib/aarch64-linux-gnu/gstreamer-1.0",
            "/usr/lib/aarch64-linux-gnu/gstreamer-1.0",
        ]
        let currentPluginPath: String = {
            guard let ptr = getenv("GST_PLUGIN_PATH") else { return "" }
            return String(cString: ptr)
        }()
        var combined = currentPluginPath
        for path in pluginPaths where !combined.contains(path) {
            combined = combined.isEmpty ? path : "\(combined):\(path)"
        }
        setenv("GST_PLUGIN_PATH", combined, 1)
        setenv("GST_PLUGIN_SCANNER", "", 1)
        setenv("GST_REGISTRY_UPDATE", "yes", 1)
    }

    private func scanPluginPaths() {
        let pluginPaths = [
            "/usr/lib/gstreamer-1.0",
            "/usr/share/nvidia-container-passthrough/usr/lib/aarch64-linux-gnu/gstreamer-1.0",
            "/usr/lib/aarch64-linux-gnu/gstreamer-1.0",
        ]
        for path in pluginPaths {
            let found = wendy_gst_registry_scan_path(path)
            logger.info("GStreamer plugin scan", metadata: [
                "path": "\(path)",
                "pluginsFound": "\(found)",
            ])
        }
    }

    private func cleanup() {
        if let tracker = nvtrackerElement {
            gst_object_unref(UnsafeMutableRawPointer(tracker))
            self.nvtrackerElement = nil
        }
        if let sink = mjpegSinkElement {
            gst_object_unref(UnsafeMutableRawPointer(sink))
            self.mjpegSinkElement = nil
        }
        if let pipeline = pipeline {
            gst_element_set_state(pipeline, GST_STATE_NULL)
            gst_object_unref(UnsafeMutableRawPointer(pipeline))
            self.pipeline = nil
        }
    }
}

// MARK: - GStreamerReader (Sendable wrapper)

/// `Sendable` wrapper that exposes detections as an `AsyncStream<[Detection]>`
/// and provides the MJPEG appsink element for the sidecar pull loop.
struct GStreamerReader: Sendable {
    private let reader: GStreamerFrameReader

    init(stream: StreamConfig) {
        self.reader = GStreamerFrameReader(stream: stream)
    }

    /// Start the DeepStream pipeline and return detections + MJPEG sink.
    ///
    /// The MJPEG sink element is valid for the lifetime of the pipeline.
    /// It is owned by the GStreamerFrameReader; callers must not release it.
    func start() async throws -> (detections: AsyncStream<[Detection]>, mjpegSink: GstElementRef?) {
        try await reader.start()
    }

    /// Stop the underlying pipeline and finish the stream.
    func stop() async {
        await reader.stop()
    }
}

// MARK: - GStreamerError

enum GStreamerError: Error, CustomStringConvertible {
    case parseFailed(String)
    case elementNotFound(String)
    case stateChangeFailed(String)
    case notStarted

    var description: String {
        switch self {
        case .parseFailed(let m): return "GStreamer parse failed: \(m)"
        case .elementNotFound(let n): return "GStreamer element not found: \(n)"
        case .stateChangeFailed(let s): return "GStreamer state change failed: \(s)"
        case .notStarted: return "GStreamer pipeline not started"
        }
    }
}
