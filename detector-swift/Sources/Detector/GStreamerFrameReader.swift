// GStreamerFrameReader.swift
// DeepStream-native RTSP detection pipeline using nvinfer + nvtracker.
//
// Pipeline (pure detection, no MJPEG branch):
//   rtspsrc → rtph264depay → h264parse → nvv4l2decoder
//     → nvstreammux → nvinfer → nvtracker → fakesink
//
// Stage 2: The tee, valve, videodrop, nvvideoconvert, nvdsosd, nvjpegenc, and
// appsink elements are gone. Video is served by mediamtx via WebRTC; the detector
// only emits detection metadata.
//
// Detection branch: pad probe on nvtracker src pad. No gst_buffer_map —
// no NVMM pinning, no memory leak.
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

// MARK: - FrameTiming

/// Per-frame component latency values extracted from DeepStream's
/// NVDS_LATENCY_MEASUREMENT_META batch user-metas.
///
/// Requires NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1 in the environment.
/// All fields are 0 when the env var is not set or the component didn't emit
/// a meta for the frame.
///
/// Bucket mapping (see nvds_shim.c for exact component name matching):
///   decodeMs      — nvv4l2decoder
///   streammuxMs   — nvstreammux only (buffer batching + pad synchronisation)
///   inferenceMs   — nvinfer only (letterbox resize + YOLO26n forward pass)
///   preprocessMs  — nvstreammux + nvinfer sum (= streammuxMs + inferenceMs);
///                   kept for backwards compatibility
///   postprocessMs — nvtracker (track management; bbox-parser runs inside nvinfer
///                   but DS doesn't break it out as a separate meta component)
///   ptsNs         — GStreamer buffer PTS in nanoseconds; -1 if no PTS available
///                   (GST_CLOCK_TIME_NONE). Used for frame-accurate sync.
struct FrameTiming: Sendable {
    /// nvv4l2decoder latency in milliseconds; 0 if unavailable.
    var decodeMs: Double
    /// nvstreammux latency in milliseconds (buffer batching only); 0 if unavailable.
    var streammuxMs: Double
    /// nvinfer latency in milliseconds (letterbox + DNN forward pass); 0 if unavailable.
    var inferenceMs: Double
    /// nvstreammux + nvinfer accumulated latency in milliseconds; 0 if unavailable.
    /// Equals streammuxMs + inferenceMs. Kept for backwards-compat (monitor.html).
    var preprocessMs: Double
    /// nvtracker latency in milliseconds; 0 if unavailable.
    var postprocessMs: Double
    /// Buffer PTS in nanoseconds (from GST_BUFFER_PTS). -1 if not available.
    var ptsNs: Int64
}

// MARK: - DetectionStream

/// Bridges the C pad-probe callback to an AsyncStream<DetectionFrame>.
///
/// The instance is heap-allocated and retained for the lifetime of the
/// GStreamer pipeline via `Unmanaged.passRetained`. Calling `finish()` ends
/// the stream; `ingest` feeds detections from the streaming thread.
///
/// Swift 6 note: `continuation` is Sendable; `ingest` can be called safely
/// from the GStreamer streaming thread without actor hops. The class is
/// marked @unchecked Sendable to cross the isolation boundary.
/// One frame's output: the detection list plus the frame's end-to-end
/// pipeline latency (now − buffer.PTS, in nanoseconds). Computed in the C
/// probe using `gst_element_get_current_running_time` on the tracker element.
struct DetectionFrame: Sendable {
    let detections: [Detection]
    let frameLatencyNs: UInt64
    /// Per-component latency from DS NVDS_LATENCY_MEASUREMENT_META metas.
    /// All fields are 0 when NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT is not set.
    /// timing.ptsNs is -1 when the buffer carries no PTS.
    let timing: FrameTiming
}

final class DetectionStream: @unchecked Sendable {

    private let continuation: AsyncStream<DetectionFrame>.Continuation
    let stream: AsyncStream<DetectionFrame>

    init() {
        var cap: AsyncStream<DetectionFrame>.Continuation!
        self.stream = AsyncStream(bufferingPolicy: .bufferingNewest(4)) { cap = $0 }
        self.continuation = cap
    }

    /// Called from the C probe callback on the GStreamer streaming thread.
    /// Safe to call from any thread — AsyncStream.Continuation is Sendable.
    func ingest(_ detections: [Detection], frameLatencyNs: UInt64, timing: FrameTiming) {
        continuation.yield(DetectionFrame(
            detections: detections,
            frameLatencyNs: frameLatencyNs,
            timing: timing
        ))
    }

    /// Finish the stream (called on pipeline teardown).
    func finish() {
        continuation.finish()
    }
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
    frameLatencyNs: UInt64,
    cTiming: WendyFrameTiming,
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

    let timing = FrameTiming(
        decodeMs: cTiming.decode_ms,
        streammuxMs: cTiming.streammux_ms,
        inferenceMs: cTiming.inference_ms,
        preprocessMs: cTiming.preprocess_ms,
        postprocessMs: cTiming.postprocess_ms,
        ptsNs: cTiming.ptsNs
    )

    ds.ingest(copy, frameLatencyNs: frameLatencyNs, timing: timing)
}

// MARK: - GStreamerFrameReader

/// Actor that owns a DeepStream GStreamer pipeline and exposes detections as
/// an `AsyncStream<DetectionFrame>`.
///
/// Stage 2: pure detection pipeline only — no tee, no MJPEG branch.
/// Video is delivered by mediamtx via WebRTC. The detector pipeline is:
///   rtspsrc → rtph264depay → h264parse → nvv4l2decoder
///     → nvstreammux → nvinfer → nvtracker → fakesink
actor GStreamerFrameReader {

    // MARK: Configuration

    let stream: StreamConfig

    // MARK: Pipeline state

    /// `GstElement *` for the parsed pipeline. NULL until `start()` succeeds.
    private var pipeline: UnsafeMutablePointer<GstElement>?

    /// `GstElement *` for the nvtracker element (probe target).
    private var nvtrackerElement: UnsafeMutablePointer<GstElement>?

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

    /// Initialise GStreamer, parse the DeepStream pipeline, install the pad
    /// probe, and transition to PLAYING.
    ///
    /// Returns the detection AsyncStream. No MJPEG sink or valve — Stage 2
    /// removed the MJPEG branch entirely.
    func start() throws -> AsyncStream<DetectionFrame> {
        guard !isRunning else {
            guard let ds = detectionStream else {
                throw GStreamerError.notStarted
            }
            return ds.stream
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

        // Create the DetectionStream and retain it for the lifetime of the probe.
        let ds = DetectionStream()
        self.detectionStream = ds
        let box = Unmanaged.passRetained(ds).toOpaque()
        self.retainedBox = box

        let id = wendy_install_detection_probe(
            tracker,
            "src",
            { count, dets, frameLatencyNs, cTiming, userData in
                wendyDetectionProbeEntry(
                    count: count,
                    dets: dets,
                    frameLatencyNs: frameLatencyNs,
                    cTiming: cTiming,
                    userData: userData
                )
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
        logger.info("DeepStream pipeline started (pure detection, no MJPEG branch)", metadata: [
            "stream": "\(stream.name)",
        ])

        return ds.stream
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

    /// Build the GStreamer pipeline description string.
    ///
    /// Stage 2: pure detection pipeline only. No tee, no MJPEG branch.
    /// Video is served by mediamtx via WebRTC; the detector only publishes
    /// detection metadata via the /detections WebSocket.
    ///
    ///   rtspsrc → rtph264depay → h264parse → nvv4l2decoder
    ///     → nvstreammux → nvinfer → nvtracker → fakesink
    private func buildPipelineString() -> String {
        let url = stream.url
        return """
        rtspsrc location=\(url) latency=200 protocols=tcp \
        ! rtph264depay ! h264parse \
        ! nvv4l2decoder \
        ! m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 \
        ! nvinfer config-file-path=/app/nvinfer_config.txt \
            filter-out-class-ids=4;6;8;9;10;11;12;13;20;21;22;23;24;25;26;27;28;29;30;31;32;33;34;35;36;37;38;39;40;41;42;43;44;45;46;47;48;49;50;51;52;53;54;55;56;57;58;59;60;61;62;63;64;65;66;67;68;69;70;71;72;73;74;75;76;77;78;79 \
        ! nvtracker name=wendy_tracker \
            ll-config-file=/app/tracker_config.yml \
            ll-lib-file=/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so \
        ! fakesink
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
        if let pipeline = pipeline {
            gst_element_set_state(pipeline, GST_STATE_NULL)
            gst_object_unref(UnsafeMutableRawPointer(pipeline))
            self.pipeline = nil
        }
    }
}

// MARK: - GStreamerReader (Sendable wrapper)

/// `Sendable` wrapper that exposes detections as an `AsyncStream<DetectionFrame>`.
struct GStreamerReader: Sendable {
    private let reader: GStreamerFrameReader

    init(stream: StreamConfig) {
        self.reader = GStreamerFrameReader(stream: stream)
    }

    /// Start the DeepStream pipeline and return the detection stream.
    func start() async throws -> AsyncStream<DetectionFrame> {
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
