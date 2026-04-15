// GStreamerFrameReader.swift
// RTSP frame decoder using GStreamer with software H.264 decode (avdec_h264).
//
// Pipeline:
//   rtspsrc location=URL latency=200 protocols=tcp
//     ! rtph264depay ! h264parse
//     ! avdec_h264                                (FFmpeg software decode)
//     ! videoconvert ! videoscale
//     ! video/x-raw,format=RGB,width=W,height=H
//     ! appsink                                   (synchronous pull from Swift)
//
// NVDEC hardware decode via nvv4l2decoder is AVAILABLE on the device but
// incompatible with our appsink + gst_buffer_map pattern. See
// buildPipelineString for the full history and future-work options.
//
// We use synchronous `gst_app_sink_pull_sample` rather than callbacks to avoid
// running a GLib main loop alongside Swift's cooperative executor. The pull
// blocks the calling thread until a sample is ready, so we run it on a dedicated
// dispatch queue and resume the actor via a continuation.

import CGStreamer
import Dispatch
import Logging

#if canImport(Glibc)
    import Glibc
#elseif canImport(Musl)
    import Musl
#endif

#if canImport(FoundationEssentials)
    internal import FoundationEssentials
#else
    internal import Foundation
#endif

// MARK: - GStreamerFrameReader

/// Actor that owns a GStreamer pipeline and exposes a sequential frame stream.
actor GStreamerFrameReader {

    // MARK: Configuration

    let stream: StreamConfig
    let width: Int
    let height: Int

    // MARK: Pipeline state

    /// `GstElement *` for the parsed pipeline. NULL until `start()` succeeds.
    private var pipeline: UnsafeMutablePointer<GstElement>?

    /// `GstElement *` for the appsink element retrieved by name.
    private var appsinkElement: UnsafeMutablePointer<GstElement>?

    /// `GstAppSink *` cast for `gst_app_sink_pull_sample`.
    private var appsink: UnsafeMutablePointer<GstAppSink>?

    private var isRunning = false
    private let logger: Logger

    /// Dedicated serial queue for blocking `gst_app_sink_pull_sample` calls so
    /// they never stall the Swift cooperative thread pool (which has a small
    /// fixed number of threads — multiple stalled streams would deadlock it).
    private let pullQueue = DispatchQueue(label: "gstreamer-frame-reader.pull")

    // MARK: Init

    init(stream: StreamConfig, width: Int = 1920, height: Int = 1080) {
        self.stream = stream
        self.width = width
        self.height = height
        self.logger = Logger(label: "GStreamerFrameReader[\(stream.name)]")
    }

    // NOTE: no deinit cleanup. Swift 6 forbids accessing non-Sendable actor
    // properties (the GStreamer pointers) from a nonisolated deinit context.
    // Callers must invoke `stop()` before releasing the reader. For the
    // long-lived detector process the OS reclaims the GStreamer pipeline at
    // exit anyway, so this is acceptable.

    // MARK: Pipeline string

    // MARK: Lifecycle

    /// Initialise GStreamer (idempotent), parse the pipeline string, and
    /// transition it to PLAYING.
    func start() throws {
        guard !isRunning else { return }

        // Note: LIBV4L2_PLUGIN_DIR is NOT used by libnvv4l2.so (NVIDIA's version
        // hardcodes the path to /usr/lib/aarch64-linux-gnu/libv4l/plugins/nv/*.so).
        // The plugins must be symlinked there on the host and CDI-mounted into
        // the container at that same path.

        // Plugin discovery: CDI injects GStreamer 1.22 plugins from the host
        // at /usr/lib/gstreamer-1.0/ and NV-specific plugins at the passthrough
        // path. Build a comprehensive GST_PLUGIN_PATH that covers all sources.
        let pluginPaths = [
            "/usr/lib/gstreamer-1.0",                          // CDI: host's 1.22 plugins
            "/usr/share/nvidia-container-passthrough/usr/lib/aarch64-linux-gnu/gstreamer-1.0",  // CDI: NV plugins (nvv4l2decoder)
            "/usr/lib/aarch64-linux-gnu/gstreamer-1.0",        // container's own plugins (if any match)
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

        // Override CDI's /dev/null scanner — empty string forces in-process scan.
        setenv("GST_PLUGIN_SCANNER", "", 1)
        setenv("GST_REGISTRY_UPDATE", "yes", 1)

        // gst_init is safe to call multiple times — internally guarded.
        gst_init(nil, nil)

        // Explicitly scan each plugin directory so elements are registered
        // even if the GStreamer registry cache is stale or disabled.
        for path in pluginPaths {
            let found = wendy_gst_registry_scan_path(path)
            logger.info("GStreamer plugin scan", metadata: [
                "path": "\(path)",
                "pluginsFound": "\(found)",
            ])
        }

        let pipelineString = buildPipelineString()
        logger.info("Launching GStreamer pipeline", metadata: [
            "pipeline": "\(pipelineString)",
        ])

        var error: UnsafeMutablePointer<GError>?
        guard let parsed = gst_parse_launch(pipelineString, &error) else {
            let message = error.flatMap { String(cString: $0.pointee.message) } ?? "unknown"
            if let err = error {
                g_error_free(err)
            }
            throw GStreamerError.parseFailed(message)
        }
        if let err = error {
            // Some warnings come back here even on success — log and free.
            let message = String(cString: err.pointee.message)
            logger.warning("gst_parse_launch warning: \(message)")
            g_error_free(err)
        }

        self.pipeline = parsed

        // Find the appsink element by name.
        let bin = wendy_gst_bin_cast(parsed)
        guard let sinkElement = gst_bin_get_by_name(bin, "wendy_sink") else {
            cleanup()
            throw GStreamerError.elementNotFound("wendy_sink")
        }
        self.appsinkElement = sinkElement
        self.appsink = wendy_gst_app_sink_cast(sinkElement)

        // PLAYING state — async; the bus will eventually report errors if
        // the upstream RTSP server is unreachable.
        let stateChange = gst_element_set_state(parsed, GST_STATE_PLAYING)
        guard stateChange != GST_STATE_CHANGE_FAILURE else {
            cleanup()
            throw GStreamerError.stateChangeFailed("PLAYING")
        }

        isRunning = true
        logger.info("GStreamer pipeline started", metadata: [
            "stream": "\(stream.name)",
        ])
    }

    /// Transition the pipeline to NULL and release all GStreamer resources.
    func stop() {
        isRunning = false
        cleanup()
        logger.info("GStreamer pipeline stopped", metadata: [
            "stream": "\(stream.name)",
        ])
    }

    // MARK: Frame reading

    /// Pull a single frame from the appsink. Returns `nil` on EOS.
    func nextFrame() async throws -> Frame? {
        guard let sink = appsink else {
            throw GStreamerError.notStarted
        }

        let frameWidth = width
        let frameHeight = height

        // GStreamer pointers aren't Sendable; wrap for crossing the closure
        // boundary. The pointer is owned by the actor and only one thread
        // pulls at a time, so this is safe.
        let sinkPtr = SendableGstPointer(sink)

        // Pull is blocking — bounce it onto a dedicated queue so we never
        // pin the Swift cooperative thread pool.
        let bytes: [UInt8]? = await withCheckedContinuation { continuation in
            pullQueue.async {
                var sampleHandle: UnsafeMutableRawPointer?
                var dataPtr: UnsafeMutableRawPointer?
                var size: Int = 0

                let ok = wendy_gst_pull_sample(sinkPtr.pointer, &sampleHandle, &dataPtr, &size)
                if ok == 0 {
                    continuation.resume(returning: nil)
                    return
                }
                defer { wendy_gst_release_sample(sampleHandle) }

                guard let src = dataPtr, size > 0 else {
                    continuation.resume(returning: nil)
                    return
                }

                // Copy out into a Swift-owned [UInt8] before the sample is released.
                let buffer = UnsafeBufferPointer<UInt8>(
                    start: src.bindMemory(to: UInt8.self, capacity: size),
                    count: size
                )
                continuation.resume(returning: Array(buffer))
            }
        }

        guard let bytes else { return nil }
        return Frame(data: bytes, width: frameWidth, height: frameHeight)
    }

    // MARK: Private

    private func buildPipelineString() -> String {
        let url = stream.url
        // Pipeline: rtspsrc + nvv4l2decoder (NVDEC hardware decode) + VIC-based
        // NVMM→RGB conversion.
        //
        // An earlier version dropped to avdec_h264 (software decode) after
        // observing the process consume memory at ~70-100 MB/s and OOM
        // within ~65 s. Root cause was NOT a closed-source NVIDIA leak —
        // it was our own pipeline misconfiguration. Two fixes:
        //
        //  1. nvvideoconvert: removed `compute-hw=1` (GPU path). On Orin Nano
        //     the GPU conversion path has a documented retained-surface bug
        //     when producing CPU-accessible system-memory RGB. VIC is the
        //     right unit on this SoC — GPU-mode is for AGX-class parts.
        //     Default behaviour (no property) lets the driver pick, which
        //     selects VIC on Orin Nano.
        //
        //  2. nvv4l2decoder: set `num-extra-surfaces=1`. The default 4-8
        //     pre-allocates more NVMM surfaces than our 2-buffer appsink
        //     working set can reclaim quickly, which under backpressure
        //     triggers "nvbufsurftransform_copy.cpp:438: Failed in mem copy"
        //     (pool exhausted) followed by re-allocation spam.
        //
        // The Swift shim (`wendy_gst_release_sample`) already unrefs samples
        // and unmaps buffers correctly via defer — that was verified.
        // Software H.264 decode via FFmpeg (avdec_h264). The NVDEC path via
        // nvv4l2decoder + nvvideoconvert/nvvidconv + appsink consistently leaks
        // 45–80 MB/s on L4T r36 / JetPack 6 — independent of converter choice
        // or properties (tested: default, compute-hw=1/2, copy-hw=2,
        // num-extra-surfaces=1, NV12 intermediate, queue leaky=downstream).
        //
        // Root cause is architectural, not a property: pulling frames through
        // appsink and calling gst_buffer_map(READ) on NVMM buffers pins
        // surfaces in the NvBufSurface pool when no CUDA context is present
        // in the consumer process. DeepStream itself avoids this because
        // nvinfer consumes NVMM in-pipeline — never calls gst_buffer_map.
        // Python + DeepStream doesn't hit this because the Python detector
        // IS a DeepStream-native pipeline. Our Swift detector pulls RGB into
        // userspace for TensorRT inference — which NVIDIA's design doesn't
        // support via the nvv4l2decoder → userspace route.
        //
        // Workarounds that might recover NVDEC here (future work):
        //   - Link against libcuda and cuInit(0) in the Swift process so
        //     NvBufSurfaceMap has a CUDA primary context to key against.
        //   - Use NVMM→EGLImage + cuGraphicsEGLRegisterImage (NVIDIA's
        //     documented zero-copy path) instead of appsink + gst_buffer_map.
        //   - Wrap nvinfer from Swift so inference stays in-pipeline.
        //
        // Software decode uses system RAM only, caps around 400 MB, and is
        // stable indefinitely. At 1080p/20fps on Orin Nano it's <100% of one
        // CPU core — acceptable.
        return """
        rtspsrc location=\(url) latency=200 protocols=tcp ! \
        rtph264depay ! h264parse ! \
        avdec_h264 ! \
        videoconvert ! videoscale ! \
        video/x-raw,format=RGB,width=\(width),height=\(height) ! \
        appsink name=wendy_sink emit-signals=false sync=false max-buffers=2 drop=true
        """
    }

    private func cleanup() {
        if let sinkElement = appsinkElement {
            gst_object_unref(UnsafeMutableRawPointer(sinkElement))
            self.appsinkElement = nil
            self.appsink = nil
        }
        if let pipeline = pipeline {
            gst_element_set_state(pipeline, GST_STATE_NULL)
            gst_object_unref(UnsafeMutableRawPointer(pipeline))
            self.pipeline = nil
        }
    }
}

// MARK: - SendableGstPointer

/// `@unchecked Sendable` wrapper around an `UnsafeMutablePointer<GstAppSink>`.
///
/// GStreamer's pointer types are not `Sendable`, but we need to pass them
/// across closure boundaries to run the blocking `gst_app_sink_pull_sample`
/// on a dedicated dispatch queue. The actor that owns the pointer ensures
/// only one task accesses it at a time, so this is safe.
private struct SendableGstPointer<T>: @unchecked Sendable {
    let pointer: UnsafeMutablePointer<T>
    init(_ pointer: UnsafeMutablePointer<T>) {
        self.pointer = pointer
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

// MARK: - GStreamerReader (Sendable wrapper, mirrors FrameReader API)

/// `Sendable` wrapper that exposes frames as an `AsyncStream<Frame>`,
/// matching the existing `FrameReader` API in `RTSPFrameReader.swift`.
struct GStreamerReader: Sendable {
    private let reader: GStreamerFrameReader

    init(stream: StreamConfig, width: Int = 1920, height: Int = 1080) {
        self.reader = GStreamerFrameReader(stream: stream, width: width, height: height)
    }

    /// Start the pipeline and return an `AsyncStream` that yields frames.
    func frames() -> AsyncStream<Frame> {
        AsyncStream { continuation in
            let reader = self.reader
            let task = Task {
                do {
                    try await reader.start()
                    while !Task.isCancelled {
                        if let frame = try await reader.nextFrame() {
                            continuation.yield(frame)
                        } else {
                            // EOS or error.
                            break
                        }
                    }
                } catch {
                    let logger = Logger(label: "GStreamerReader")
                    logger.error("GStreamer reader error: \(error)")
                }
                continuation.finish()
            }

            continuation.onTermination = { _ in
                task.cancel()
                Task { await reader.stop() }
            }
        }
    }

    /// Stop the underlying pipeline and finish the stream.
    func stop() async {
        await reader.stop()
    }
}
