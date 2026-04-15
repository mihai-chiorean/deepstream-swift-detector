// Detector.swift
// Main entry point for the Swift YOLO26n object detector with VLM integration.
//
// Ties together:
//   - TensorRT engine loading (DetectorEngine) — YOLO26 NMS-free detection
//   - RTSP frame decoding (RTSPFrameReader via FFmpeg subprocess)
//   - YOLO preprocessing and inference (no NMS — YOLO26 is end-to-end)
//   - IoU-based multi-object tracking (IOUTracker)
//   - VLM descriptions via llama.cpp sidecar (VLMClient)
//   - Bounding box rendering and JPEG encoding (FrameRenderer)
//   - Hummingbird HTTP server on :9090 (metrics, MJPEG, health)
//   - Prometheus metrics (Metrics.swift)

import ArgumentParser
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

// Redirect stdout/stderr to a log file very early in startup.
//
// WendyOS runs the container with stdout/stderr wired to a containerd named
// pipe that nothing reads, so `print(...)` and swift-log messages go into the
// void. Writing to a real file lets us debug the detection pipeline via
// `ssh root@<device> "cat /app/detector.log"`.
//
// Called from main() before any logger is constructed so nothing is lost.
@inline(__always)
private func redirectStdioToFile(_ path: String) {
    let fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
    guard fd >= 0 else { return }
    // Write a startup marker so each process invocation is visibly delimited.
    let marker = "\n=== Detector starting (pid \(getpid())) ===\n"
    _ = marker.withCString { cstr in
        write(fd, cstr, strlen(cstr))
    }
    _ = dup2(fd, 1)   // stdout
    _ = dup2(fd, 2)   // stderr
    // Note: we don't call setvbuf(stdout, ...) here because Swift 6 flags
    // the global `stdout` as non-Sendable. Writes to fd 1/2 via the raw
    // syscall path (which swift-log's StreamLogHandler uses internally)
    // are already line-buffered by the kernel on character device fds.
    close(fd)
}

@main
struct Detector: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        abstract: "YOLO26n object detector with VLM — Swift/TensorRT edition"
    )

    // Engine path resolution order:
    //   1. The image-bundled engine at /app/yolo26n_b2_fp16.engine (pre-built,
    //      shipped in the container image via wendy.json `resources`). Fastest
    //      first-boot — no ONNX→engine rebuild needed.
    //   2. The persist-volume cache at /var/cache/detector/yolo26n_b2_fp16.engine
    //      (used when the image-bundled path is missing; useful during development
    //      when the engine is being rebuilt).
    //   3. If neither exists, DetectorEngine falls back to ONNX and writes the
    //      result to the persist-volume cache for next time.
    @Option(name: .long, help: "Path to serialised TensorRT engine file")
    var enginePath: String = "/app/yolo26n_b2_fp16.engine"

    @Option(name: .long, help: "Fallback cache path used if the engine must be rebuilt from ONNX")
    var engineCachePath: String = "/var/cache/detector/yolo26n_b2_fp16.engine"

    @Option(name: .long, help: "Path to ONNX model (fallback if engine missing)")
    var onnxPath: String = "/app/yolo26n.onnx"

    @Option(name: .long, help: "Path to labels.txt (one class name per line)")
    var labelsPath: String = "/app/labels.txt"

    @Option(name: .long, help: "Path to streams.json configuration")
    var streamsPath: String = "/app/streams.json"

    @Option(name: .long, help: "HTTP server port")
    var port: Int = 9090

    @Option(name: .long, help: "VLM server URL (llama.cpp sidecar)")
    var vlmURL: String = "http://127.0.0.1:8090"

    func run() async throws {
        // Redirect stdio BEFORE touching any logger. WendyOS containerd wires
        // stdout/stderr to an unread named pipe; everything printed there is
        // lost. The log file lives in the `detector-cache` persistence volume
        // at /var/cache/detector/detector.log so it survives container restarts.
        // Readable via:
        //   ssh root@<device> "cat /var/lib/wendy-agent/storage/detector-cache/detector.log"
        redirectStdioToFile("/var/cache/detector/detector.log")

        var logger = Logger(label: "detector")
        logger.logLevel = .info

        // ---------------------------------------------------------------
        // 1. Load stream configuration
        // ---------------------------------------------------------------

        let streamsConfig = try StreamsConfig.load(from: streamsPath)
        let enabledStreams = streamsConfig.streams.filter(\.enabled)

        guard !enabledStreams.isEmpty else {
            logger.error("No enabled streams found in \(streamsPath)")
            return
        }

        logger.info("Loaded \(enabledStreams.count) stream(s)", metadata: [
            "streams": "\(enabledStreams.map(\.name))",
        ])

        // ---------------------------------------------------------------
        // 2. Initialise the TensorRT detector engine
        // ---------------------------------------------------------------

        logger.info("Loading TensorRT engine...")
        let engine = try await DetectorEngine(
            enginePath: enginePath,
            engineCachePath: engineCachePath,
            onnxPath: onnxPath,
            labelsPath: labelsPath
        )
        logger.info("DetectorEngine ready")

        // ---------------------------------------------------------------
        // 3. Shared state for the HTTP server
        // ---------------------------------------------------------------

        let detectorState = DetectorState()

        // ---------------------------------------------------------------
        // 4. VLM client for track descriptions
        // ---------------------------------------------------------------

        let vlmClient = VLMClient(baseURL: vlmURL)
        let trackFinalizer = TrackFinalizer(vlmClient: vlmClient)

        // Check VLM availability in the background.
        Task {
            let available = await vlmClient.checkHealth()
            if available {
                logger.info("VLM server available at \(vlmURL)")
            } else {
                logger.warning("VLM server not available at \(vlmURL) — descriptions disabled")
            }
        }

        // ---------------------------------------------------------------
        // 5. Start the HTTP server in the background
        // ---------------------------------------------------------------

        let httpTask = Task {
            do {
                try await startHTTPServer(
                    state: detectorState,
                    metrics: metrics,
                    vlmClient: vlmClient,
                    port: port
                )
            } catch {
                logger.critical("HTTP server failed: \(error)")
            }
        }

        logger.info("HTTP server starting on port \(port)")

        // ---------------------------------------------------------------
        // 6. Run one detection loop per enabled stream
        // ---------------------------------------------------------------

        let taskLogger = logger
        await withTaskGroup(of: Void.self) { group in

            for stream in enabledStreams {
                group.addTask {
                    await runStreamDetectionLoop(
                        stream: stream,
                        engine: engine,
                        state: detectorState,
                        trackFinalizer: trackFinalizer,
                        logger: taskLogger
                    )
                }
            }

            // Wait for SIGTERM/SIGINT — the task group keeps running until
            // all children complete (which they won't unless the streams
            // are stopped or the process is signalled).
            await group.waitForAll()
        }

        // If we get here the streams have stopped; cancel the HTTP server.
        httpTask.cancel()
    }
}

// MARK: - Per-stream detection loop

/// Reads frames from an RTSP stream, runs YOLO detection + tracking, updates
/// metrics, and pushes rendered frames to the MJPEG server.
private func runStreamDetectionLoop(
    stream: StreamConfig,
    engine: DetectorEngine,
    state: DetectorState,
    trackFinalizer: TrackFinalizer,
    logger: Logger
) async {
    var logger = logger
    logger[metadataKey: "stream"] = "\(stream.name)"

    // Pre-create metric handles for this stream.
    let fps = fpsGauge(stream: stream.name)
    let activeTracks = activeTracksGauge(stream: stream.name)
    let framesProcessed = framesProcessedCounter(stream: stream.name)
    let inferenceLatency = inferenceLatencyHistogram(stream: stream.name)
    let totalLatency = totalLatencyHistogram(stream: stream.name)

    // Tracker state (mutated per frame).
    var tracker = IOUTracker()

    // Preallocated scratch buffer for MJPEG rendering.
    // Reused every rendered frame to avoid a ~6 MB heap allocation at 20 FPS
    // (124 MB/s allocation pressure exhausted swap on Jetson Orin Nano within
    // ~65 s when a stream client was connected).
    var renderScratch = [UInt8]()

    // Render throttle: cap MJPEG output at 8 FPS (125 ms minimum interval).
    // Full inference still runs at the source frame rate; we only skip the
    // expensive copy + JPEG-encode step when the budget hasn't elapsed.
    var lastRenderTime = ContinuousClock.now
    let renderInterval = Duration.milliseconds(125)   // 8 FPS cap

    // Cache for per-class counter handles (avoids Mutex lock per class per frame).
    var classCounterCache: [String: CounterMetric] = [:]

    // FPS calculation state.
    var frameCount: UInt64 = 0
    var fpsWindowStart = ContinuousClock.now

    // Cache the label list (immutable after init, avoids per-frame actor hop).
    let classLabels = engine.postprocessor.labels

    // Frame reader: GStreamer in-process with avdec_h264 software decode.
    // Hardware decode (nvv4l2decoder) is broken at the WendyOS platform level:
    // libnvv4l2.so's v4l2_fd_open fails to claim /dev/v4l2-nvdec even on the
    // host (outside containers). Needs platform fix in meta-tegra/meta-wendyos.
    let reader = GStreamerReader(stream: stream)

    logger.info("Starting detection loop")

    for await frame in reader.frames() {
        let frameStart = ContinuousClock.now

        do {
            // --- Inference ---
            let inferStart = ContinuousClock.now
            var detections = try await engine.detect(frame: frame)
            let inferEnd = ContinuousClock.now
            inferenceLatency.observe(durationMs(inferEnd - inferStart))

            // --- Tracking ---
            detections = tracker.update(detections: detections)
            activeTracks.set(Double(tracker.confirmedTrackCount))

            // --- VLM: submit finalized tracks for description ---
            for finalized in tracker.finalizedTracks {
                await trackFinalizer.submit(
                    track: finalized,
                    frame: frame,
                    labels: classLabels
                )
            }

            // --- Metrics ---
            frameCount += 1
            framesProcessed.inc()

            // Record per-class detection counts. Cache counter handles
            // to avoid repeated Mutex lock + dictionary lookup per frame.
            var classCounts: [String: Int] = [:]
            for det in detections {
                classCounts[det.label, default: 0] += 1
            }
            for (className, count) in classCounts {
                let counter: CounterMetric
                if let cached = classCounterCache[className] {
                    counter = cached
                } else {
                    let handle = detectionsTotalCounter(stream: stream.name, class_: className)
                    classCounterCache[className] = handle
                    counter = handle
                }
                counter.inc(by: Double(count))
            }

            // FPS: compute over a sliding 1-second window.
            let now = ContinuousClock.now
            let elapsedSeconds = durationSeconds(now - fpsWindowStart)
            if elapsedSeconds >= 1.0 {
                fps.set(Double(frameCount) / elapsedSeconds)
                frameCount = 0
                fpsWindowStart = now
            }

            // --- Frame rendering (only if MJPEG clients connected) ---
            //
            // The rendering path is throttled to 8 FPS to bound memory pressure:
            //   -  is a preallocated buffer reused each frame,
            //     eliminating the per-frame 6.2 MB heap allocation that caused
            //     continuous swap churn at 20 FPS (observed: 88 MB/s swap growth,
            //     OOM in ~65 s on an 8 GB Orin Nano with a 5 GiB cgroup cap).
            //   - The 125 ms gate skips encoding on frames where the MJPEG client
            //     wouldn't benefit (sub-8 FPS MJPEG is visually acceptable for a
            //     surveillance preview; the inference pipeline runs unthrottled).
            if await state.shouldExtractFrames {
                let now = ContinuousClock.now
                if now - lastRenderTime >= renderInterval {
                    lastRenderTime = now
                    if let jpeg = FrameRenderer.renderFrame(
                        frame.data,
                        into: &renderScratch,
                        width: frame.width,
                        height: frame.height,
                        detections: detections
                    ) {
                        await state.setFrame(jpeg)
                    }
                }
            }

            // --- Total latency ---
            let frameEnd = ContinuousClock.now
            totalLatency.observe(durationMs(frameEnd - frameStart))

        } catch {
            logger.error("Detection error: \(error)")
        }
    }

    logger.warning("Detection loop ended for stream \(stream.name)")
}

// MARK: - Duration conversion helpers

/// Converts a Duration to milliseconds, accounting for both the seconds
/// and attoseconds components. `.components.attoseconds` only returns the
/// fractional part — without adding the whole-seconds portion the result
/// silently loses any duration >= 1 second.
private func durationMs(_ d: Duration) -> Double {
    let c = d.components
    return Double(c.seconds) * 1000.0 + Double(c.attoseconds) / 1_000_000_000_000_000
}

/// Converts a Duration to seconds (Double).
private func durationSeconds(_ d: Duration) -> Double {
    let c = d.components
    return Double(c.seconds) + Double(c.attoseconds) / 1_000_000_000_000_000_000
}
