// Detector.swift
// Main entry point for the Swift DeepStream object detector.
//
// Ties together:
//   - DeepStream pipeline (GStreamerFrameReader) — nvinfer + nvtracker in-pipeline,
//     pure detection path (no tee, no MJPEG branch). Stage 2: video is served by
//     mediamtx via WebRTC; the detector only publishes detection metadata.
//   - Pad-probe-driven detection stream (AsyncStream<DetectionFrame>)
//   - Detection broadcaster (DetectionBroadcaster) — fans out per-frame JSON
//     to N WebSocket /detections clients; called inline from the detection loop
//   - Track-finalizer for VLM integration (currently disabled)
//   - Hummingbird HTTP server on :9090 (metrics, health, detections)
//   - Prometheus metrics (Metrics.swift)
//
// Reconnect: runStreamDetectionLoop now uses GStreamerReader.startWithReconnect()
// to automatically recover from upstream EOS or pipeline errors. The
// DetectionStream is long-lived; WebSocket subscribers are not dropped on reconnect.

import ArgumentParser
import Logging
import Synchronization

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

@inline(__always)
private func redirectStdioToFile(_ path: String) {
    let fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
    guard fd >= 0 else { return }
    let marker = "\n=== Detector starting (pid \(getpid())) ===\n"
    _ = marker.withCString { cstr in write(fd, cstr, strlen(cstr)) }
    _ = dup2(fd, 1)
    _ = dup2(fd, 2)
    close(fd)
}

@main
struct Detector: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        abstract: "DeepStream object detector — nvinfer + nvtracker in-pipeline edition"
    )

    @Option(name: .long, help: "Path to streams.json configuration")
    var streamsPath: String = "/app/streams.json"

    @Option(name: .long, help: "HTTP server port")
    var port: Int = 9090

    @Option(name: .long, help: "VLM server URL (llama.cpp sidecar — currently disabled)")
    var vlmURL: String = "http://127.0.0.1:8090"

    func run() async throws {
        redirectStdioToFile("/var/cache/detector/detector.log")

        var logger = Logger(label: "detector")
        logger.logLevel = .info

        let streamsConfig = try StreamsConfig.load(from: streamsPath)
        let enabledStreams = streamsConfig.streams.filter(\.enabled)

        guard !enabledStreams.isEmpty else {
            logger.error("No enabled streams found in \(streamsPath)")
            return
        }

        logger.info("Loaded \(enabledStreams.count) stream(s)", metadata: [
            "streams": "\(enabledStreams.map(\.name))",
        ])

        let vlmClient = VLMClient(baseURL: vlmURL)
        let trackFinalizer = TrackFinalizer(vlmClient: vlmClient)

        // DetectionBroadcaster is created upfront so /detections WebSocket clients
        // can subscribe before the pipeline starts. The broadcaster's distribute()
        // method is called inline from the detection loop on each frame.
        let detectorState = DetectorState()
        let broadcaster = DetectionBroadcaster()
        await detectorState.setDetectionBroadcaster(broadcaster)
        logger.info("Detection broadcaster created")

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

        let taskLogger = logger
        await withTaskGroup(of: Void.self) { group in
            for stream in enabledStreams {
                group.addTask {
                    await runStreamDetectionLoop(
                        stream: stream,
                        broadcaster: broadcaster,
                        trackFinalizer: trackFinalizer,
                        logger: taskLogger
                    )
                }
            }
            await group.waitForAll()
        }

        httpTask.cancel()
    }
}

// MARK: - FPS window state

/// Small Mutex-protected struct that holds per-frame FPS accounting vars.
///
/// Placed behind a Mutex so the `onCycleStart` closure (called from the
/// actor's reconnect Task) can safely reset the window while the detection
/// loop reads it on each frame — without an actor hop.
private struct FPSWindow: Sendable {
    var frameCount: UInt64 = 0
    var windowStart: ContinuousClock.Instant = ContinuousClock.now
}

// MARK: - Per-stream detection loop

private func runStreamDetectionLoop(
    stream: StreamConfig,
    broadcaster: DetectionBroadcaster,
    trackFinalizer: TrackFinalizer,
    logger: Logger
) async {
    var logger = logger
    logger[metadataKey: "stream"] = "\(stream.name)"

    // Metric handles — registered once, reused across reconnect cycles.
    let fps = fpsGauge(stream: stream.name)
    let reconnects = reconnectsTotalCounter(stream: stream.name)
    let framesProcessed = framesProcessedCounter(stream: stream.name)
    let activeTracks = activeTracksGauge(stream: stream.name)
    let totalLatency = totalLatencyHistogram(stream: stream.name)
    let decodeLatency = decodeLatencyHistogram(stream: stream.name)
    let streammuxLatency = streammuxLatencyHistogram(stream: stream.name)
    let inferenceLatency = inferenceLatencyHistogram(stream: stream.name)
    let preprocessLatency = preprocessLatencyHistogram(stream: stream.name)
    let postprocessLatency = postprocessLatencyHistogram(stream: stream.name)

    // Class-detection counter cache — accumulates across reconnect cycles.
    var classCounterCache: [String: CounterMetric] = [:]
    var previousTrackerIds = Set<Int>()

    // FPS window state behind a Mutex so the onCycleStart closure can reset
    // it safely from the actor's reconnect Task (different concurrency domain).
    let fpsWindow = Mutex(FPSWindow())

    let reader = GStreamerReader(stream: stream)

    logger.info("Starting DeepStream detection loop (with reconnect)")

    do {
        // onCycleStart is called at the beginning of every pipeline cycle
        // (including the first). It resets FPS window state so the gauge
        // does not report stale values during a reconnect.
        let detectionStream = try await reader.startWithReconnect(
            fpsGauge: fps,
            reconnectsCounter: reconnects,
            onCycleStart: {
                fpsWindow.withLock { w in
                    w.frameCount = 0
                    w.windowStart = ContinuousClock.now
                }
            }
        )

        // Detection loop: sole consumer of detectionStream.
        // The loop runs for the lifetime of the process; the reconnect logic
        // inside GStreamerReader rebuilds the pipeline without finishing this
        // AsyncStream, so the for-await loop never exits on a reconnect.
        for await frame in detectionStream {
            let detections = frame.detections
            let currentTrackerIds = Set(detections.compactMap(\.trackId))
            let disappearedIds = previousTrackerIds.subtracting(currentTrackerIds)
            for goneId in disappearedIds {
                await trackFinalizer.submitDisappeared(trackId: goneId)
            }
            previousTrackerIds = currentTrackerIds

            // Fan out to WebSocket /detections clients (non-blocking; slow clients
            // drop stale frames via per-client AsyncStream bufferingNewest(4)).
            await broadcaster.distribute(frame)

            framesProcessed.inc()

            // Per-frame gauges + histograms.
            activeTracks.set(Double(currentTrackerIds.count))
            if frame.frameLatencyNs > 0 {
                totalLatency.observe(Double(frame.frameLatencyNs) / 1_000_000.0)  // ns → ms
            }

            // DeepStream component latency histograms.
            // Non-zero only when NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1.
            let t = frame.timing
            if t.decodeMs > 0 { decodeLatency.observe(t.decodeMs) }
            if t.streammuxMs > 0 { streammuxLatency.observe(t.streammuxMs) }
            if t.inferenceMs > 0 { inferenceLatency.observe(t.inferenceMs) }
            if t.preprocessMs > 0 { preprocessLatency.observe(t.preprocessMs) }
            if t.postprocessMs > 0 { postprocessLatency.observe(t.postprocessMs) }

            var classCounts: [Int: Int] = [:]
            for det in detections {
                classCounts[det.classId, default: 0] += 1
            }
            for (classId, count) in classCounts {
                let className = "class_\(classId)"
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

            // FPS window accounting — read+write under Mutex so onCycleStart
            // can safely reset from a concurrent context.
            fpsWindow.withLock { w in
                w.frameCount += 1
                let now = ContinuousClock.now
                let elapsed = durationSeconds(now - w.windowStart)
                if elapsed >= 1.0 {
                    fps.set(Double(w.frameCount) / elapsed)
                    w.frameCount = 0
                    w.windowStart = now
                }
            }
        }

    } catch {
        logger.error("Failed to start DeepStream pipeline: \(error)")
    }

    logger.warning("Detection loop ended for stream \(stream.name)")
    await broadcaster.stop()
    await reader.stopReconnecting()
}

private func durationSeconds(_ d: Duration) -> Double {
    let c = d.components
    return Double(c.seconds) + Double(c.attoseconds) / 1_000_000_000_000_000_000
}
