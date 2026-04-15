// Detector.swift
// Main entry point for the Swift DeepStream object detector.
//
// Ties together:
//   - DeepStream pipeline (GStreamerFrameReader) — nvinfer + nvtracker in-pipeline,
//     plus an MJPEG sidecar branch via tee (single rtspsrc)
//   - Pad-probe-driven detection stream (AsyncStream<[Detection]>)
//   - MJPEG distributor (MJPEGSidecar) — pulls from the tee branch's appsink
//   - Track-finalizer for VLM integration (currently disabled)
//   - Hummingbird HTTP server on :9090 (metrics, health, stream)
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

        // MJPEGSidecar is created upfront and registered with DetectorState so
        // the HTTP server can hand it to /stream clients immediately. The sidecar
        // is wired to the actual appsink after the GStreamer pipeline starts.
        let detectorState = DetectorState()
        let mjpegSidecar = MJPEGSidecar()
        await detectorState.setMJPEGSidecar(mjpegSidecar)
        logger.info("MJPEG sidecar created")

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
                        mjpegSidecar: mjpegSidecar,
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

// MARK: - Per-stream detection loop

private func runStreamDetectionLoop(
    stream: StreamConfig,
    mjpegSidecar: MJPEGSidecar,
    trackFinalizer: TrackFinalizer,
    logger: Logger
) async {
    var logger = logger
    logger[metadataKey: "stream"] = "\(stream.name)"

    let fps = fpsGauge(stream: stream.name)
    let framesProcessed = framesProcessedCounter(stream: stream.name)
    var classCounterCache: [String: CounterMetric] = [:]
    var frameCount: UInt64 = 0
    var fpsWindowStart = ContinuousClock.now
    var previousTrackerIds = Set<Int>()

    let reader = GStreamerReader(stream: stream)

    logger.info("Starting DeepStream detection loop")

    do {
        let (detectionStream, mjpegSinkRef) = try await reader.start()

        // Wire up the MJPEG sidecar to the appsink in the tee-split pipeline.
        // This avoids opening a second RTSP connection (camera limit = 1 session).
        if let sinkRef = mjpegSinkRef {
            await mjpegSidecar.attachSink(sinkRef)
            logger.info("MJPEG sidecar wired to pipeline appsink")
        } else {
            logger.warning("No MJPEG appsink in pipeline — /stream will not produce frames")
        }

        for await detections in detectionStream {
            let currentTrackerIds = Set(detections.compactMap(\.trackId))
            let disappearedIds = previousTrackerIds.subtracting(currentTrackerIds)
            for goneId in disappearedIds {
                await trackFinalizer.submitDisappeared(trackId: goneId)
            }
            previousTrackerIds = currentTrackerIds

            frameCount += 1
            framesProcessed.inc()

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

            let now = ContinuousClock.now
            let elapsedSeconds = durationSeconds(now - fpsWindowStart)
            if elapsedSeconds >= 1.0 {
                fps.set(Double(frameCount) / elapsedSeconds)
                frameCount = 0
                fpsWindowStart = now
            }
        }

    } catch {
        logger.error("Failed to start DeepStream pipeline: \(error)")
    }

    logger.warning("Detection loop ended for stream \(stream.name)")
    await mjpegSidecar.stop()
    await reader.stop()
}

private func durationSeconds(_ d: Duration) -> Double {
    let c = d.components
    return Double(c.seconds) + Double(c.attoseconds) / 1_000_000_000_000_000_000
}
