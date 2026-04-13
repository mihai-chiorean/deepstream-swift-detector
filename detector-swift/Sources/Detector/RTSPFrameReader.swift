internal import Foundation
import Dispatch
import Logging

// MARK: - Frame

/// A single decoded video frame in RGB24 format.
struct Frame: Sendable {
    let data: [UInt8]
    let width: Int
    let height: Int
}

// MARK: - StreamConfig

/// Matches one entry in streams.json.
struct StreamConfig: Codable, Sendable {
    let name: String
    let url: String
    let enabled: Bool

    init(name: String, url: String, enabled: Bool = true) {
        self.name = name
        self.url = url
        self.enabled = enabled
    }
}

// MARK: - StreamsConfig

/// Top-level container that matches the streams.json format.
struct StreamsConfig: Codable, Sendable {
    let streams: [StreamConfig]

    /// Load and decode streams.json from the given file-system path.
    static func load(from path: String) throws -> StreamsConfig {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(StreamsConfig.self, from: data)
    }
}

// MARK: - RTSPFrameReader

/// Manages a single FFmpeg subprocess that decodes an RTSP stream and delivers
/// raw RGB24 frames through `nextFrame()`.
///
/// The actor restarts FFmpeg automatically when the process exits or the pipe
/// breaks, using a 2-second back-off between attempts.
actor RTSPFrameReader {

    // MARK: Configuration

    let stream: StreamConfig
    let width: Int
    let height: Int

    // MARK: Private state

    private var process: Process?
    private var pipeHandle: FileHandle?
    private var isRunning = false
    private let logger: Logger

    /// Dedicated serial queue for blocking pipe reads so they never stall
    /// the Swift cooperative thread pool (which has limited threads).
    private let readQueue = DispatchQueue(label: "rtsp-frame-reader.pipe-read")

    /// Byte count for one complete frame (RGB24 = 3 bytes per pixel).
    private var frameByteCount: Int { width * height * 3 }

    // MARK: Init

    init(stream: StreamConfig, width: Int = 1920, height: Int = 1080) {
        self.stream = stream
        self.width = width
        self.height = height
        self.logger = Logger(label: "RTSPFrameReader[\(stream.name)]")
    }

    // MARK: Lifecycle

    /// Spawn the FFmpeg subprocess. Safe to call again after `stop()`.
    func start() async throws {
        guard !isRunning else { return }
        try spawnProcess()
        isRunning = true
        logger.info("Started ffmpeg for stream", metadata: [
            "stream": "\(stream.name)",
            "url": "\(stream.url)",
            "resolution": "\(width)x\(height)",
        ])
    }

    /// Terminate the FFmpeg subprocess and close the pipe.
    func stop() {
        isRunning = false
        terminateProcess()
        logger.info("Stopped ffmpeg for stream", metadata: ["stream": "\(stream.name)"])
    }

    // MARK: Frame reading

    /// Read exactly one frame from the pipe.
    ///
    /// If FFmpeg has exited or the pipe breaks, the method waits 2 seconds and
    /// restarts FFmpeg before retrying. Throws only when `stop()` has been called.
    func nextFrame() async throws -> Frame {
        while true {
            // Ensure the process is alive before reading.
            if pipeHandle == nil || !(process?.isRunning ?? false) {
                guard isRunning else {
                    throw RTSPError.stopped
                }
                try await restartWithBackoff()
            }

            guard let handle = pipeHandle else {
                guard isRunning else { throw RTSPError.stopped }
                continue
            }

            // Capture the byte count as a plain Int before crossing isolation
            // boundaries so the detached task does not need to hop back to the actor.
            let byteCount = frameByteCount
            let frameWidth = width
            let frameHeight = height

            // Reading from the pipe is a blocking call — run it on a
            // dedicated dispatch queue (NOT the cooperative thread pool, which
            // has limited threads and would deadlock under multiple streams).
            let bytes = await withCheckedContinuation { continuation in
                readQueue.async {
                    let data = handle.readData(ofLength: byteCount)
                    continuation.resume(returning: data)
                }
            }

            if bytes.count == byteCount {
                return Frame(data: Array(bytes), width: frameWidth, height: frameHeight)
            }

            // Short read means the pipe closed (process exited).
            logger.warning(
                "Pipe closed, got \(bytes.count) bytes (expected \(byteCount)) — restarting ffmpeg",
                metadata: ["stream": "\(stream.name)"]
            )
            terminateProcess()
            guard isRunning else { throw RTSPError.stopped }
            try await restartWithBackoff()
        }
    }

    // MARK: Private helpers

    private func spawnProcess() throws {
        let pipe = Pipe()

        let proc = Process()
        // Static ffmpeg binary bundled as a wendy.json resource at /app/ffmpeg.
        // Falls back to /usr/bin/ffmpeg if present in the container image.
        let ffmpegPath = ["/app/ffmpeg", "/usr/bin/ffmpeg"]
            .first { FileManager.default.fileExists(atPath: $0) } ?? "/usr/bin/ffmpeg"
        proc.executableURL = URL(fileURLWithPath: ffmpegPath)
        proc.arguments = ffmpegArguments()
        proc.standardOutput = pipe
        // Suppress stderr from polluting our stdout; the process writes warnings there.
        proc.standardError = FileHandle.nullDevice

        // When the process terminates, log and let `nextFrame()` detect the closed pipe.
        // We use [weak self] because the actor could be deallocated before the handler fires.
        proc.terminationHandler = { [weak self] terminated in
            guard let self else { return }
            Task {
                await self.handleProcessTermination(exitCode: terminated.terminationStatus)
            }
        }

        try proc.run()

        process = proc
        pipeHandle = pipe.fileHandleForReading
    }

    private func terminateProcess() {
        if let proc = process, proc.isRunning {
            proc.terminate()
        }
        // Close our read-end of the pipe so any blocked read unblocks immediately.
        try? pipeHandle?.close()
        process = nil
        pipeHandle = nil
    }

    private func handleProcessTermination(exitCode: Int32) {
        // Only log if the reader is still supposed to be running (not a clean stop).
        guard isRunning else { return }
        logger.warning(
            "ffmpeg exited unexpectedly",
            metadata: [
                "stream": "\(stream.name)",
                "exitCode": "\(exitCode)",
            ]
        )
    }

    private func restartWithBackoff() async throws {
        guard isRunning else { throw RTSPError.stopped }
        logger.info(
            "Waiting 2 s before restarting ffmpeg",
            metadata: ["stream": "\(stream.name)"]
        )
        try await Task.sleep(for: .seconds(2))
        guard isRunning else { throw RTSPError.stopped }
        terminateProcess()
        try spawnProcess()
        logger.info(
            "Restarted ffmpeg for stream",
            metadata: ["stream": "\(stream.name)"]
        )
    }

    /// Whether the stream URL points to a local file rather than an RTSP source.
    private var isFileSource: Bool {
        let url = stream.url
        return url.hasPrefix("/") || url.hasPrefix("file://")
    }

    /// Build the argument list for the ffmpeg subprocess.
    ///
    /// For RTSP sources: TCP transport, 500 ms latency buffer, 10-second timeout.
    /// For file sources: loop infinitely at realtime speed (conference demo fallback).
    private func ffmpegArguments() -> [String] {
        var args: [String] = ["-hide_banner", "-loglevel", "warning"]

        if isFileSource {
            // Loop the file forever and pace output at realtime speed.
            args += ["-stream_loop", "-1", "-re"]
        } else {
            // RTSP-specific connection & buffering options (must appear before -i).
            // RTSP-specific connection & buffering options (must appear before -i).
            // Note: `-stimeout` was removed in ffmpeg 7.x. In 7.x, `-timeout`
            // is the socket timeout option (no longer "listen mode" as in 6.x).
            args += [
                "-rtsp_transport", "tcp",
                "-analyzeduration", "500000",   // 500 ms in microseconds
                "-probesize", "500000",         // 500 KB probe size
                "-timeout", "10000000",         // 10 s socket timeout in microseconds
            ]
            // NOTE: Hardware decode (e.g. -hwaccel cuda) requires an FFmpeg
            // build with NVDEC support, which Ubuntu 22.04's ffmpeg package
            // does not include. For now we use software decode — a future
            // optimisation is to swap to a hardware-accelerated decoder via
            // GStreamer (nvv4l2decoder) for ~30% CPU savings per stream.
        }

        args += [
            "-i", stream.url,
            "-vf", "scale=\(width):\(height)",
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "pipe:1",
        ]

        return args
    }
}

// MARK: - RTSPError

enum RTSPError: Error, Sendable {
    /// `stop()` was called while waiting for or reading a frame.
    case stopped
}

// MARK: - FrameReader

/// A `Sendable` convenience wrapper around `RTSPFrameReader` that exposes
/// frames as an `AsyncStream<Frame>`.
struct FrameReader: Sendable {
    private let reader: RTSPFrameReader

    init(stream: StreamConfig, width: Int = 1920, height: Int = 1080) {
        self.reader = RTSPFrameReader(stream: stream, width: width, height: height)
    }

    /// Start the reader and return an `AsyncStream` that yields decoded frames.
    ///
    /// The stream finishes when `stop()` is called on the underlying actor or
    /// a non-recoverable error occurs.
    func frames() -> AsyncStream<Frame> {
        AsyncStream { continuation in
            let reader = self.reader
            let task = Task {
                do {
                    try await reader.start()
                    while !Task.isCancelled {
                        let frame = try await reader.nextFrame()
                        continuation.yield(frame)
                    }
                } catch is RTSPError {
                    // Reader was stopped cleanly — finish the stream.
                } catch {
                    // Unexpected error — finish the stream so callers don't hang.
                    let logger = Logger(label: "FrameReader")
                    logger.error("Unhandled frame-reader error: \(error)")
                }
                continuation.finish()
            }

            continuation.onTermination = { _ in
                task.cancel()
                Task { await reader.stop() }
            }
        }
    }

    /// Stop the underlying FFmpeg process and finish the stream.
    func stop() async {
        await reader.stop()
    }
}
