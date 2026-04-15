// MJPEGSidecar.swift
// On-demand MJPEG frame distributor for the /stream HTTP endpoint.
//
// Design: the MJPEG appsink lives inside GStreamerFrameReader's tee-split
// pipeline (not a separate GStreamer pipeline). MJPEGSidecar receives a
// GstElementRef to the appsink from GStreamerFrameReader after the main
// pipeline starts.
//
// The pull loop runs in a detached Task and pulls JPEG frames from the appsink.
// When there are no subscribers the loop still runs but broadcast() discards
// the frame immediately — cost is just nvjpegenc encoding overhead (~5% GPU).
//
// Single rtspsrc constraint: both detection and MJPEG share one RTSP/H.264
// decode path via the tee element in GStreamerFrameReader's pipeline. Opening
// a second rtspsrc connection to this camera would be dropped (camera limit
// is 1 session). This design avoids that entirely.
//
// nvjpegenc output: system-memory JPEG (malloc'd pages). gst_buffer_map on
// these is cheap — no NVMM pinning, no memory leak. The NVMM canary
// (/sys/kernel/debug/nvmap/iovmm/allocations) stays flat.
//
// Concurrency:
//   MJPEGSidecar is an actor. The appsink pull loop runs in a detached Task
//   that is started once (in attachSink) and runs until the pipeline stops.
//   Each HTTP client subscribes via subscribeFrames() and receives frames via
//   an AsyncStream<[UInt8]>.

import CGStreamer
import Logging

#if canImport(Glibc)
    import Glibc
#elseif canImport(Musl)
    import Musl
#endif

// MARK: - MJPEGClientID

/// Opaque client identifier. UInt64 counter avoids the Foundation dependency.
struct MJPEGClientID: Hashable, Sendable {
    let value: UInt64
}

// MARK: - MJPEGSidecar

/// Actor that distributes MJPEG frames from the shared GStreamer appsink
/// to zero or more HTTP /stream clients.
///
/// Lifecycle:
///   - Call `attachSink(_:)` once after GStreamerFrameReader starts.
///   - Each HTTP /stream request calls `subscribeFrames()`.
///   - Caller MUST call `unsubscribe(id:)` when done.
///   - `stop()` cancels the pull loop and finishes all subscriber streams.
actor MJPEGSidecar {

    private let logger: Logger

    // MARK: Pull task state

    private var pullTask: Task<Void, Never>?

    // MARK: Subscriber registry

    private var nextClientID: UInt64 = 0
    private var subscribers: [MJPEGClientID: AsyncStream<[UInt8]>.Continuation] = [:]

    // MARK: Init

    init() {
        self.logger = Logger(label: "MJPEGSidecar")
    }

    // MARK: Public API

    /// Wire up the appsink element from the main pipeline and start the pull loop.
    ///
    /// The `sinkRef.element` pointer must remain valid for the lifetime of the
    /// pipeline — GStreamerFrameReader owns it and releases it in stop().
    func attachSink(_ sinkRef: GstElementRef) {
        guard pullTask == nil else {
            logger.warning("attachSink called twice — ignoring")
            return
        }

        // Wrap in @unchecked Sendable so Swift 6 region-isolation allows the
        // C pointer to cross the Task.detached boundary.
        //
        // Safety: the element pointer is valid until GStreamerFrameReader.stop()
        // sets the pipeline to NULL, at which point gst_app_sink_pull_sample
        // returns NULL and the loop exits before the pointer is released.
        struct PullContext: @unchecked Sendable {
            let appSink: UnsafeMutablePointer<GstAppSink>
            let sidecar: MJPEGSidecar
        }

        let ctx = PullContext(
            appSink: wendy_gst_app_sink_cast(sinkRef.element),
            sidecar: self
        )

        logger.info("MJPEG appsink attached — starting pull loop")

        pullTask = Task.detached(priority: .background) {
            let rawSink = ctx.appSink
            let sidecar = ctx.sidecar

            while !Task.isCancelled {
                var outSample: UnsafeMutableRawPointer?
                var outData: UnsafeMutableRawPointer?
                var outSize: Int = 0

                // Blocking pull — returns when a JPEG frame is available,
                // or returns 0 when the appsink is in NULL/EOS state.
                let ok = wendy_gst_pull_sample(rawSink, &outSample, &outData, &outSize)
                guard ok == 1, let handle = outSample, let dataPtr = outData, outSize > 0 else {
                    // EOS, NULL state, or pipeline torn down — exit cleanly.
                    break
                }

                // Copy JPEG bytes into a Swift-owned array before releasing the
                // GstSample. The data pointer is invalid after wendy_gst_release_sample.
                let jpeg = Array(UnsafeBufferPointer(
                    start: dataPtr.assumingMemoryBound(to: UInt8.self),
                    count: outSize
                ))

                wendy_gst_release_sample(handle)

                // Actor hop: fan out the frame to all active subscribers.
                await sidecar.broadcast(jpeg)
            }

            await sidecar.pullLoopEnded()
        }
    }

    /// Subscribe to JPEG frames. Returns the subscriber ID + an AsyncStream.
    ///
    /// The caller MUST call `unsubscribe(id:)` when done consuming.
    func subscribeFrames() -> (id: MJPEGClientID, frames: AsyncStream<[UInt8]>) {
        let id = MJPEGClientID(value: nextClientID)
        nextClientID &+= 1

        let (stream, continuation) = AsyncStream<[UInt8]>.makeStream()
        subscribers[id] = continuation

        logger.info("MJPEG client subscribed", metadata: [
            "clientID": "\(id.value)",
            "total": "\(subscribers.count)",
        ])

        return (id, stream)
    }

    /// Unsubscribe a client. Finishes its AsyncStream.
    func unsubscribe(id: MJPEGClientID) {
        subscribers[id]?.finish()
        subscribers.removeValue(forKey: id)
        logger.info("MJPEG client unsubscribed", metadata: [
            "clientID": "\(id.value)",
            "remaining": "\(subscribers.count)",
        ])
    }

    /// Stop the pull loop and finish all subscriber streams.
    /// Call when the main pipeline is being torn down.
    func stop() {
        pullTask?.cancel()
        pullTask = nil
        for continuation in subscribers.values {
            continuation.finish()
        }
        subscribers.removeAll()
        logger.info("MJPEG sidecar stopped")
    }

    // MARK: Private — called from pull task via actor hop

    private func broadcast(_ jpeg: [UInt8]) {
        for continuation in subscribers.values {
            continuation.yield(jpeg)
        }
    }

    private func pullLoopEnded() {
        logger.info("MJPEG pull loop ended")
        for continuation in subscribers.values {
            continuation.finish()
        }
        subscribers.removeAll()
        pullTask = nil
    }
}
