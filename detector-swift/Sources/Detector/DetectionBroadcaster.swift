// DetectionBroadcaster.swift
// Fan-out actor that distributes detection frames to N WebSocket clients.
//
// Design mirrors MJPEGSidecar: a per-client continuation map. The caller
// (runStreamDetectionLoop) is the SOLE consumer of DetectionStream and calls
// distribute(_:) on each frame. This avoids the dual-consumer problem that
// would arise from giving the broadcaster its own background loop over the
// same AsyncStream.
//
// Slow clients get stale frames dropped (AsyncStream bufferingNewest(4) per
// client). A slow client CANNOT block the detector: distribute() yields to
// each client's per-client AsyncStream continuation without awaiting delivery.
//
// WebSocket message schema (JSON):
//   {
//     "frameNum": 42,
//     "ptsNs": 1234567890,
//     "timestampMs": 1713301234567,
//     "detections": [
//       {"classId": 2, "confidence": 0.87, "x": 100, "y": 200,
//        "w": 150, "h": 80, "trackId": 3}
//     ]
//   }
//
// ptsNs is -1 when the buffer carried no GStreamer PTS (GST_CLOCK_TIME_NONE).
// timestampMs is wall-clock milliseconds at probe time (Swift side).
//
// Stage 2 note: ptsNs is load-bearing for frame-accurate canvas sync via
// requestVideoFrameCallback in the browser.

import Logging

#if canImport(FoundationEssentials)
    internal import FoundationEssentials
#else
    internal import Foundation
#endif

// MARK: - DetectionClientID

/// Opaque per-WebSocket-client identifier.
struct DetectionClientID: Hashable, Sendable {
    let value: UInt64
}

// MARK: - DetectionBroadcaster

/// Actor that receives detection frames from the detection loop and fans them
/// out as JSON to N WebSocket clients.
///
/// The caller (runStreamDetectionLoop) is the sole consumer of DetectionStream.
/// On each frame it calls `distribute(_:)`, which serialises the frame to JSON
/// and yields it to every subscribed client's per-client AsyncStream.
///
/// Lifecycle:
///   - Create before the pipeline starts.
///   - Each WebSocket connection calls `subscribe()` and receives an
///     `AsyncStream<String>` of JSON messages.
///   - The handler MUST call `unsubscribe(id:)` when the connection closes.
///   - `stop()` finishes all client streams.
actor DetectionBroadcaster {

    private let logger: Logger

    // MARK: Client registry

    private var nextClientID: UInt64 = 0
    private var clients: [DetectionClientID: AsyncStream<String>.Continuation] = [:]

    // MARK: Init

    init() {
        self.logger = Logger(label: "DetectionBroadcaster")
    }

    // MARK: Public API

    /// Distribute one detection frame to all subscribed WebSocket clients.
    ///
    /// Called from the detection loop on every frame. Non-blocking: each client
    /// continuation uses .bufferingNewest(4) so stale frames are dropped
    /// without blocking the caller.
    func distribute(_ frame: DetectionFrame) {
        guard !clients.isEmpty else { return }
        let json = Self.encodeFrame(frame)
        for continuation in clients.values {
            continuation.yield(json)
        }
    }

    /// Subscribe a new WebSocket client. Returns (id, AsyncStream<String>).
    ///
    /// Each message is a JSON string per the schema above.
    /// The caller MUST call `unsubscribe(id:)` when the WebSocket closes.
    func subscribe() -> (id: DetectionClientID, messages: AsyncStream<String>) {
        let id = DetectionClientID(value: nextClientID)
        nextClientID &+= 1

        let (stream, continuation) = AsyncStream<String>.makeStream(
            bufferingPolicy: .bufferingNewest(4)
        )
        clients[id] = continuation

        logger.info("Detection WebSocket client subscribed", metadata: [
            "clientID": "\(id.value)",
            "total": "\(clients.count)",
        ])

        return (id, stream)
    }

    /// Unsubscribe a client and finish its AsyncStream.
    func unsubscribe(id: DetectionClientID) {
        clients[id]?.finish()
        clients.removeValue(forKey: id)

        logger.info("Detection WebSocket client unsubscribed", metadata: [
            "clientID": "\(id.value)",
            "remaining": "\(clients.count)",
        ])
    }

    /// Finish all client streams (call when the pipeline stops).
    func stop() {
        for continuation in clients.values {
            continuation.finish()
        }
        clients.removeAll()
        logger.info("Detection broadcaster stopped")
    }

    // MARK: JSON encoding (hand-rolled — avoids Foundation JSONEncoder per-frame overhead)

    private static func encodeFrame(_ frame: DetectionFrame) -> String {
        let nowMs = Int64(Date().timeIntervalSince1970 * 1000)
        let frameNum = frame.detections.first?.frameNum ?? 0
        let ptsNs = frame.timing.ptsNs

        var json = "{\"frameNum\":\(frameNum),\"ptsNs\":\(ptsNs),\"timestampMs\":\(nowMs),\"detections\":["

        for (i, det) in frame.detections.enumerated() {
            if i > 0 { json += "," }
            json += "{"
            json += "\"classId\":\(det.classId),"
            json += "\"confidence\":\(String(format: "%.4f", det.confidence)),"
            json += "\"x\":\(Int(det.x)),"
            json += "\"y\":\(Int(det.y)),"
            json += "\"w\":\(Int(det.width)),"
            json += "\"h\":\(Int(det.height)),"
            json += "\"trackId\":\(det.trackId ?? 0)"
            json += "}"
        }

        json += "]}"
        return json
    }
}
