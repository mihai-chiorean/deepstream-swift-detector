// TrackFinalizer.swift
// Bridges the object tracker and VLM client.
//
// When the IOUTracker prunes a confirmed track (the object left the scene),
// this component crops the best frame and sends it to the VLM for a
// natural-language description. Requests are bounded by a queue with
// backpressure — if the VLM is slower than tracks are finalizing, the
// oldest pending requests are dropped.

import Logging

#if canImport(FoundationEssentials)
    internal import FoundationEssentials
#else
    internal import Foundation
#endif

// MARK: - TrackFinalizer

/// Manages the VLM description pipeline for finalized tracks.
///
/// The finalizer maintains a bounded queue of pending VLM requests and
/// processes them serially to avoid GPU memory pressure. It runs as a
/// long-lived Task that drains the queue.
actor TrackFinalizer {

    // MARK: Configuration

    private let vlmClient: VLMClient
    private let logger: Logger

    /// Maximum number of pending VLM requests before dropping oldest.
    private let maxPending = 10

    // MARK: Queue state

    private var pending: [PendingDescription] = []
    private var processingTask: Task<Void, Never>?

    // MARK: Init

    init(vlmClient: VLMClient) {
        self.vlmClient = vlmClient
        self.logger = Logger(label: "TrackFinalizer")
    }

    // MARK: Public API

    /// Submit a finalized track for VLM description.
    ///
    /// The crop is extracted from the provided frame data using the track's
    /// best bounding box. If the queue is full, the oldest request is dropped.
    ///
    /// - Parameters:
    ///   - track: The finalized track metadata.
    ///   - frame: The current video frame (may not be the best frame, but is
    ///     used as a fallback — ideally we'd cache the best frame per track,
    ///     but that requires significant memory. For v1, the current frame
    ///     with the tracked bbox is a reasonable approximation).
    ///   - labels: Class label list for human-readable names.
    func submit(
        track: FinalizedTrack,
        frame: Frame,
        labels: [String]
    ) {
        // Crop the region of interest from the frame.
        guard let jpegCrop = cropAndEncode(
            frame: frame,
            bbox: track.bestBBox
        ) else {
            logger.warning("Failed to crop frame for track \(track.id)")
            return
        }

        let label = track.classId < labels.count
            ? labels[track.classId]
            : "\(track.classId)"

        let prompt = promptForClass(label)

        let request = PendingDescription(
            trackId: track.id,
            label: label,
            jpegData: jpegCrop,
            prompt: prompt
        )

        // Bounded queue with backpressure — drop oldest if full.
        if pending.count >= maxPending {
            let dropped = pending.removeFirst()
            logger.info("VLM queue full, dropping track \(dropped.trackId)")
        }
        pending.append(request)

        // Ensure the processing loop is running.
        if processingTask == nil {
            processingTask = Task { await processQueue() }
        }
    }

    // MARK: Private

    /// Drain the pending queue, sending each crop to the VLM serially.
    private func processQueue() async {
        while !pending.isEmpty {
            let request = pending.removeFirst()

            _ = await vlmClient.describe(
                jpegData: request.jpegData,
                prompt: request.prompt,
                trackId: request.trackId,
                label: request.label
            )
        }
        processingTask = nil
    }

    /// Crop a bounding box region from an RGB frame and JPEG-encode it.
    ///
    /// Uses the same FrameRenderer JPEG encoder (libturbojpeg) that the
    /// MJPEG stream uses.
    private func cropAndEncode(frame: Frame, bbox: BBox) -> [UInt8]? {
        let x0 = max(0, Int(bbox.x))
        let y0 = max(0, Int(bbox.y))
        let x1 = min(frame.width, Int(bbox.x + bbox.width))
        let y1 = min(frame.height, Int(bbox.y + bbox.height))

        let cropW = x1 - x0
        let cropH = y1 - y0
        guard cropW > 0, cropH > 0 else { return nil }

        // Extract the crop from the packed RGB frame.
        var cropData = [UInt8](repeating: 0, count: cropW * cropH * 3)
        for row in 0 ..< cropH {
            let srcOffset = ((y0 + row) * frame.width + x0) * 3
            let dstOffset = row * cropW * 3
            cropData.withUnsafeMutableBufferPointer { dst in
                frame.data.withUnsafeBufferPointer { src in
                    dst.baseAddress!.advanced(by: dstOffset)
                        .update(from: src.baseAddress!.advanced(by: srcOffset),
                                count: cropW * 3)
                }
            }
        }

        return JPEGEncoder.encode(rgb: cropData, width: cropW, height: cropH, quality: 90)
    }

    /// Generate a context-appropriate prompt for the VLM based on object class.
    private func promptForClass(_ label: String) -> String {
        switch label {
        case "person":
            return "Describe this person briefly: what they are wearing, what they are doing, and any notable features. Respond in one sentence."
        case "car", "truck", "bus":
            return "Describe this vehicle briefly: color, type, make if identifiable, and any notable features. Respond in one sentence."
        case "bicycle", "motorcycle":
            return "Describe this vehicle and its rider briefly. Respond in one sentence."
        default:
            return "Describe what you see in this image briefly. Respond in one sentence."
        }
    }
}

// MARK: - PendingDescription

private struct PendingDescription {
    let trackId: Int
    let label: String
    let jpegData: [UInt8]
    let prompt: String
}
