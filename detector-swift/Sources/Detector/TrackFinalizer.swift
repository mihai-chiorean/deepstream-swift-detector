// TrackFinalizer.swift
// Bridges the nvtracker ID-disappearance signal and the VLM client.
//
// When the Detector loop detects that a tracker ID has disappeared (the object
// left the scene), it calls submitDisappeared(trackId:). This component queues
// a metadata-only record and, when VLM is re-enabled, would send a crop for
// description.
//
// VLM path status: DISABLED for this milestone.
//   Crop extraction requires NvBufSurfaceMap on an NVMM surface. That path is
//   deferred until the pipeline acquires a surface reference at finalization
//   time (plan §8 option a). For now TrackFinalizer logs the event and no-ops.
//
// TODO: VLM path — re-enable once NvBufSurfaceMap-based crop is implemented
//       (plan §8 option a).

import Logging

// MARK: - FinalizedTrackRecord

/// Metadata record for a finalized track (no frame bytes — VLM crop deferred).
private struct FinalizedTrackRecord: Sendable {
    let trackId: Int
}

// MARK: - TrackFinalizer

/// Manages VLM description requests for finalized tracks.
///
/// Currently operates in metadata-only mode: VLM requests are no-ops returning
/// an error until the NvBufSurfaceMap crop path is implemented.
actor TrackFinalizer {

    private let vlmClient: VLMClient
    private let logger: Logger

    init(vlmClient: VLMClient) {
        self.vlmClient = vlmClient
        self.logger = Logger(label: "TrackFinalizer")
    }

    // MARK: Public API

    /// Notify that a tracker ID has disappeared from the detection stream.
    ///
    /// This is the entry point for track finalization. In the current milestone
    /// the VLM crop path is disabled and this method only logs the event.
    ///
    /// - Parameter trackId: The nvtracker-assigned object ID that disappeared.
    func submitDisappeared(trackId: Int) async {
        logger.info("Track finalized", metadata: [
            "trackId": "\(trackId)",
        ])

        // TODO: VLM path — re-enable once NvBufSurfaceMap-based crop is
        // implemented (plan §8 option a). Expected call:
        //   let crop = extractCropViaNvBufSurface(trackId: trackId)
        //   _ = await vlmClient.describe(jpegData: crop, prompt: ...,
        //                                trackId: trackId, label: ...)
        _ = await vlmClient.describeDisabled(trackId: trackId)
    }
}
