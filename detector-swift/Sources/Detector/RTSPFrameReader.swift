// RTSPFrameReader.swift
// Retained for StreamConfig and StreamsConfig definitions used by GStreamerFrameReader.
//
// The RTSPFrameReader actor (FFmpeg subprocess RTSP decode) has been removed.
// The DeepStream pipeline in GStreamerFrameReader.swift handles RTSP decode
// via rtspsrc + nvv4l2decoder in-pipeline. See PORT_PLAN.md §4.

internal import Foundation
import Logging

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
