// VLMClient.swift
// Async client for a local llama.cpp server running a vision-language model.
//
// Architecture:
//   The llama.cpp server runs as a Docker sidecar on port 8090, serving an
//   OpenAI-compatible /v1/chat/completions endpoint with multimodal support.
//   This client sends JPEG-encoded image crops with a text prompt and receives
//   natural-language descriptions.
//
// Model: Qwen3-VL-2B Q4_K_M (~1.5 GB GPU) — fits alongside YOLO26 (~200 MB)
//        on Jetson Orin Nano 8GB with headroom to spare.
//
// GPU arbitration: The llama.cpp server manages its own CUDA context. A
//   semaphore in this actor limits concurrent VLM requests to 1, preventing
//   memory pressure spikes when multiple tracks finalize simultaneously.

import AsyncHTTPClient
import NIOCore
import Logging

#if canImport(FoundationEssentials)
    internal import FoundationEssentials
#else
    internal import Foundation
#endif

// MARK: - VLMClient

/// Actor that sends image crops to a local VLM service for description.
///
/// All requests are serialised through the actor to avoid flooding the GPU
/// with concurrent VLM inference jobs.
actor VLMClient {

    // MARK: Configuration

    /// Base URL of the llama.cpp server (e.g. "http://127.0.0.1:8090").
    let baseURL: String

    /// HTTP client for making requests to the VLM server.
    private let httpClient: HTTPClient

    private let logger: Logger

    /// Whether the VLM server has been confirmed reachable.
    private var isAvailable: Bool = false

    /// Recent descriptions, kept for the HTTP API.
    private var recentDescriptions: [VLMDescription] = []

    /// Maximum number of recent descriptions to retain.
    private let maxDescriptions = 50

    // MARK: Init

    init(baseURL: String = "http://127.0.0.1:8090") {
        self.baseURL = baseURL
        self.httpClient = HTTPClient(eventLoopGroupProvider: .singleton)
        self.logger = Logger(label: "VLMClient")
    }

    deinit {
        try? httpClient.syncShutdown()
    }

    // MARK: Public API

    /// Describe an image crop using the VLM.
    ///
    /// - Parameters:
    ///   - jpegData: JPEG-encoded image bytes.
    ///   - prompt: Text prompt describing what to look for.
    ///   - trackId: The tracker ID for this object (for logging/storage).
    ///   - label: The YOLO class label (e.g. "person", "car").
    /// - Returns: The VLM's text description, or nil if the server is unavailable.
    func describe(
        jpegData: [UInt8],
        prompt: String,
        trackId: Int,
        label: String
    ) async -> String? {
        let b64 = Data(jpegData).base64EncodedString()

        let requestBody = ChatCompletionRequest(
            messages: [
                ChatMessage(role: "user", content: [
                    ContentPart(type: "image_url", text: nil,
                                imageURL: ImageURL(url: "data:image/jpeg;base64,\(b64)")),
                    ContentPart(type: "text", text: prompt, imageURL: nil),
                ])
            ],
            maxTokens: 256,
            temperature: 0.1
        )

        do {
            let bodyData = try JSONEncoder().encode(requestBody)

            var request = HTTPClientRequest(url: "\(baseURL)/v1/chat/completions")
            request.method = .POST
            request.headers.add(name: "Content-Type", value: "application/json")
            request.body = .bytes(ByteBuffer(bytes: bodyData))

            let response = try await httpClient.execute(request, timeout: .seconds(60))
            let responseBody = try await response.body.collect(upTo: 1024 * 1024)  // 1 MB max
            let responseData = Data(responseBody.readableBytesView)

            let completion = try JSONDecoder().decode(
                ChatCompletionResponse.self,
                from: responseData
            )

            guard let text = completion.choices.first?.message.content else {
                return nil
            }

            isAvailable = true

            // Store the description for the HTTP API.
            let description = VLMDescription(
                trackId: trackId,
                label: label,
                description: text,
                timestamp: Date()
            )
            recentDescriptions.append(description)
            if recentDescriptions.count > maxDescriptions {
                recentDescriptions.removeFirst()
            }

            logger.info("VLM described track", metadata: [
                "trackId": "\(trackId)",
                "label": "\(label)",
                "description": "\(text.prefix(100))",
            ])

            return text

        } catch {
            logger.warning("VLM request failed: \(error)")
            isAvailable = false
            return nil
        }
    }

    /// No-op stub used when the VLM crop path is disabled.
    ///
    /// Returns a runtime error string rather than nil so callers can log a
    /// meaningful message instead of silently dropping the event.
    ///
    /// TODO: VLM path — remove this stub and call describe(...) once
    ///       NvBufSurfaceMap-based crop extraction is implemented (plan §8 option a).
    func describeDisabled(trackId: Int) async -> String? {
        logger.debug("VLM temporarily disabled — skipping crop for track \(trackId)")
        return nil
    }

    /// Check whether the VLM server is reachable.
    func checkHealth() async -> Bool {
        do {
            var request = HTTPClientRequest(url: "\(baseURL)/health")
            request.method = .GET
            let response = try await httpClient.execute(request, timeout: .seconds(5))
            isAvailable = response.status == .ok
        } catch {
            isAvailable = false
        }
        return isAvailable
    }

    /// Returns the current availability status.
    var available: Bool { isAvailable }

    /// Returns recent VLM descriptions for the HTTP API.
    func getRecentDescriptions() -> [VLMDescription] {
        recentDescriptions
    }
}

// MARK: - VLM Description storage

/// A VLM-generated description of a tracked object.
struct VLMDescription: Codable, Sendable {
    let trackId: Int
    let label: String
    let description: String
    let timestamp: Date
}

// MARK: - OpenAI-compatible request/response types

private struct ChatCompletionRequest: Encodable {
    let model: String = "qwen3-vl"
    let messages: [ChatMessage]
    let maxTokens: Int
    let temperature: Float

    enum CodingKeys: String, CodingKey {
        case model, messages
        case maxTokens = "max_tokens"
        case temperature
    }
}

private struct ChatMessage: Encodable {
    let role: String
    let content: [ContentPart]
}

private struct ContentPart: Encodable {
    let type: String
    let text: String?
    let imageURL: ImageURL?

    enum CodingKeys: String, CodingKey {
        case type, text
        case imageURL = "image_url"
    }
}

private struct ImageURL: Encodable {
    let url: String
}

private struct ChatCompletionResponse: Decodable {
    let choices: [Choice]

    struct Choice: Decodable {
        let message: Message
    }

    struct Message: Decodable {
        let content: String?
    }
}
