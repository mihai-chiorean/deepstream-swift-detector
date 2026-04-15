// HTTPServer.swift
// HTTP server for the Swift DeepStream detector.
//
// Exposes:
//   GET /metrics               - Prometheus text format metrics
//   GET /health                - JSON health status
//   GET /stream                - MJPEG multipart/x-mixed-replace live video
//   GET /api/vlm_descriptions  - Recent VLM descriptions
//   GET /api/vlm_status        - VLM service status
//
// /stream: served by MJPEGSidecar. The sidecar pipeline (rtspsrc →
// nvv4l2decoder → nvvideoconvert → videorate → nvjpegenc → appsink) starts
// on the first client connection and stops when the last disconnects.
// gst_buffer_map is called only on system-memory JPEG buffers output by
// nvjpegenc — no NVMM pinning, no leak.
//
// Port: 9090 (matching Python detector default)

import Hummingbird
import Logging
import NIOCore

#if canImport(FoundationEssentials)
    internal import FoundationEssentials
#else
    internal import Foundation
#endif

// MARK: - DetectorState

/// Shared mutable state that the HTTP server reads from the detection pipeline.
actor DetectorState: Sendable {

    // MARK: MJPEG sidecar

    private(set) var mjpegSidecar: MJPEGSidecar?

    func setMJPEGSidecar(_ sidecar: MJPEGSidecar) {
        self.mjpegSidecar = sidecar
    }

    // MARK: Legacy frame store (retained for forward compatibility)

    private(set) var latestJPEGFrame: [UInt8]?
    private(set) var mjpegClientCount: Int = 0

    var shouldExtractFrames: Bool { mjpegClientCount > 0 }

    func setFrame(_ jpeg: [UInt8]) {
        latestJPEGFrame = jpeg
    }

    func getFrame() -> [UInt8]? { latestJPEGFrame }
}

// MARK: - AllOriginsMiddleware

struct AllOriginsMiddleware<Context: RequestContext>: RouterMiddleware {
    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        var response = try await next(request, context)
        response.headers[.accessControlAllowOrigin] = "*"
        response.headers[.accessControlAllowMethods] = "GET, POST, OPTIONS"
        response.headers[.accessControlAllowHeaders] = "Content-Type"
        return response
    }
}

// MARK: - Router builder

func buildRouter(
    state: DetectorState,
    metrics: MetricsRegistry,
    vlmClient: VLMClient
) -> Router<BasicRequestContext> {

    let router = Router(context: BasicRequestContext.self)
    router.middlewares.add(AllOriginsMiddleware())

    // ------------------------------------------------------------------
    // GET /metrics  - Prometheus text exposition format
    // ------------------------------------------------------------------
    router.get("metrics") { _, _ -> Response in
        let body = metrics.render()
        var headers = HTTPFields()
        headers[.contentType] = "text/plain; version=0.0.4; charset=utf-8"
        return Response(
            status: .ok,
            headers: headers,
            body: ResponseBody(byteBuffer: ByteBuffer(string: body))
        )
    }

    // ------------------------------------------------------------------
    // GET /health  - JSON health check
    // ------------------------------------------------------------------
    router.get("health") { _, _ -> Response in
        let timestamp = Date().ISO8601Format()
        let json = "{\"status\":\"healthy\",\"timestamp\":\"\(timestamp)\"}"
        var headers = HTTPFields()
        headers[.contentType] = "application/json"
        return Response(
            status: .ok,
            headers: headers,
            body: ResponseBody(byteBuffer: ByteBuffer(string: json))
        )
    }

    // ------------------------------------------------------------------
    // GET /stream  - MJPEG multipart/x-mixed-replace video stream
    //
    // Each HTTP request opens a persistent streaming connection. The
    // MJPEGSidecar pipeline starts on the first connection and stops
    // when the last client disconnects.
    //
    // Frame format (RFC 2046 multipart):
    //   --mjpegframe\r\n
    //   Content-Type: image/jpeg\r\n
    //   Content-Length: <N>\r\n
    //   \r\n
    //   <JPEG bytes>
    //   \r\n
    // ------------------------------------------------------------------
    router.get("stream") { _, _ -> Response in
        guard let sidecar = await state.mjpegSidecar else {
            let json = "{\"error\":\"MJPEG sidecar not initialised\",\"status\":503}"
            var headers = HTTPFields()
            headers[.contentType] = "application/json"
            return Response(
                status: .serviceUnavailable,
                headers: headers,
                body: ResponseBody(byteBuffer: ByteBuffer(string: json))
            )
        }

        let (clientID, frameStream) = await sidecar.subscribeFrames()

        var headers = HTTPFields()
        headers[.contentType] = "multipart/x-mixed-replace; boundary=mjpegframe"
        headers[.cacheControl] = "no-cache, no-store, must-revalidate"

        let body = ResponseBody { writer in
            // Stream JPEG frames until the client disconnects or the sidecar ends.
            // When writing throws (client disconnected), we break out, unsubscribe,
            // and call finish so Hummingbird closes the connection cleanly.
            var writeError: (any Error)?
            for await jpeg in frameStream {
                var part = ByteBuffer()
                part.writeString("--mjpegframe\r\n")
                part.writeString("Content-Type: image/jpeg\r\n")
                part.writeString("Content-Length: \(jpeg.count)\r\n")
                part.writeString("\r\n")
                part.writeBytes(jpeg)
                part.writeString("\r\n")
                do {
                    try await writer.write(part)
                } catch {
                    // Client disconnected — normal, not an error to propagate.
                    writeError = error
                    break
                }
            }

            // Unsubscribe regardless of how the loop exited.
            await sidecar.unsubscribe(id: clientID)

            // Close the response body. If the client already disconnected,
            // this will throw — that's fine, Hummingbird handles it.
            if writeError == nil {
                try await writer.finish(nil)
            } else {
                // Swallow the finish error — client is gone.
                try? await writer.finish(nil)
            }
        }

        return Response(status: .ok, headers: headers, body: body)
    }

    // ------------------------------------------------------------------
    // GET /api/vlm_descriptions  - Recent VLM descriptions
    // ------------------------------------------------------------------
    router.get("api/vlm_descriptions") { _, _ -> Response in
        let descriptions = await vlmClient.getRecentDescriptions()
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let wrapper = VLMDescriptionsResponse(
            count: descriptions.count,
            descriptions: descriptions
        )
        let data = (try? encoder.encode(wrapper)) ?? Data("{}".utf8)
        var headers = HTTPFields()
        headers[.contentType] = "application/json"
        return Response(
            status: .ok,
            headers: headers,
            body: ResponseBody(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    // ------------------------------------------------------------------
    // GET /api/vlm_status  - VLM service availability
    // ------------------------------------------------------------------
    router.get("api/vlm_status") { _, _ -> Response in
        let available = await vlmClient.available
        let json = "{\"available\":\(available)}"
        var headers = HTTPFields()
        headers[.contentType] = "application/json"
        return Response(
            status: .ok,
            headers: headers,
            body: ResponseBody(byteBuffer: ByteBuffer(string: json))
        )
    }

    return router
}

// MARK: - VLM response types

private struct VLMDescriptionsResponse: Encodable {
    let count: Int
    let descriptions: [VLMDescription]
}

// MARK: - Server entry point

func startHTTPServer(
    state: DetectorState,
    metrics: MetricsRegistry,
    vlmClient: VLMClient,
    port: Int = 9090
) async throws {
    var logger = Logger(label: "detector.http")
    logger.logLevel = .info

    let router = buildRouter(state: state, metrics: metrics, vlmClient: vlmClient)

    let app = Application(
        router: router,
        configuration: .init(
            address: .hostname("0.0.0.0", port: port)
        ),
        logger: logger
    )

    logger.info(
        "HTTP server listening",
        metadata: ["address": "0.0.0.0:\(port)"]
    )

    try await app.runService()
}
