// HTTPServer.swift
// HTTP server for the Swift DeepStream detector.
//
// Exposes:
//   GET /metrics               - Prometheus text format metrics
//   GET /health                - JSON health status
//   GET /stream                - 301 redirect to /webrtc/relayed/whep (WHEP endpoint)
//   GET /detections            - WebSocket: per-frame detection JSON stream
//   GET /api/vlm_descriptions  - Recent VLM descriptions
//   GET /api/vlm_status        - VLM service availability
//
// Stage 2: The /stream MJPEG endpoint is replaced by a 301 redirect to
// /webrtc/relayed/whep (the mediamtx WHEP endpoint via monitor_proxy).
// The MJPEGSidecar and valve wiring are gone. Video is delivered by mediamtx
// via WebRTC; the detector only publishes detections.
//
// /detections: WebSocket endpoint. Each connected browser tab receives a JSON
// message per detection frame:
//   { frameNum, ptsNs, timestampMs, detections: [{classId, confidence, x, y, w, h, trackId}] }
// The DetectionBroadcaster actor fans out from the single DetectionStream to N
// clients; a slow client drops stale frames (bufferingNewest(4) per client).
//
// Port: 9090 (matching Python detector default)
//
// WebSocket setup: HummingbirdWebSocket requires a BasicWebSocketRequestContext
// (superset of BasicRequestContext) and an Application configured with
// HTTPServerBuilder.http1WebSocketUpgrade(webSocketRouter:). Both HTTP and WS
// routes are registered on the same BasicWebSocketRequestContext router.

import Hummingbird
import HummingbirdWebSocket
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

    // MARK: Detection broadcaster

    private(set) var detectionBroadcaster: DetectionBroadcaster?

    func setDetectionBroadcaster(_ broadcaster: DetectionBroadcaster) {
        self.detectionBroadcaster = broadcaster
    }
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
) -> Router<BasicWebSocketRequestContext> {

    let router = Router(context: BasicWebSocketRequestContext.self)
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
    // GET /stream  - 301 redirect to WHEP WebRTC endpoint
    //
    // Stage 2: MJPEG is gone. Anything still hitting /stream is redirected
    // to the mediamtx WHEP endpoint via monitor_proxy. RTCPeerConnection
    // clients should POST directly to /webrtc/relayed/whep.
    // ------------------------------------------------------------------
    router.get("stream") { _, _ -> Response in
        var headers = HTTPFields()
        headers[.location] = "/webrtc/relayed/whep"
        return Response(
            status: .movedPermanently,
            headers: headers,
            body: ResponseBody(byteBuffer: ByteBuffer(string: ""))
        )
    }

    // ------------------------------------------------------------------
    // GET /detections  - WebSocket: per-frame detection JSON stream
    //
    // Each connected browser tab receives one JSON message per detection
    // frame. The DetectionBroadcaster fans out from the single shared
    // DetectionStream; slow clients drop stale frames rather than blocking
    // the detector.
    //
    // Message schema:
    //   {
    //     "frameNum": 42,
    //     "ptsNs": 1234567890,
    //     "timestampMs": 1713301234567,
    //     "detections": [
    //       {"classId":2,"confidence":0.87,"x":100,"y":200,
    //        "w":150,"h":80,"trackId":3}
    //     ]
    //   }
    //
    // ptsNs is -1 when no GStreamer PTS is available (GST_CLOCK_TIME_NONE).
    //
    // Handler pattern:
    //   - defer { unsubscribe } ensures cleanup on TCP drop / tab close.
    //   - Inbound is drained in a child task so the WebSocket channel stays
    //     healthy (pings, close frames). Outbound sends detection JSON.
    //   - Write errors (TCP drop) are caught and break the loop cleanly.
    // ------------------------------------------------------------------
    router.ws("detections") { inbound, outbound, _ in
        guard let broadcaster = await state.detectionBroadcaster else {
            return
        }

        let (clientID, messageStream) = await broadcaster.subscribe()

        // withTaskGroup runs inbound drain + outbound send concurrently.
        // The group exits when BOTH tasks complete. We cancel the group
        // (by throwing) when the outbound loop finishes so the drain task
        // also exits.
        await withTaskGroup(of: Void.self) { group in
            // Task 1: drain inbound frames (pings / browser → server messages).
            // The WebSocket framework handles pings/pongs internally inside next().
            // We drain here to ensure close frames are processed and the connection
            // is properly acknowledged. We don't use client-sent data.
            group.addTask {
                do { for try await _ in inbound {} } catch {}
            }

            // Task 2: send detection JSON to the client.
            group.addTask {
                do {
                    for await message in messageStream {
                        try await outbound.write(.text(message))
                    }
                } catch {
                    // TCP drop or client close — normal. Break out of loop.
                }
            }

            // Wait for either task to finish (client disconnect or stream end),
            // then cancel the other task.
            await group.next()
            group.cancelAll()
        }

        // Always unsubscribe after the handler exits, regardless of how we got here.
        await broadcaster.unsubscribe(id: clientID)
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

    // Use http1WebSocketUpgrade so the same router handles both HTTP routes
    // and WebSocket upgrade requests matched by path (/detections).
    let app = Application(
        router: router,
        server: .http1WebSocketUpgrade(webSocketRouter: router),
        configuration: .init(
            address: .hostname("0.0.0.0", port: port)
        ),
        logger: logger
    )

    logger.info(
        "HTTP server listening (with WebSocket /detections)",
        metadata: ["address": "0.0.0.0:\(port)"]
    )

    try await app.runService()
}
