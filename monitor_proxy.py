#!/usr/bin/env python3
"""Reverse proxy + static server for monitor.html.

- GET /               → serves monitor.html
- GET /detector/<p>   → proxies to http://JETSON:9090/<p>
- GET /gpu/<p>        → proxies to http://JETSON:9091/<p>
- GET|POST|PATCH|OPTIONS|DELETE /webrtc/<p>
                      → proxies to http://JETSON:8889/<p>
                        (mediamtx WebRTC/WHEP signalling — HTTP only)

Streaming MJPEG (/detector/stream) is forwarded chunked, never buffered.

WebSocket upgrade:
  Paths under /detector/ that arrive with "Upgrade: websocket" are tunnelled
  as a raw TCP byte-pipe to JETSON:9090.  No ws:// library is needed — once
  the HTTP handshake is forwarded and the upstream responds with 101, both
  sockets are spliced bidirectionally until either side closes.  This covers
  the /detector/detections WebSocket endpoint in the Swift detector.

WebRTC notes:
  - WHEP signalling (HTTP POST /webrtc/relayed/whep) is proxied here.
  - The actual video flow is UDP RTP between the browser and the Jetson
    directly (ICE candidates advertise 192.168.68.70:8189 and 10.42.0.2:8189).
  - No UDP proxying is done or needed: the macbook browser connects to
    192.168.68.70:8189 directly over WiFi (same /22 subnet).
  - PATCH (ICE trickle) and DELETE (teardown) are also forwarded.
"""
import http.server
import socketserver
import socket
import select
import threading
import urllib.request
import urllib.error
import sys
import os

JETSON = os.environ.get("JETSON_HOST", "10.42.0.2")
PORT   = int(os.environ.get("PROXY_PORT", "8001"))
HERE   = os.path.dirname(os.path.abspath(__file__))

UPSTREAMS = {
    "/detector/": f"http://{JETSON}:9090/",
    "/gpu/":      f"http://{JETSON}:9091/",
    "/webrtc/":   f"http://{JETSON}:8889/",
}

# Upstream host+port for WebSocket tunnelling (path-prefix → (host, port))
WS_UPSTREAMS = {
    "/detector/": (JETSON, 9090),
}

# Methods that carry a request body and must be forwarded to upstream.
_BODY_METHODS = {"POST", "PUT", "PATCH"}


def _splice(a: socket.socket, b: socket.socket):
    """Bidirectional byte-pipe between two sockets. Blocks until either closes."""
    socks = [a, b]
    try:
        while True:
            readable, _, exceptional = select.select(socks, [], socks, 30.0)
            if exceptional:
                break
            if not readable:
                # 30-second idle timeout — keeps the thread from leaking.
                break
            for src in readable:
                dst = b if src is a else a
                try:
                    data = src.recv(65536)
                except OSError:
                    data = b""
                if not data:
                    return
                try:
                    dst.sendall(data)
                except OSError:
                    return
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ------------------------------------------------------------------
    # WebSocket upgrade detection
    # ------------------------------------------------------------------

    def _is_ws_upgrade(self) -> bool:
        upgrade = self.headers.get("Upgrade", "").lower()
        conn   = self.headers.get("Connection", "").lower()
        return upgrade == "websocket" and "upgrade" in conn

    def _handle_ws_tunnel(self):
        """Forward a WebSocket upgrade request as a raw TCP tunnel.

        Steps:
          1. Reconstruct the opening HTTP request and send it to the upstream.
          2. Read the upstream's HTTP response (must be 101 Switching Protocols).
          3. Forward the 101 back to the browser.
          4. Splice both sockets bidirectionally until either side closes.
        """
        # Find the upstream host+port for this path.
        upstream_host, upstream_port = None, None
        upstream_path = self.path
        for prefix, (host, port) in WS_UPSTREAMS.items():
            if self.path.startswith(prefix):
                upstream_host = host
                upstream_port = port
                upstream_path = "/" + self.path[len(prefix):]
                break

        if upstream_host is None:
            self.send_error(502, "No WebSocket upstream configured for this path")
            return

        # Reconstruct the request line + headers to forward upstream.
        # We need to pass all WS handshake headers (Sec-WebSocket-Key, etc.).
        forward_headers = []
        skip = {"host", "connection", "upgrade"}  # we'll add our own
        for key, val in self.headers.items():
            if key.lower() not in skip:
                forward_headers.append(f"{key}: {val}")

        request_text = (
            f"GET {upstream_path} HTTP/1.1\r\n"
            f"Host: {upstream_host}:{upstream_port}\r\n"
            f"Connection: Upgrade\r\n"
            f"Upgrade: websocket\r\n"
        ) + "\r\n".join(forward_headers) + "\r\n\r\n"

        try:
            upstream_sock = socket.create_connection((upstream_host, upstream_port), timeout=10)
        except OSError as e:
            self.send_error(502, f"Cannot connect to upstream: {e}")
            return

        try:
            upstream_sock.sendall(request_text.encode())

            # Read upstream's HTTP response headers.
            response_buf = b""
            while b"\r\n\r\n" not in response_buf:
                chunk = upstream_sock.recv(4096)
                if not chunk:
                    self.send_error(502, "Upstream closed during WS handshake")
                    upstream_sock.close()
                    return
                response_buf += chunk
                if len(response_buf) > 8192:
                    self.send_error(502, "Upstream WS handshake response too large")
                    upstream_sock.close()
                    return

            header_end = response_buf.index(b"\r\n\r\n")
            header_bytes = response_buf[:header_end]
            leftover = response_buf[header_end + 4:]

            # Parse status line.
            header_lines = header_bytes.decode(errors="replace").split("\r\n")
            status_line = header_lines[0]
            parts = status_line.split(" ", 2)
            if len(parts) < 2 or parts[1] != "101":
                self.send_error(502, f"Upstream WS handshake failed: {status_line}")
                upstream_sock.close()
                return

            # Forward the 101 + all upstream headers to the browser.
            # We write to the raw connection socket to avoid buffering.
            reply = f"HTTP/1.1 101 Switching Protocols\r\n"
            for line in header_lines[1:]:
                if line:
                    reply += line + "\r\n"
            reply += "\r\n"
            self.wfile.write(reply.encode())
            if leftover:
                self.wfile.write(leftover)
            self.wfile.flush()

            # Grab the underlying client socket.
            client_sock = self.connection

            # If upstream sent WS frames in leftover they're already written.
            # Now splice both sockets until either closes.
            _splice(client_sock, upstream_sock)

        except OSError as e:
            try:
                upstream_sock.close()
            except OSError:
                pass
            # Connection may already be gone — log and return.
            self.log_message("WS tunnel error: %s", str(e))

    # ------------------------------------------------------------------

    def _dispatch(self, method):
        # WebSocket upgrade — handle before regular HTTP proxy.
        if method == "GET" and self._is_ws_upgrade():
            self._handle_ws_tunnel()
            return
        for prefix, base in UPSTREAMS.items():
            if self.path.startswith(prefix):
                self._proxy(base + self.path[len(prefix):], method)
                return
        if method == "GET":
            self._static()
        else:
            self.send_error(405, "Method not allowed")

    def do_GET(self):     self._dispatch("GET")
    def do_POST(self):    self._dispatch("POST")
    def do_PATCH(self):   self._dispatch("PATCH")
    def do_DELETE(self):  self._dispatch("DELETE")
    def do_OPTIONS(self): self._dispatch("OPTIONS")

    def _static(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "":
            path = "/monitor.html"
        fs = os.path.normpath(os.path.join(HERE, path.lstrip("/")))
        if not fs.startswith(HERE) or not os.path.isfile(fs):
            self.send_error(404, "Not found")
            return
        body = open(fs, "rb").read()
        ctype = "text/html; charset=utf-8" if fs.endswith(".html") else "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self, url, method="GET"):
        # Read request body for methods that carry one.
        body = None
        if method in _BODY_METHODS:
            cl = self.headers.get("Content-Length")
            if cl:
                body = self.rfile.read(int(cl))

        try:
            req = urllib.request.Request(url, data=body, method=method)
            # Forward Content-Type so WHEP SDP POSTs are accepted upstream.
            ct = self.headers.get("Content-Type")
            if ct:
                req.add_header("Content-Type", ct)
            # Forward If-Match (used by WHEP PATCH trickle-ICE).
            im = self.headers.get("If-Match")
            if im:
                req.add_header("If-Match", im)
            # No Accept-Encoding — let upstream send raw bytes.
            resp = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as e:
            # Re-forward upstream HTTP errors (e.g. 400 Bad Request from WHEP).
            self.send_response(e.code)
            ctype = e.headers.get("Content-Type", "application/json")
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            body_err = e.read()
            self.send_header("Content-Length", str(len(body_err)))
            self.end_headers()
            self.wfile.write(body_err)
            return
        except urllib.error.URLError as e:
            self.send_error(502, f"Upstream error: {e}")
            return
        except socket.timeout:
            self.send_error(504, "Upstream timeout")
            return

        # Forward status + select headers. Use chunked transfer for streams.
        self.send_response(resp.status)
        ctype = resp.headers.get("Content-Type", "application/octet-stream")
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, If-Match")
        self.send_header("Access-Control-Expose-Headers", "ETag, ID, Accept-Patch, Link, Location")
        # Forward WHEP-specific response headers.
        for hdr in ("ETag", "ID", "Location", "Accept-Patch", "Link"):
            val = resp.headers.get(hdr)
            if val:
                self.send_header(hdr, val)
        is_stream = "multipart/x-mixed-replace" in ctype
        if is_stream:
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Connection", "close")
            self.protocol_version = "HTTP/1.0"  # no chunking, raw stream
        else:
            cl = resp.headers.get("Content-Length")
            if cl:
                self.send_header("Content-Length", cl)
        self.end_headers()

        try:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                resp.close()
            except Exception:
                pass


class ThreadedHTTP(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    srv = ThreadedHTTP(("0.0.0.0", PORT), Handler)
    print(
        f"monitor proxy: http://0.0.0.0:{PORT}/  →  "
        f"detector={JETSON}:9090  gpu={JETSON}:9091  webrtc={JETSON}:8889",
        flush=True,
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
