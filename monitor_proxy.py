#!/usr/bin/env python3
"""Reverse proxy + static server for monitor.html.

- GET /               → serves monitor.html
- GET /detector/<p>   → proxies to http://JETSON:9090/<p>
- GET /gpu/<p>        → proxies to http://JETSON:9091/<p>

Streaming MJPEG (/detector/stream) is forwarded chunked, never buffered.
"""
import http.server
import socketserver
import socket
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
}

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        # Pick upstream prefix
        for prefix, base in UPSTREAMS.items():
            if self.path.startswith(prefix):
                self._proxy(base + self.path[len(prefix):])
                return
        # Static file serving
        self._static()

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

    def _proxy(self, url):
        try:
            req = urllib.request.Request(url, method="GET")
            # No Accept-Encoding — let upstream send raw bytes.
            resp = urllib.request.urlopen(req, timeout=5)
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
    print(f"monitor proxy: http://0.0.0.0:{PORT}/  →  detector={JETSON}:9090  gpu={JETSON}:9091", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
