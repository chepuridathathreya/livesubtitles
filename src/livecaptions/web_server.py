import json
import queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .engine import LANGUAGES, CaptionEngine

PAGE = Path(__file__).resolve().parent / "static" / "index.html"


def make_server(engine: CaptionEngine, host: str, port: int, key: str) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the terminal for captions and status
            pass

        def _authorised(self) -> bool:
            query = parse_qs(urlparse(self.path).query)
            return query.get("key", [""])[0] == key

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if not self._authorised():
                self._send(403, b"forbidden", "text/plain")
            elif path == "/":
                page = PAGE.read_text(encoding="utf-8").replace(
                    "__LANGUAGES__", json.dumps(LANGUAGES, ensure_ascii=False)
                )
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/events":
                self._stream_events()
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if not self._authorised():
                self._send(403, b"forbidden", "text/plain")
                return
            path = urlparse(self.path).path
            body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            try:
                data = json.loads(body or b"{}")
            except json.JSONDecodeError:
                data = {}
            if path == "/language":
                engine.set_language(str(data.get("language", "")))
            elif path == "/paused":
                engine.set_paused(bool(data.get("paused")))
            else:
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, b"{}", "application/json")

        def _stream_events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            listener = engine.subscribe()
            try:
                while True:
                    try:
                        event = listener.get(timeout=15)
                        payload = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    except queue.Empty:
                        payload = ": keep-alive\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                engine.unsubscribe(listener)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
