"""Dependency-free HTTP server for the citation UI and JSON API."""

from __future__ import annotations

import json
import mimetypes
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from . import VidqaError
from .pipeline import AskService, NoResultsError

ROOT = Path(__file__).resolve().parents[1]
WEB_PUBLIC = ROOT / "web" / "public"
APP_JS = ROOT / ".local" / "web" / "app.js"


class VidqaServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: AskService) -> None:
        super().__init__(address, VidqaHandler)
        self.service = service


class VidqaHandler(BaseHTTPRequestHandler):
    server: VidqaServer

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[server] {self.address_string()} {format % args}", file=sys.stderr)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not getattr(self, "_head_only", False):
            self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _file(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "file not found")
            return
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        status = HTTPStatus.OK
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
            if not match:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            raw_start, raw_end = match.groups()
            if raw_start:
                start = int(raw_start)
            if raw_end:
                end = min(size - 1, int(raw_end))
            if start >= size or start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        content_type = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if getattr(self, "_head_only", False):
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    def _static(self, relative: str, fallback: Path | None = None) -> None:
        path = fallback or WEB_PUBLIC / relative
        if not path.exists():
            self._error(HTTPStatus.NOT_FOUND, f"missing asset: {relative}")
            return
        self._file(path)

    def do_HEAD(self) -> None:  # noqa: N802
        self._head_only = True
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        self._head_only = getattr(self, "_head_only", False)
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        try:
            if path == "/api/health":
                self._json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "index": self.server.service.index.stats(),
                        "synthesizer": getattr(self.server.service.synthesizer, "provider", "local"),
                        "grounder": getattr(self.server.service.grounder, "method", "local"),
                    },
                )
                return
            if path == "/api/videos":
                self._json(HTTPStatus.OK, {"videos": self._videos()})
                return
            scene_match = re.fullmatch(r"/api/videos/([^/]+)/scenes", path)
            if scene_match:
                video_id = scene_match.group(1)
                payloads = self.server.service.index.payloads_for_video(video_id)
                if not payloads:
                    self._error(HTTPStatus.NOT_FOUND, f"unknown video: {video_id}")
                    return
                self._json(HTTPStatus.OK, {"video_id": video_id, "scenes": payloads})
                return
            if path == "/api/ask":
                question = query.get("q", [""])[0].strip()
                if not question:
                    self._error(HTTPStatus.BAD_REQUEST, "query parameter q is required")
                    return
                if len(question) > 500:
                    self._error(HTTPStatus.BAD_REQUEST, "question is longer than 500 characters")
                    return
                try:
                    k = int(query.get("k", ["5"])[0])
                except ValueError:
                    k = 5
                k = max(1, min(10, k))
                video_id = query.get("video_id", [None])[0]
                try:
                    answer = self.server.service.ask(question, k=k, video_id=video_id)
                except NoResultsError as exc:
                    self._error(HTTPStatus.NOT_FOUND, str(exc))
                    return
                self._json(HTTPStatus.OK, answer.to_dict())
                return
            if path.startswith("/media/"):
                self._serve_media(path[len("/media/") :])
                return
            if path in {"", "/", "/index.html"}:
                self._static("index.html")
                return
            if path == "/app.js":
                if not APP_JS.exists():
                    self._error(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "UI bundle missing; run ./verify.sh to build .local/web/app.js",
                    )
                    return
                self._file(APP_JS, "text/javascript; charset=utf-8")
                return
            if path == "/styles.css":
                self._static("styles.css")
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except VidqaError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self.log_error("unhandled: %s", exc)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal server error")

    def _videos(self) -> list[dict[str, Any]]:
        records = self.server.service.index.videos()
        for record in records:
            record["video_url"] = "/media/" + str(record.get("file_path", "")).replace("\\", "/")
            record["scene_count"] = sum(
                1
                for payload in self.server.service.index.payloads
                if payload["video_id"] == record["video_id"]
            )
        return records

    def _serve_media(self, relative: str) -> None:
        workdir = self.server.service.workdir.resolve()
        candidate = (workdir / relative).resolve()
        try:
            candidate.relative_to(workdir)
        except ValueError:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        self._file(candidate)


def serve(service: AskService, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = VidqaServer((host, port), service)
    print(f"[serve] http://{host}:{port} ({service.index.size} scenes)", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
