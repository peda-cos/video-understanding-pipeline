"""Thin, streaming ffmpeg/ffprobe wrappers.

No OpenCV/PyAV dependency: frames are decoded to raw RGB through a pipe so
long videos do not accumulate in memory.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from . import VidqaError


class MediaError(VidqaError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise MediaError(f"{tool} is required but was not found on PATH")
    return path


def _run(cmd: list[str], *, input_bytes: bytes | None = None) -> bytes:
    proc = subprocess.run(cmd, input=input_bytes, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()[-800:]
        raise MediaError(f"{cmd[0]} failed ({proc.returncode}): {detail}")
    return proc.stdout


def probe(path: str | Path) -> dict[str, Any]:
    """Return normalized metadata for a media file."""
    out = _run(
        [
            _require("ffprobe"),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    try:
        raw = json.loads(out)
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe returned invalid JSON for {path}") from exc
    streams = raw.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"no video stream in {path}")
    duration = raw.get("format", {}).get("duration")
    if duration in (None, "N/A"):
        duration = video.get("duration")
    try:
        duration_ms = int(round(float(duration) * 1000))
    except (TypeError, ValueError):
        duration_ms = 0
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    if not fps:
        fps = 25.0
    return {
        "duration_ms": duration_ms,
        "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)),
        "fps": fps,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "codec": video.get("codec_name", ""),
    }


def duration_ms(path: str | Path) -> int:
    return int(probe(path)["duration_ms"])


def has_audio(path: str | Path) -> bool:
    return bool(probe(path)["has_audio"])


def to_720p(src: str | Path, dst: str | Path) -> None:
    """Copy or transcode a source so it is at most 720p high."""
    info = probe(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if info["height"] <= 720 and Path(src).suffix.lower() == ".mp4":
        shutil.copy2(src, dst)
        return
    cmd = [
        _require("ffmpeg"),
        "-v",
        "error",
        "-y",
        "-i",
        str(src),
        "-vf",
        "scale=-2:720",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
    ]
    if info["has_audio"]:
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(dst)]
    _run(cmd)


def _read_exact(stream: Any, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_frames(
    path: str | Path,
    fps: float = 4.0,
    width: int = 160,
    height: int = 90,
) -> Iterator[np.ndarray]:
    """Yield RGB frames sampled at `fps`, resized to exactly width x height."""
    cmd = [
        _require("ffmpeg"),
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"fps={fps},scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = width * height * 3
    try:
        assert proc.stdout is not None
        while True:
            raw = _read_exact(proc.stdout, frame_size)
            if raw is None:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
        proc.wait(timeout=30)
        if proc.returncode != 0:
            detail = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", "replace")
            raise MediaError(f"ffmpeg frame decode failed: {detail.strip()[-500:]}")
    finally:
        if proc.poll() is None:
            proc.kill()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()


def extract_jpeg(src: str | Path, at_ms: int, dst: str | Path) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _require("ffmpeg"),
            "-v",
            "error",
            "-y",
            "-ss",
            f"{max(0, at_ms) / 1000:.3f}",
            "-i",
            str(src),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(dst),
        ]
    )
    if not dst.exists() or dst.stat().st_size == 0:
        raise MediaError(f"failed to extract a frame at {at_ms}ms from {src}")


def decode_image(path: str | Path, width: int = 192) -> np.ndarray:
    """Decode an image to RGB, resized to `width` with an even height."""
    info = probe(path)
    if not info["width"] or not info["height"]:
        raise MediaError(f"cannot determine image dimensions for {path}")
    height = max(2, int(round(info["height"] * width / info["width"] / 2)) * 2)
    cmd = [
        _require("ffmpeg"),
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    raw = _run(cmd)
    expected = width * height * 3
    if len(raw) < expected:
        raise MediaError(f"short image decode for {path}: {len(raw)} < {expected}")
    return np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, 3).copy()
