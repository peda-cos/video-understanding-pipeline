"""Content-aware shot boundary detection over downscaled HSV frames."""

from __future__ import annotations

import numpy as np

from . import media


def _hsv_delta(prev: np.ndarray, curr: np.ndarray) -> float:
    dh = np.abs(curr[..., 0] - prev[..., 0])
    dh = np.minimum(dh, 360.0 - dh)
    ds = np.abs(curr[..., 1] - prev[..., 1]) * 255.0
    dv = np.abs(curr[..., 2] - prev[..., 2]) * 255.0
    return float(np.mean((dh + ds + dv) / 3.0))


def detect_scenes(
    path: str,
    threshold: float = 27.0,
    min_scene_ms: int = 600,
    fps: float = 4.0,
) -> list[tuple[int, int]]:
    """Return [(start_ms, end_ms)] for hard cuts, quantized to 1000/fps ms."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    total_ms = media.duration_ms(path)
    if total_ms <= 0:
        raise media.MediaError(f"video has no measurable duration: {path}")

    cuts: list[int] = []
    prev: np.ndarray | None = None
    scene_start = 0
    frame_ms = 1000.0 / fps
    for index, frame in enumerate(media.iter_frames(path, fps=fps, width=160, height=90)):
        hsv = _rgb_hsv(frame)
        if prev is not None:
            score = _hsv_delta(prev, hsv)
            at_ms = int(round(index * frame_ms))
            if score > threshold and at_ms - scene_start >= min_scene_ms:
                cuts.append(at_ms)
                scene_start = at_ms
        prev = hsv

    if not cuts:
        return [(0, total_ms)]
    boundaries = [0, *cuts, total_ms]
    scenes: list[tuple[int, int]] = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        if end - start < min_scene_ms and scenes:
            prev_start, _ = scenes[-1]
            scenes[-1] = (prev_start, end)
        elif end > start:
            scenes.append((start, end))
    if scenes and scenes[-1][1] < total_ms:
        start, _ = scenes[-1]
        scenes[-1] = (start, total_ms)
    return scenes


def _rgb_hsv(frame: np.ndarray) -> np.ndarray:
    # Local import-free HSV conversion to keep the detector independent of vision.
    arr = frame.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    delta = maxc - minc
    hue = np.zeros_like(maxc)
    mask = delta > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        red = mask & (maxc == r)
        green = mask & (maxc == g) & ~red
        blue = mask & ~red & ~green
        hue[red] = (60.0 * ((g[red] - b[red]) / delta[red])) % 360.0
        hue[green] = 60.0 * ((b[green] - r[green]) / delta[green] + 2.0)
        hue[blue] = 60.0 * ((r[blue] - g[blue]) / delta[blue] + 4.0)
    sat = np.where(maxc > 1e-6, delta / np.maximum(maxc, 1e-6), 0.0)
    return np.stack([hue, sat, maxc], axis=-1)
