"""Deterministic classical-vision analysis for keyframes.

This is the offline stand-in for a VLM: it measures real pixels (colours,
connected blobs, edges, motion between sampled frames) and emits structured
facts plus a template caption. It is deliberately small and inspectable.
"""

from __future__ import annotations

import math
import re
from collections import deque
from typing import Any

import numpy as np

COLOR_NAMES = [
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "magenta",
    "white",
    "gray",
]
SHAPES = ["rectangle", "circle"]
DIRECTIONS = ["left-to-right", "right-to-left", "up", "down", "stationary"]

_HUE_RANGES: dict[str, tuple[float, float]] = {
    "red": (0.0, 15.0),
    "orange": (15.0, 40.0),
    "yellow": (40.0, 70.0),
    "green": (70.0, 160.0),
    "cyan": (160.0, 200.0),
    "blue": (200.0, 250.0),
    "purple": (250.0, 290.0),
    "magenta": (290.0, 330.0),
}


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8/float RGB (H,W,3) to HSV with H in degrees, S/V in [0,1]."""
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    delta = maxc - minc
    hue = np.zeros_like(maxc)
    nonzero = delta > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        red_mask = nonzero & (maxc == r)
        green_mask = nonzero & (maxc == g) & ~red_mask
        blue_mask = nonzero & ~red_mask & ~green_mask
        hue[red_mask] = (60.0 * ((g[red_mask] - b[red_mask]) / delta[red_mask])) % 360.0
        hue[green_mask] = 60.0 * ((b[green_mask] - r[green_mask]) / delta[green_mask] + 2.0)
        hue[blue_mask] = 60.0 * ((r[blue_mask] - g[blue_mask]) / delta[blue_mask] + 4.0)
    sat = np.where(maxc > 1e-6, delta / np.maximum(maxc, 1e-6), 0.0)
    return np.stack([hue, sat, maxc], axis=-1).astype(np.float32)


def classify_pixels(hsv: np.ndarray) -> dict[str, np.ndarray]:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    chromatic = (s >= 0.45) & (v >= 0.25)
    masks: dict[str, np.ndarray] = {}
    for name, (lo, hi) in _HUE_RANGES.items():
        if name == "red":
            masks[name] = chromatic & ((h < hi) | (h >= 330.0))
        else:
            masks[name] = chromatic & (h >= lo) & (h < hi)
    masks["white"] = (s < 0.18) & (v >= 0.75)
    masks["gray"] = (s < 0.20) & (v >= 0.30) & (v < 0.75)
    return masks


def connected_components(mask: np.ndarray) -> list[dict[str, Any]]:
    """4-neighbour components with bounding boxes."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, Any]] = []
    ys, xs = np.nonzero(mask)
    for seed_y, seed_x in zip(ys.tolist(), xs.tolist(), strict=False):
        if visited[seed_y, seed_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(seed_y, seed_x)])
        visited[seed_y, seed_x] = True
        area = 0
        min_x = max_x = seed_x
        min_y = max_y = seed_y
        sum_x = 0
        sum_y = 0
        while queue:
            y, x = queue.popleft()
            area += 1
            sum_x += x
            sum_y += y
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        components.append(
            {
                "area": area,
                "x0": min_x,
                "y0": min_y,
                "x1": max_x,
                "y1": max_y,
                "cx": sum_x / area / max(1, width - 1),
                "cy": sum_y / area / max(1, height - 1),
            }
        )
    return components


def detect_objects(frame: np.ndarray) -> list[dict[str, Any]]:
    """Find coloured blobs, dropping the dominant background component."""
    hsv = rgb_to_hsv(frame)
    masks = classify_pixels(hsv)
    total = float(frame.shape[0] * frame.shape[1])
    objects: list[dict[str, Any]] = []
    for color in COLOR_NAMES:
        for comp in connected_components(masks[color]):
            area_frac = comp["area"] / total
            if area_frac < 0.003 or area_frac > 0.25:
                continue
            bw = comp["x1"] - comp["x0"] + 1
            bh = comp["y1"] - comp["y0"] + 1
            fill = comp["area"] / float(bw * bh)
            if fill < 0.45:
                continue
            aspect = max(bw, bh) / max(1, min(bw, bh))
            objects.append(
                {
                    "color": color,
                    "shape": "rectangle" if aspect >= 1.30 else "circle",
                    "area_frac": round(area_frac, 6),
                    "cx": round(comp["cx"], 6),
                    "cy": round(comp["cy"], 6),
                    "fill_ratio": round(fill, 6),
                }
            )
    objects.sort(key=lambda o: (COLOR_NAMES.index(o["color"]), o["cx"], o["cy"]))
    return objects


def edge_density(frame: np.ndarray) -> float:
    arr = frame.astype(np.float32) / 255.0
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    return float(((gx > 0.08).mean() + (gy > 0.08).mean()) / 2.0)


def brightness(frame: np.ndarray) -> float:
    hsv = rgb_to_hsv(frame)
    return float(hsv[..., 2].mean())


def brightness_label(value: float) -> str:
    if value < 0.35:
        return "dark"
    if value < 0.65:
        return "medium"
    return "bright"


def dominant_colors(frame: np.ndarray, min_frac: float = 0.15) -> list[str]:
    hsv = rgb_to_hsv(frame)
    masks = classify_pixels(hsv)
    total = float(frame.shape[0] * frame.shape[1])
    fractions = [(name, float(mask.mean())) for name, mask in masks.items()]
    fractions = [(name, frac) for name, frac in fractions if frac >= min_frac]
    fractions.sort(key=lambda item: (-item[1], COLOR_NAMES.index(item[0])))
    return [name for name, _ in fractions[:2]]


def _direction(dx: float, dy: float, threshold: float = 0.08) -> str:
    if abs(dx) < threshold and abs(dy) < threshold:
        return "stationary"
    if abs(dx) >= abs(dy):
        return "left-to-right" if dx > 0 else "right-to-left"
    return "down" if dy > 0 else "up"


def analyze_scene(frames: list[np.ndarray], times_ms: list[int]) -> dict[str, Any]:
    """Produce facts and a caption from sampled scene frames."""
    if not frames:
        raise ValueError("analyze_scene needs at least one frame")
    if len(frames) != len(times_ms):
        raise ValueError("frames and times_ms must have the same length")
    middle = len(frames) // 2
    key = frames[middle]
    objects_by_frame = [detect_objects(frame) for frame in frames]
    key_objects = objects_by_frame[middle]

    object_counts: dict[str, int] = {}
    for obj in key_objects:
        label = f"{obj['color']} {obj['shape']}"
        object_counts[label] = object_counts.get(label, 0) + 1

    colors = dominant_colors(key)
    if not colors:
        colors = ["gray"]
    brightness_value = brightness(key)
    bright_label = brightness_label(brightness_value)

    tracks: list[dict[str, Any]] = []
    for color in COLOR_NAMES:
        per_frame = [[o for o in frame if o["color"] == color] for frame in objects_by_frame]
        if not all(len(items) == 1 for items in per_frame):
            continue
        first, last = per_frame[0][0], per_frame[-1][0]
        dx = last["cx"] - first["cx"]
        dy = last["cy"] - first["cy"]
        direction = _direction(dx, dy)
        tracks.append(
            {
                "color": color,
                "shape": per_frame[middle][0]["shape"],
                "direction": direction,
                "displacement": round(math.hypot(dx, dy), 6),
                "start_t_ms": int(times_ms[0]),
                "end_t_ms": int(times_ms[-1]),
            }
        )

    max_displacement = max((t["displacement"] for t in tracks), default=0.0)
    if max_displacement < 0.02:
        motion = "static"
    elif max_displacement < 0.15:
        motion = "low"
    else:
        motion = "high"

    count_phrases = [
        f"{count} {label}{'' if count == 1 else 's'}"
        for label, count in sorted(object_counts.items())
    ]
    caption_parts = [f"A {bright_label} scene with {', '.join(colors)} background."]
    if count_phrases:
        caption_parts.append("Visible: " + ", ".join(count_phrases) + ".")
    moving = [t for t in tracks if t["direction"] != "stationary"]
    if moving:
        best = max(moving, key=lambda t: t["displacement"])
        caption_parts.append(
            f"The {best['color']} {best['shape']} moves {best['direction'].replace('-', ' ')}."
        )
    elif tracks:
        best = tracks[0]
        caption_parts.append(f"The {best['color']} {best['shape']} stays still.")

    return {
        "background_colors": colors,
        "brightness": bright_label,
        "object_counts": object_counts,
        "objects": key_objects,
        "tracks": tracks,
        "motion": motion,
        "caption": " ".join(caption_parts),
    }


_COLOR_SYNONYMS = {"grey": "gray", "violet": "purple"}


def parse_visual_terms(text: str) -> dict[str, Any]:
    lowered = text.lower()
    words = re.findall(r"[a-z]+", lowered)
    colors: list[str] = []
    for word in words:
        canonical = _COLOR_SYNONYMS.get(word, word)
        if canonical in COLOR_NAMES and canonical not in colors:
            colors.append(canonical)
    shapes: list[str] = []
    for word in words:
        singular = "rectangle" if word.startswith("rectang") else "circle" if word.startswith("circl") else ""
        if singular and singular not in shapes:
            shapes.append(singular)
    direction = ""
    normalized = lowered.replace("-", " ")
    if re.search(r"\bleft\s+to\s+right\b", normalized):
        direction = "left-to-right"
    elif re.search(r"\bright\s+to\s+left\b", normalized):
        direction = "right-to-left"
    elif re.search(r"\bmov\w*\s+up\b|\bupwards?\b", normalized):
        direction = "up"
    elif re.search(r"\bmov\w*\s+down\b|\bdownwards?\b", normalized):
        direction = "down"
    elif re.search(r"\bstationary\b|\bstays?\s+still\b|\bnot\s+move\b", normalized):
        direction = "stationary"
    count_intent = bool(re.search(r"\bhow\s+many\b|\bnumber\s+of\b", normalized))
    return {
        "colors": colors,
        "shapes": shapes,
        "direction": direction,
        "count_intent": count_intent,
    }


def visual_descriptor(frames: list[np.ndarray], facts: dict[str, Any]) -> np.ndarray:
    """80-d handcrafted scene descriptor shared with the query encoder."""
    if not frames:
        raise ValueError("visual_descriptor needs at least one frame")
    middle = frames[len(frames) // 2]
    hsv = rgb_to_hsv(middle)
    masks = classify_pixels(hsv)
    color_fractions = np.array(
        [float(masks[name].mean()) for name in COLOR_NAMES], dtype=np.float32
    )
    counts = np.zeros(len(COLOR_NAMES), dtype=np.float32)
    for label, count in facts.get("object_counts", {}).items():
        parts = label.split()
        if parts and parts[0] in COLOR_NAMES:
            counts[COLOR_NAMES.index(parts[0])] += float(count)
    counts = np.clip(counts, 0.0, 2.0) / 2.0

    total_objects = sum(facts.get("object_counts", {}).values())
    shape_vec = np.zeros(3, dtype=np.float32)
    if total_objects:
        for label, count in facts.get("object_counts", {}).items():
            shape = label.split()[-1]
            idx = 0 if shape == "rectangle" else 1 if shape == "circle" else 2
            shape_vec[idx] += float(count) / total_objects
    stats = np.array(
        [
            float(hsv[..., 2].mean()),
            float(hsv[..., 1].mean()),
            edge_density(middle),
        ],
        dtype=np.float32,
    )

    tracks = facts.get("tracks", [])
    best = max(tracks, key=lambda t: t.get("displacement", 0.0), default=None)
    motion_vec = np.zeros(6, dtype=np.float32)
    if best is not None:
        motion_vec[0] = min(1.0, float(best.get("displacement", 0.0)) / 0.5)
        direction = best.get("direction", "stationary")
        motion_vec[1 + DIRECTIONS.index(direction)] = 1.0

    grid = np.zeros(48, dtype=np.float32)
    arr = middle.astype(np.float32) / 255.0
    gh, gw = arr.shape[0] // 4, arr.shape[1] // 4
    for gy in range(4):
        for gx in range(4):
            cell = arr[gy * gh : (gy + 1) * gh, gx * gw : (gx + 1) * gw]
            offset = (gy * 4 + gx) * 3
            grid[offset : offset + 3] = cell.reshape(-1, 3).mean(axis=0)

    vec = np.concatenate(
        [color_fractions, counts, shape_vec, stats, motion_vec, grid]
    ).astype(np.float32)
    if vec.size != 80:
        raise AssertionError(f"visual descriptor must be 80-d, got {vec.size}")
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def visual_query(text: str) -> np.ndarray:
    """Encode textual colour/shape/motion terms into the visual descriptor space."""
    terms = parse_visual_terms(text)
    vec = np.zeros(80, dtype=np.float32)
    for color in terms["colors"]:
        idx = COLOR_NAMES.index(color)
        vec[idx] = 0.8
        vec[len(COLOR_NAMES) + idx] = 0.6
    for shape in terms["shapes"]:
        idx = 20 + (0 if shape == "rectangle" else 1)
        vec[idx] = 1.0
    center = 26
    if terms["direction"]:
        vec[center] = 1.0
        vec[center + 1 + DIRECTIONS.index(terms["direction"])] = 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec
