"""Deterministic synthetic video corpus with independently labelled queries."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from . import VidqaError

BACKGROUNDS: dict[str, tuple[int, int, int]] = {
    "blue": (30, 60, 120),
    "green": (40, 100, 60),
    "red": (120, 40, 50),
    "yellow": (130, 115, 35),
    "cyan": (30, 110, 120),
    "purple": (85, 45, 115),
}
OBJECT_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (230, 60, 60),
    "green": (70, 200, 90),
    "blue": (80, 130, 235),
    "yellow": (235, 210, 70),
    "cyan": (70, 210, 220),
    "magenta": (220, 80, 200),
}
KEYWORDS = [
    "orchard",
    "lantern",
    "harbor",
    "meadow",
    "kestrel",
    "cobalt",
    "velvet",
    "signal",
    "garden",
    "compass",
    "ember",
    "willow",
    "canyon",
    "tundra",
    "beacon",
    "quartz",
    "marble",
    "ripple",
    "summit",
    "hazel",
    "falcon",
    "pebble",
    "saffron",
    "indigo",
    "delta",
    "coral",
    "brisk",
    "lumen",
    "cedar",
    "onyx",
    "tidal",
    "nimbus",
    "sorrel",
    "violet",
    "garnet",
    "pylon",
]
DIRECTIONS = ["left-to-right", "right-to-left", "stationary"]
SHAPES = ["rectangle", "circle"]


def _draw_rectangle(frame: np.ndarray, cx: float, cy: float, w: int, h: int, rgb: tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    x0 = max(0, int(round(cx - w / 2)))
    x1 = min(width, int(round(cx + w / 2)))
    y0 = max(0, int(round(cy - h / 2)))
    y1 = min(height, int(round(cy + h / 2)))
    frame[y0:y1, x0:x1] = rgb


def _draw_circle(frame: np.ndarray, cx: float, cy: float, radius: int, rgb: tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    frame[mask] = rgb


def _draw_object(
    frame: np.ndarray,
    shape: str,
    cx: float,
    cy: float,
    rgb: tuple[int, int, int],
) -> None:
    if shape == "rectangle":
        _draw_rectangle(frame, cx, cy, 44, 30, rgb)
    else:
        _draw_circle(frame, cx, cy, 18, rgb)


def _object_center(obj: dict[str, Any], progress: float) -> tuple[float, float]:
    if not obj.get("mover"):
        return float(obj["x_frac"]), float(obj["y_frac"])
    direction = obj["direction"]
    if direction == "stationary":
        return 0.5, 0.5
    start, end = (0.2, 0.8) if direction == "left-to-right" else (0.8, 0.2)
    return start + (end - start) * progress, 0.48


def _build_specs(
    videos: int,
    scenes_per_video: int,
    scene_ms: int,
    scene_width: int,
    scene_height: int,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total_scenes = videos * scenes_per_video
    rotated = KEYWORDS[seed % len(KEYWORDS) :] + KEYWORDS[: seed % len(KEYWORDS)]
    keywords = (rotated * (total_scenes // len(KEYWORDS) + 1))[:total_scenes]
    all_scenes: list[dict[str, Any]] = []
    video_specs: list[dict[str, Any]] = []
    for video_index in range(videos):
        video_id = f"v{video_index + 1:03d}"
        video_scenes: list[dict[str, Any]] = []
        for local_index in range(scenes_per_video):
            global_index = video_index * scenes_per_video + local_index
            keyword = keywords[global_index]
            background = list(BACKGROUNDS)[(global_index + seed) % len(BACKGROUNDS)]
            available_colors = [c for c in OBJECT_COLORS if c != background]
            mover_color = available_colors[(global_index + seed) % len(available_colors)]
            mover_shape = SHAPES[global_index % 2]
            direction = DIRECTIONS[global_index % len(DIRECTIONS)]
            count_color = next(
                c for c in available_colors if c != mover_color
            )
            pair_shapes = ["rectangle", "rectangle"] if global_index % 2 == 0 else ["rectangle", "circle"]
            objects = [
                {
                    "color": mover_color,
                    "shape": mover_shape,
                    "x_frac": 0.5,
                    "y_frac": 0.48,
                    "mover": True,
                    "direction": direction,
                },
                {
                    "color": count_color,
                    "shape": pair_shapes[0],
                    "x_frac": 0.28,
                    "y_frac": 0.25,
                    "mover": False,
                    "direction": "stationary",
                },
                {
                    "color": count_color,
                    "shape": pair_shapes[1],
                    "x_frac": 0.72,
                    "y_frac": 0.75,
                    "mover": False,
                    "direction": "stationary",
                },
            ]
            start_ms = local_index * scene_ms
            scene = {
                "video_id": video_id,
                "scene_id": local_index + 1,
                "start_ms": start_ms,
                "end_ms": start_ms + scene_ms,
                "background": background,
                "keyword": keyword,
                "objects": objects,
                "action_window": [start_ms + scene_ms // 4, start_ms + 3 * scene_ms // 4],
                "width": scene_width,
                "height": scene_height,
                "scene_ms": scene_ms,
            }
            video_scenes.append(scene)
            all_scenes.append(scene)
        video_specs.append({"video_id": video_id, "scenes": video_scenes})
    return video_specs, all_scenes


def _words_for_scene(scene: dict[str, Any]) -> list[dict[str, Any]]:
    mover = next(obj for obj in scene["objects"] if obj["mover"])
    direction_text = {
        "left-to-right": "moves left to right",
        "right-to-left": "moves right to left",
        "stationary": "stays still",
    }[mover["direction"]]
    sentence = (
        f"the {scene['keyword']} scene shows a {mover['color']} {mover['shape']} "
        f"that {direction_text} near the measured marker"
    )
    tokens = sentence.split()
    start = scene["start_ms"] + scene["scene_ms"] // 5
    end = scene["end_ms"] - scene["scene_ms"] // 5
    step = max(1, (end - start) // max(1, len(tokens)))
    words = []
    for token_index, token in enumerate(tokens):
        word_start = start + token_index * step
        words.append(
            {
                "word": token,
                "start_ms": word_start,
                "end_ms": min(end, word_start + max(120, step - 30)),
            }
        )
    return words


def _render_video(
    video_id: str,
    scenes: list[dict[str, Any]],
    destination: Path,
    fps: int,
) -> None:
    width = int(scenes[0]["width"])
    height = int(scenes[0]["height"])
    scene_ms = int(scenes[0]["scene_ms"])
    frames_per_scene = max(1, int(round(scene_ms * fps / 1000)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=16000:cl=mono",
        "-shortest",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "32k",
        str(destination),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for scene in scenes:
            background = np.array(BACKGROUNDS[scene["background"]], dtype=np.uint8)
            for frame_index in range(frames_per_scene):
                frame = np.empty((height, width, 3), dtype=np.uint8)
                frame[:] = background
                t = frame_index / frames_per_scene
                progress = min(1.0, max(0.0, (t - 0.25) / 0.5))
                for obj in scene["objects"]:
                    cx, cy = _object_center(obj, progress)
                    _draw_object(
                        frame,
                        obj["shape"],
                        cx * width,
                        cy * height,
                        OBJECT_COLORS[obj["color"]],
                    )
                proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr else b""
        returncode = proc.wait(timeout=120)
        if returncode != 0:
            raise VidqaError(
                f"ffmpeg failed to render {video_id}: {stderr.decode('utf-8', 'replace')[-600:]}"
            )
    finally:
        if proc.poll() is None:
            proc.kill()
        if proc.stderr is not None:
            proc.stderr.close()


def _build_benchmark(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query_id = 1

    def add(scene: dict[str, Any], question: str, answer: str, category: str, window: list[int]) -> None:
        nonlocal query_id
        rows.append(
            {
                "query_id": f"q{query_id:03d}",
                "question": question,
                "category": category,
                "video_id": scene["video_id"],
                "scene_id": scene["scene_id"],
                "answer": answer,
                "window_ms": window,
                "source": "synthetic",
            }
        )
        query_id += 1

    for scene in scenes:
        count_color = next(
            obj["color"] for obj in scene["objects"] if not obj["mover"]
        )
        count = sum(
            1
            for obj in scene["objects"]
            if obj["color"] == count_color and obj["shape"] == "rectangle"
        )
        add(
            scene,
            f"How many {count_color} rectangles are visible in the scene about {scene['keyword']}?",
            str(count),
            "counting",
            [scene["start_ms"], scene["end_ms"]],
        )

    for scene in scenes:
        mover = next(obj for obj in scene["objects"] if obj["mover"])
        answer = "yes" if mover["direction"] == "left-to-right" else "no"
        add(
            scene,
            f"Does the {mover['color']} {mover['shape']} move from left to right in the scene about {scene['keyword']}?",
            answer,
            "action",
            list(scene["action_window"]),
        )

    for scene in scenes[: min(10, len(scenes))]:
        present = next(obj for obj in scene["objects"] if obj["mover"])
        add(
            scene,
            f"Is there a {present['color']} {present['shape']} in the scene about {scene['keyword']}?",
            "yes",
            "descriptive",
            [scene["start_ms"], scene["end_ms"]],
        )

    for scene in scenes[10:20]:
        present_combos = {(o["color"], o["shape"]) for o in scene["objects"]}
        combo = next(
            ((c, s) for c in OBJECT_COLORS for s in SHAPES if (c, s) not in present_combos),
            ("magenta", "circle"),
        )
        add(
            scene,
            f"Is there a {combo[0]} {combo[1]} in the scene about {scene['keyword']}?",
            "no",
            "descriptive",
            [scene["start_ms"], scene["end_ms"]],
        )

    for scene in (scenes[10:30] if len(scenes) >= 30 else scenes):
        add(
            scene,
            f"What color dominates the background of the scene about {scene['keyword']}?",
            scene["background"],
            "descriptive",
            [scene["start_ms"], scene["end_ms"]],
        )
    return rows


def build_corpus(
    out_dir: str | Path,
    seed: int = 7,
    videos: int = 6,
    scenes_per_video: int = 5,
    scene_ms: int = 4000,
    width: int = 384,
    height: int = 216,
    fps: int = 8,
) -> dict[str, Any]:
    if videos < 1 or scenes_per_video < 1:
        raise VidqaError("corpus needs at least one video and one scene")
    out_dir = Path(out_dir)
    (out_dir / "media").mkdir(parents=True, exist_ok=True)
    (out_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    video_specs, scenes = _build_specs(
        videos, scenes_per_video, scene_ms, width, height, seed
    )
    for video in video_specs:
        video_id = video["video_id"]
        _render_video(video_id, video["scenes"], out_dir / "media" / f"{video_id}.mp4", fps)
        words: list[dict[str, Any]] = []
        for scene in video["scenes"]:
            words.extend(_words_for_scene(scene))
        (out_dir / "transcripts" / f"{video_id}.words.json").write_text(
            json.dumps({"language": "en", "words": words}, indent=2),
            encoding="utf-8",
        )
    benchmark = _build_benchmark(scenes)
    with (out_dir / "benchmark.jsonl").open("w", encoding="utf-8") as handle:
        for row in benchmark:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    specs = {
        "seed": seed,
        "fps": fps,
        "videos": video_specs,
        "benchmark_size": len(benchmark),
    }
    (out_dir / "specs.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "videos": len(video_specs),
        "scenes": len(scenes),
        "queries": len(benchmark),
        "duration_ms": len(scenes) * scene_ms,
    }
