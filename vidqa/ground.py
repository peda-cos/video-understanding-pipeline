"""Temporal grounding: refine a query window inside a scene."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import VidqaError
from .caption import default_transport
from .embed import TextEmbedder, content_tokens, cosine
from .models import Scene
from .vision import visual_query


@dataclass
class Grounding:
    start_ms: int
    end_ms: int
    score: float
    method: str = "local"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "score": round(self.score, 6),
            "method": self.method,
        }


def iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return float(inter / union) if union > 0 else 0.0


def _sample(items: list[int], limit: int) -> list[int]:
    if len(items) <= limit:
        return items
    indices = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]
    return sorted({items[i] for i in indices})


class LocalGrounder:
    def __init__(self, embedder: TextEmbedder | None = None) -> None:
        self.embedder = embedder or TextEmbedder()

    def ground(
        self,
        question: str,
        scene: Scene,
        thumb_vectors: np.ndarray,
    ) -> Grounding:
        scene_start, scene_end = scene.start_ms, scene.end_ms
        scene_len = max(1, scene_end - scene_start)
        if scene_len <= 600 or not scene.words:
            return Grounding(scene_start, scene_end, 0.0)
        query_vec = self.embedder.encode(question)
        q_visual = visual_query(question)
        query_tokens = set(content_tokens(question))

        starts = {scene_start}
        ends = {scene_end}
        for word in scene.words:
            if scene_start <= word.start_ms < scene_end:
                starts.add(word.start_ms)
            if scene_start < word.end_ms <= scene_end:
                ends.add(word.end_ms)
        for thumb in scene.thumbs:
            starts.add(max(scene_start, thumb.t_ms - 300))
            ends.add(min(scene_end, thumb.t_ms + 300))
        starts_sorted = _sample(sorted(starts), 12)
        ends_sorted = _sample(sorted(ends), 12)

        best: tuple[tuple[float, int, int], int, int] | None = None
        for start in starts_sorted:
            for end in ends_sorted:
                length = end - start
                if length < 600 or length > min(scene_len, 60_000):
                    continue
                window_words = [
                    w for w in scene.words if start <= (w.start_ms + w.end_ms) // 2 < end
                ]
                if window_words:
                    text = " ".join(w.text for w in window_words)
                    text_score = cosine(query_vec, self.embedder.encode(text))
                    window_tokens = set(content_tokens(text))
                    lexical = (
                        len(query_tokens & window_tokens) / len(query_tokens)
                        if query_tokens
                        else 0.0
                    )
                else:
                    text_score = 0.0
                    lexical = 0.0
                visual = 0.0
                indices = [
                    i for i, thumb in enumerate(scene.thumbs) if start <= thumb.t_ms <= end
                ]
                if not indices and len(scene.thumbs):
                    nearest = min(
                        range(len(scene.thumbs)),
                        key=lambda i: abs(scene.thumbs[i].t_ms - (start + length // 2)),
                    )
                    indices = [nearest]
                for i in indices:
                    if i < thumb_vectors.shape[0]:
                        visual = max(visual, cosine(q_visual, thumb_vectors[i]))
                prior = math.exp(-0.5 * ((length / scene_len - 0.4) / 0.3) ** 2)
                score = 0.45 * text_score + 0.25 * lexical + 0.20 * visual + 0.10 * prior
                key = (score, -length, -start)
                if best is None or key > best[0]:
                    best = (key, start, end)
        if best is None:
            return Grounding(scene_start, scene_end, 0.0)
        return Grounding(best[1], best[2], best[0][0])


class HTTPGrounder:
    """TimeLens/VideoITG-style adapter with a local fallback."""

    def __init__(
        self,
        url: str,
        transport: Callable[..., tuple[int, str]] | None = None,
        fallback: LocalGrounder | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not url:
            raise VidqaError("GROUNDER_URL is empty")
        self.url = url
        self.transport = transport or default_transport
        self.fallback = fallback or LocalGrounder()
        self.timeout = timeout

    def ground(
        self,
        question: str,
        scene: Scene,
        thumb_vectors: np.ndarray,
    ) -> Grounding:
        body = {
            "query": question,
            "video_id": scene.video_id,
            "scene_id": scene.scene_id,
            "start_ms": scene.start_ms,
            "end_ms": scene.end_ms,
            "words": [w.to_dict() for w in scene.words],
        }
        try:
            status, text = self.transport(
                "POST",
                self.url,
                {"Content-Type": "application/json"},
                body,
                self.timeout,
            )
            if status >= 400:
                raise VidqaError(f"grounder HTTP {status}")
            parsed = json.loads(text)
            start = int(parsed["start_ms"])
            end = int(parsed["end_ms"])
            if not (scene.start_ms <= start < end <= scene.end_ms):
                raise VidqaError("grounder returned a window outside the scene")
            return Grounding(start, end, float(parsed.get("score", 0.0)), method="http")
        except Exception:
            return self.fallback.ground(question, scene, thumb_vectors)
