"""Answer synthesis: deterministic extractive local mode and Gemini adapter."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .caption import GeminiClient, ProviderError, extract_json, image_part
from .ground import Grounding
from .models import Citation, RetrievalHit, Scene
from .vision import parse_visual_terms


@dataclass
class Synthesis:
    answer: str
    category: str
    evidence: dict[str, Any]
    citations: list[Citation] = field(default_factory=list)
    confidence: float | None = None
    provider: str = "local"


def classify_question(question: str) -> str:
    terms = parse_visual_terms(question)
    if terms["count_intent"]:
        return "counting"
    lowered = question.lower()
    if re.search(
        r"\b(does|did|before|after|while|move|moves|moving|enter|enters|leave|leaves|"
        r"pour|stir)\b",
        lowered,
    ):
        return "action"
    return "descriptive"


def citation_frames(scene: Scene, start_ms: int, end_ms: int, limit: int = 4) -> list[dict[str, Any]]:
    in_window = [t for t in scene.thumbs if start_ms <= t.t_ms <= end_ms]
    if not in_window:
        in_window = sorted(
            scene.thumbs, key=lambda t: abs(t.t_ms - (start_ms + end_ms) // 2)
        )[:1]
    return [
        {"t_ms": t.t_ms, "path": t.path, "url": t.url}
        for t in in_window[:limit]
    ]


def citation_for_scene(scene: Scene, start_ms: int, end_ms: int) -> Citation:
    start = max(scene.start_ms, min(start_ms, scene.end_ms - 1))
    end = min(scene.end_ms, max(end_ms, start + 1))
    return Citation(
        video_id=scene.video_id,
        scene_id=scene.scene_id,
        start_ms=start,
        end_ms=end,
        frames=citation_frames(scene, start, end),
    )


def validate_citations(
    raw: list[dict[str, Any]],
    hits: list[RetrievalHit],
    fallback: bool = True,
) -> list[Citation]:
    allowed = {(h.scene.video_id, h.scene.scene_id): h.scene for h in hits}
    citations: list[Citation] = []
    seen: set[tuple[str, int]] = set()
    for item in raw:
        try:
            key = (str(item["video_id"]), int(item["scene_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        scene = allowed.get(key)
        if scene is None or key in seen:
            continue
        try:
            start = int(item["start_ms"])
            end = int(item["end_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        seen.add(key)
        citations.append(citation_for_scene(scene, start, end))
    if not citations and hits and fallback:
        top = hits[0].scene
        citations.append(citation_for_scene(top, top.start_ms, top.end_ms))
    return citations


def _plural(count: int, label: str) -> str:
    return label if count == 1 else f"{label}s"


class LocalExtractiveSynthesizer:
    provider = "local"

    def synthesize(
        self,
        question: str,
        hits: list[RetrievalHit],
        grounding: Grounding,
    ) -> Synthesis:
        scene = hits[0].scene
        facts = scene.facts or {}
        terms = parse_visual_terms(question)
        category = classify_question(question)
        colors = terms["colors"]
        shapes = terms["shapes"]

        if category == "counting":
            counts: dict[str, int] = facts.get("object_counts", {})
            matching = [
                (label, int(value))
                for label, value in counts.items()
                if (not colors or label.split()[0] in colors)
                and (not shapes or label.split()[-1] in shapes)
            ]
            total = sum(value for _, value in matching)
            described = " ".join(colors + shapes) or "objects"
            answer = f"{total} {_plural(total, described)} visible in scene {scene.scene_id}."
            evidence = {
                "scene_id": scene.scene_id,
                "object_counts": counts,
                "caption": scene.caption,
                "transcript": scene.transcript,
            }
            citation = citation_for_scene(scene, scene.start_ms, scene.end_ms)
            return Synthesis(answer, category, evidence, [citation], provider=self.provider)

        if category == "action":
            direction = terms["direction"] or "left-to-right"
            track = None
            for candidate in facts.get("tracks", []):
                if colors and candidate["color"] not in colors:
                    continue
                if shapes and candidate["shape"] not in shapes:
                    continue
                track = candidate
                break
            if track is None:
                answer = (
                    f"Insufficient evidence about that action in scene {scene.scene_id}."
                )
            elif track["direction"] == direction:
                answer = (
                    f"Yes, the {track['color']} {track['shape']} moves "
                    f"{direction.replace('-', ' ')} between {_fmt_ms(grounding.start_ms)} "
                    f"and {_fmt_ms(grounding.end_ms)}."
                )
            elif track["direction"] == "stationary":
                answer = (
                    f"No, the {track['color']} {track['shape']} remains stationary in "
                    f"scene {scene.scene_id}."
                )
            else:
                answer = (
                    f"No, the {track['color']} {track['shape']} moves "
                    f"{track['direction'].replace('-', ' ')}, not "
                    f"{direction.replace('-', ' ')}."
                )
            evidence = {
                "scene_id": scene.scene_id,
                "tracks": facts.get("tracks", []),
                "caption": scene.caption,
                "transcript": scene.transcript,
            }
            citation = citation_for_scene(scene, grounding.start_ms, grounding.end_ms)
            return Synthesis(answer, category, evidence, [citation], provider=self.provider)

        lowered = question.lower()
        background = facts.get("background_colors", [])
        if "background" in lowered and "color" in lowered and background:
            answer = f"The background is {background[0]} in scene {scene.scene_id}."
        elif re.search(r"\bis there\b|\bare there\b", lowered) and (colors or shapes):
            present = any(
                (not colors or label.split()[0] in colors)
                and (not shapes or label.split()[-1] in shapes)
                for label in facts.get("object_counts", {})
            )
            subject = " ".join(colors + shapes)
            answer = (
                f"Yes, there is a {subject} in scene {scene.scene_id}."
                if present
                else f"No, there is no {subject} in scene {scene.scene_id}."
            )
        else:
            answer = f"{scene.caption} (scene {scene.scene_id})"
        evidence = {
            "scene_id": scene.scene_id,
            "caption": scene.caption,
            "facts": facts,
            "transcript": scene.transcript,
        }
        citation = citation_for_scene(scene, scene.start_ms, scene.end_ms)
        return Synthesis(answer, category, evidence, [citation], provider=self.provider)


def _fmt_ms(ms: int) -> str:
    seconds, millis = divmod(max(0, ms), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


class GeminiSynthesizer:
    provider = "gemini"

    def __init__(
        self,
        client: GeminiClient | None = None,
        asset_root: str | Path = ".",
        **kwargs: Any,
    ) -> None:
        self.client = client or GeminiClient(**kwargs)
        self.asset_root = Path(asset_root)

    def synthesize(
        self,
        question: str,
        hits: list[RetrievalHit],
        grounding: Grounding,
    ) -> Synthesis:
        top = hits[:3]
        lines = [
            "Answer the question using only the provided scenes.",
            "Return JSON exactly like:",
            '{"answer": "...", "category": "counting|action|descriptive", '
            '"citations": [{"video_id": "...", "scene_id": 1, "start_ms": 0, "end_ms": 0}], '
            '"confidence": 0.0}',
            "Every citation must refer to one of the provided scenes and stay inside it.",
            f"Question: {question}",
        ]
        for hit in top:
            scene = hit.scene
            lines.append(
                f"Scene {scene.video_id}:{scene.scene_id} "
                f"[{scene.start_ms}-{scene.end_ms}ms] caption={scene.caption!r} "
                f"transcript={scene.transcript!r}"
            )
        parts: list[dict[str, Any]] = [{"text": "\n".join(lines)}]
        for hit in top:
            if hit.scene.keyframe_path:
                try:
                    parts.append(image_part(str(self.asset_root / hit.scene.keyframe_path)))
                except OSError:
                    continue
        rendered = self.client.generate(parts)
        parsed = extract_json(rendered)
        answer = str(parsed.get("answer", "")).strip()
        if not answer:
            raise ProviderError("Gemini answer was empty")
        raw_citations = parsed.get("citations", [])
        if not isinstance(raw_citations, list):
            raw_citations = []
        citations = validate_citations(raw_citations, hits, fallback=False)
        if not citations:
            raise ProviderError("Gemini returned no valid citations")
        category = str(parsed.get("category") or classify_question(question))
        confidence = parsed.get("confidence")
        return Synthesis(
            answer=answer,
            category=category if category in {"counting", "action", "descriptive"} else classify_question(question),
            evidence={"scenes": [h.scene.scene_id for h in top], "grounding": grounding.to_dict()},
            citations=citations,
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            provider=self.provider,
        )
