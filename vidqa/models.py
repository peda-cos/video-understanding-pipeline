"""JSON-safe data models shared by ingest, retrieval, and the web API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Word:
    text: str
    start_ms: int
    end_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Word":
        return cls(
            text=str(data["text"]),
            start_ms=int(data["start_ms"]),
            end_ms=int(data["end_ms"]),
        )


@dataclass
class Thumb:
    t_ms: int
    path: str
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Thumb":
        return cls(
            t_ms=int(data["t_ms"]),
            path=str(data["path"]),
            url=str(data.get("url", "")),
        )


@dataclass
class Video:
    video_id: str
    source: str
    file_path: str
    duration_ms: int
    width: int
    height: int
    fps: float
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Video":
        return cls(
            video_id=str(data["video_id"]),
            source=str(data.get("source", "")),
            file_path=str(data["file_path"]),
            duration_ms=int(data["duration_ms"]),
            width=int(data["width"]),
            height=int(data["height"]),
            fps=float(data["fps"]),
            title=str(data.get("title", "")),
        )


@dataclass
class Scene:
    video_id: str
    scene_id: int
    start_ms: int
    end_ms: int
    keyframe_path: str = ""
    keyframe_url: str = ""
    thumbs: list[Thumb] = field(default_factory=list)
    caption: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    words: list[Word] = field(default_factory=list)
    transcript: str = ""

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def to_dict(self, include_words: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "video_id": self.video_id,
            "scene_id": self.scene_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "keyframe_path": self.keyframe_path,
            "keyframe_url": self.keyframe_url,
            "thumbs": [t.to_dict() for t in self.thumbs],
            "caption": self.caption,
            "facts": self.facts,
            "transcript": self.transcript,
        }
        if include_words:
            data["words"] = [w.to_dict() for w in self.words]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scene":
        return cls(
            video_id=str(data["video_id"]),
            scene_id=int(data["scene_id"]),
            start_ms=int(data["start_ms"]),
            end_ms=int(data["end_ms"]),
            keyframe_path=str(data.get("keyframe_path", "")),
            keyframe_url=str(data.get("keyframe_url", "")),
            thumbs=[Thumb.from_dict(t) for t in data.get("thumbs", [])],
            caption=str(data.get("caption", "")),
            facts=dict(data.get("facts", {})),
            words=[Word.from_dict(w) for w in data.get("words", [])],
            transcript=str(data.get("transcript", "")),
        )


@dataclass
class RetrievalHit:
    scene: Scene
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scene.to_dict(include_words=False),
            "score": round(self.score, 6),
            "ranks": self.ranks,
            "scores": {k: round(v, 6) for k, v in self.scores.items()},
        }


@dataclass
class Citation:
    video_id: str
    scene_id: int
    start_ms: int
    end_ms: int
    frames: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    question: str
    answer: str
    category: str
    citations: list[Citation]
    evidence: dict[str, Any]
    retrieval: list[dict[str, Any]]
    confidence: float
    provider: str
    grounding: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "citations": [c.to_dict() for c in self.citations],
            "evidence": self.evidence,
            "retrieval": self.retrieval,
            "confidence": round(self.confidence, 6),
            "provider": self.provider,
            "grounding": self.grounding,
        }
