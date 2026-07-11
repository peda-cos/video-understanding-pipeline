"""ASR adapters and scene-level word alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from . import VidqaError
from .models import Word


class TranscriptUnavailable(VidqaError):
    pass


class WordTranscriber(Protocol):
    def transcribe(self, media_path: str | Path) -> list[Word]: ...


def _parse_word(data: dict) -> Word:
    if "start_ms" in data:
        start = int(data["start_ms"])
        end = int(data["end_ms"])
    else:
        start = int(round(float(data.get("start", 0.0)) * 1000))
        end = int(round(float(data.get("end", 0.0)) * 1000))
    text = str(data.get("word", data.get("text", ""))).strip()
    if not text:
        raise TranscriptUnavailable("transcript contains an empty word")
    if end <= start:
        raise TranscriptUnavailable(f"invalid word interval: {data}")
    return Word(text=text, start_ms=start, end_ms=max(end, start + 1))


def load_words_file(path: str | Path) -> list[Word]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("words", [])
    if not isinstance(raw, list):
        raise TranscriptUnavailable(f"words must be a list in {path}")
    words = [_parse_word(item) for item in raw]
    return sorted(words, key=lambda w: (w.start_ms, w.end_ms))


class FixtureTranscriber:
    """Read deterministic word timestamps from a JSON sidecar."""

    def __init__(self, transcripts_dir: str | Path | None = None) -> None:
        self.transcripts_dir = Path(transcripts_dir) if transcripts_dir else None

    def sidecar_for(self, media_path: str | Path) -> Path:
        source = Path(media_path)
        name = f"{source.stem}.words.json"
        candidates: list[Path] = []
        if self.transcripts_dir:
            candidates.append(self.transcripts_dir / name)
        candidates.append(source.with_name(name))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        searched = ", ".join(str(c) for c in candidates)
        raise TranscriptUnavailable(
            f"no word-timestamp transcript found for {source.name} (searched: {searched}). "
            "Pass --transcripts, place the sidecar next to the media, or use --asr faster-whisper."
        )

    def transcribe(self, media_path: str | Path) -> list[Word]:
        return load_words_file(self.sidecar_for(media_path))


class FasterWhisperTranscriber:
    """Optional real ASR path; requires the `faster-whisper` package and a model."""

    def __init__(
        self,
        model_size: str = "turbo",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TranscriptUnavailable(
                "faster-whisper is not installed; run "
                "`uv run --with faster-whisper python -m vidqa ... --asr faster-whisper`"
            ) from exc
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, media_path: str | Path) -> list[Word]:
        segments, _info = self._model.transcribe(str(media_path), word_timestamps=True)
        words: list[Word] = []
        for segment in segments:
            for word in segment.words or []:
                text = str(word.word).strip()
                if not text:
                    continue
                start = int(round(float(word.start) * 1000))
                end = int(round(float(word.end) * 1000))
                if end <= start:
                    continue
                words.append(Word(text=text, start_ms=start, end_ms=end))
        return sorted(words, key=lambda w: (w.start_ms, w.end_ms))


def slice_words_by_scenes(
    words: list[Word], scenes: list[tuple[int, int]]
) -> list[list[Word]]:
    """Assign each word to the scene containing its midpoint, clipping to bounds."""
    buckets: list[list[Word]] = [[] for _ in scenes]
    if not scenes:
        return buckets
    for word in words:
        midpoint = (word.start_ms + word.end_ms) // 2
        index = len(scenes) - 1
        for i, (start, end) in enumerate(scenes):
            if start <= midpoint < end:
                index = i
                break
            if midpoint < start:
                index = max(0, i - 1)
                break
        start, end = scenes[index]
        clamped_start = max(start, word.start_ms)
        clamped_end = min(end, word.end_ms)
        if clamped_end <= clamped_start:
            clamped_start = max(start, min(word.start_ms, end - 1))
            clamped_end = min(end, clamped_start + 1)
        buckets[index].append(
            Word(text=word.text, start_ms=clamped_start, end_ms=clamped_end)
        )
    return buckets


def transcript_text(words: list[Word]) -> str:
    return " ".join(w.text.strip() for w in words if w.text.strip()).strip()
