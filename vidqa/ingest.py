"""Ingest walker: materialize media, segment scenes, align ASR, analyze, persist."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import VidqaError, asr, media, segment, vision
from .analyze import LocalAnalyzer
from .caption import Captioner
from .embed import TextEmbedder
from .models import Scene, Thumb, Video

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
THUMB_FRACTIONS = (0.12, 0.50, 0.88)


@dataclass
class IngestResult:
    video_id: str
    title: str
    scene_count: int
    skipped: bool
    elapsed_s: float
    source: str


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return cleaned[:48] or "video"


def _source_hash(source: str) -> str:
    return hashlib.blake2b(source.encode("utf-8"), digest_size=4).hexdigest()


def read_catalog(workdir: str | Path) -> dict[str, Any]:
    path = Path(workdir) / "catalog.json"
    if not path.exists():
        return {"version": 1, "videos": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("videos"), list):
        raise VidqaError(f"invalid catalog at {path}")
    return data


def write_catalog(workdir: str | Path, catalog: dict[str, Any]) -> None:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "catalog.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(catalog, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def catalog_record(workdir: str | Path, video_id: str) -> dict[str, Any] | None:
    for record in read_catalog(workdir)["videos"]:
        if record["video_id"] == video_id:
            return record
    return None


def catalog_by_source(workdir: str | Path, source: str) -> dict[str, Any] | None:
    for record in read_catalog(workdir)["videos"]:
        if record.get("source") == source:
            return record
    return None


def _pick_video_id(source: str, catalog: dict[str, Any]) -> str:
    existing = next((r for r in catalog["videos"] if r.get("source") == source), None)
    if existing:
        return str(existing["video_id"])
    base = _slug(Path(source).stem if not _is_url(source) else source.rstrip("/").split("/")[-1])
    used = {str(r["video_id"]) for r in catalog["videos"]}
    if base not in used:
        return base
    return f"{base}-{_source_hash(source)}"


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _download_url(source: str, destination: Path) -> Path:
    try:
        import yt_dlp  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        raise VidqaError(
            "yt-dlp is not installed; run `uv run --with yt-dlp python -m vidqa ingest URL ...`"
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    template = str(destination.parent / f"{destination.stem}.%(ext)s")
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format",
        "mp4",
        "-o",
        template,
        "--no-playlist",
        source,
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()[-800:]
        raise VidqaError(f"yt-dlp failed: {detail}")
    candidates = sorted(
        p
        for p in destination.parent.glob(f"{destination.stem}.*")
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not candidates:
        raise VidqaError(f"yt-dlp produced no video for {source}")
    return candidates[0]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _media_url(relative_path: str) -> str:
    return "/media/" + relative_path.replace(os.sep, "/")


def _extract_scene_assets(
    source_path: Path,
    staging: Path,
    video_id: str,
    scenes: list[tuple[int, int]],
) -> list[Scene]:
    analyzer = LocalAnalyzer()
    out: list[Scene] = []
    for index, (start_ms, end_ms) in enumerate(scenes, start=1):
        duration = end_ms - start_ms
        keyframe_ms = start_ms + duration // 2
        thumb_times = [
            int(start_ms + duration * frac) for frac in THUMB_FRACTIONS
        ]
        thumb_times[len(thumb_times) // 2] = keyframe_ms
        thumbs: list[Thumb] = []
        frame_files: list[Path] = []
        for t_ms in thumb_times:
            filename = f"s{index:03d}_t{t_ms:08d}.jpg"
            rel = f"videos/{video_id}/frames/{filename}"
            local = staging / "frames" / filename
            media.extract_jpeg(source_path, t_ms, local)
            frame_files.append(local)
            thumbs.append(Thumb(t_ms=t_ms, path=rel, url=_media_url(rel)))
        image_arrays = [media.decode_image(local, width=192) for local in frame_files]
        facts = analyzer.analyze(image_arrays, thumb_times)[1]
        caption = str(facts["caption"])
        rel_keyframe = f"videos/{video_id}/frames/s{index:03d}_t{keyframe_ms:08d}.jpg"
        out.append(
            Scene(
                video_id=video_id,
                scene_id=index,
                start_ms=start_ms,
                end_ms=end_ms,
                keyframe_path=rel_keyframe,
                keyframe_url=_media_url(rel_keyframe),
                thumbs=thumbs,
                caption=caption,
                facts=facts,
                transcript="",
            )
        )
    return out


def _apply_captioner(
    scenes: list[Scene],
    staging: Path,
    captioner: Captioner | None,
) -> None:
    if captioner is None:
        return
    for scene in scenes:
        try:
            caption = captioner.caption(
                str(staging / "frames" / Path(scene.keyframe_path).name), scene.facts
            )
        except VidqaError:
            raise
        except Exception as exc:  # provider failures must not be silently hidden
            raise VidqaError(f"captioner failed for scene {scene.scene_id}: {exc}") from exc
        if caption.strip():
            scene.caption = caption.strip()


def _plausible_keyframe(scene: Scene) -> bool:
    return bool(scene.keyframe_path) and bool(scene.keyframe_url)


def ingest_one(
    source: str,
    workdir: str | Path,
    transcripts_dir: str | Path | None = None,
    transcriber: asr.WordTranscriber | None = None,
    captioner: Captioner | None = None,
    force: bool = False,
    scene_threshold: float = 27.0,
    min_scene_ms: int = 600,
) -> IngestResult:
    started = time.perf_counter()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog(workdir)
    video_id = _pick_video_id(source, catalog)
    final_dir = workdir / "videos" / video_id
    existing = next((r for r in catalog["videos"] if r["video_id"] == video_id), None)
    if final_dir.joinpath("manifest.json").exists() and not force:
        return IngestResult(
            video_id=video_id,
            title=str(existing.get("title", video_id)) if existing else video_id,
            scene_count=int(existing.get("scene_count", 0)) if existing else 0,
            skipped=True,
            elapsed_s=time.perf_counter() - started,
            source=source,
        )

    staging = workdir / ".staging" / f"{video_id}-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    try:
        downloaded = staging / "download.mp4"
        if _is_url(source):
            raw = _download_url(source, downloaded)
        else:
            raw = Path(source).expanduser().resolve()
            if not raw.exists():
                raise VidqaError(f"source does not exist: {source}")
        materialized = staging / "source.mp4"
        media.to_720p(raw, materialized)
        if raw != materialized:
            try:
                raw.resolve().relative_to(staging.resolve())
                raw.unlink(missing_ok=True)
            except ValueError:
                pass
        info = media.probe(materialized)
        if info["duration_ms"] <= 0:
            raise VidqaError(f"materialized video has no duration: {source}")

        scene_ranges = segment.detect_scenes(
            str(materialized),
            threshold=scene_threshold,
            min_scene_ms=min_scene_ms,
        )
        scenes = _extract_scene_assets(materialized, staging, video_id, scene_ranges)
        _apply_captioner(scenes, staging, captioner)

        active_transcriber = transcriber
        if active_transcriber is None:
            active_transcriber = asr.FixtureTranscriber(transcripts_dir)
        try:
            transcript_source = (
                raw
                if isinstance(active_transcriber, asr.FixtureTranscriber)
                and not _is_url(source)
                else materialized
            )
            words = active_transcriber.transcribe(transcript_source)
        except asr.TranscriptUnavailable as exc:
            raise VidqaError(str(exc)) from exc
        sliced = asr.slice_words_by_scenes(words, scene_ranges)
        for scene, scene_words in zip(scenes, sliced, strict=True):
            scene.words = scene_words
            scene.transcript = asr.transcript_text(scene_words)

        embedder = TextEmbedder()
        caption_vecs: list[np.ndarray] = []
        keyframe_vecs: list[np.ndarray] = []
        transcript_vecs: list[np.ndarray] = []
        transcript_present: list[bool] = []
        thumb_vecs: list[np.ndarray] = []
        for scene in scenes:
            frame_arrays = [
                media.decode_image(staging / "frames" / Path(thumb.path).name, width=192)
                for thumb in scene.thumbs
            ]
            caption_vecs.append(embedder.encode(scene.caption))
            keyframe_vecs.append(vision.visual_descriptor(frame_arrays, scene.facts))
            transcript_vecs.append(embedder.encode(scene.transcript))
            transcript_present.append(bool(scene.transcript.strip()))
            thumb_vecs.append(
                np.stack(
                    [
                        vision.visual_descriptor([frame], scene.facts)
                        for frame in frame_arrays
                    ]
                )
            )

        video = Video(
            video_id=video_id,
            source=source,
            file_path=f"videos/{video_id}/source.mp4",
            duration_ms=int(info["duration_ms"]),
            width=int(info["width"]),
            height=int(info["height"]),
            fps=float(info["fps"]),
            title=existing.get("title", "") if existing else Path(source).stem,
        )
        manifest = {
            "version": 1,
            "video": video.to_dict(),
            "scenes": [scene.to_dict(include_words=True) for scene in scenes],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        _write_jsonl(
            staging / "scenes.jsonl",
            [scene.to_dict(include_words=False) for scene in scenes],
        )
        np.savez_compressed(
            staging / "vectors.npz",
            caption=np.stack(caption_vecs).astype(np.float32),
            keyframe=np.stack(keyframe_vecs).astype(np.float32),
            transcript=np.stack(transcript_vecs).astype(np.float32),
            transcript_present=np.array(transcript_present, dtype=bool),
            thumbs=np.stack(thumb_vecs).astype(np.float32),
        )

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            trash = workdir / ".trash" / f"{video_id}-{uuid.uuid4().hex[:8]}"
            trash.parent.mkdir(parents=True, exist_ok=True)
            os.replace(final_dir, trash)
            shutil.rmtree(trash, ignore_errors=True)
        os.replace(staging, final_dir)

        record = {
            "video_id": video_id,
            "source": source,
            "file_path": video.file_path,
            "title": video.title or video_id,
            "duration_ms": video.duration_ms,
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "scene_count": len(scenes),
        }
        catalog["videos"] = [
            r for r in catalog["videos"] if r["video_id"] != video_id
        ] + [record]
        catalog["videos"].sort(key=lambda r: r["video_id"])
        write_catalog(workdir, catalog)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return IngestResult(
        video_id=video_id,
        title=video.title or video_id,
        scene_count=len(scenes),
        skipped=False,
        elapsed_s=time.perf_counter() - started,
        source=source,
    )


def expand_sources(paths: list[str]) -> list[str]:
    expanded: list[str] = []
    for raw in paths:
        if _is_url(raw):
            expanded.append(raw)
            continue
        path = Path(raw).expanduser()
        if path.is_dir():
            expanded.extend(
                str(p)
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
            )
        elif path.exists():
            expanded.append(str(path))
        else:
            raise VidqaError(f"source does not exist: {raw}")
    return expanded


def ingest_many(
    sources: list[str],
    workdir: str | Path,
    transcripts_dir: str | Path | None = None,
    transcriber: asr.WordTranscriber | None = None,
    captioner: Captioner | None = None,
    force: bool = False,
    limit: int | None = None,
    scene_threshold: float = 27.0,
    min_scene_ms: int = 600,
) -> list[IngestResult]:
    paths = expand_sources(sources)
    if limit is not None:
        paths = paths[:limit]
    results: list[IngestResult] = []
    for source in paths:
        result = ingest_one(
            source,
            workdir=workdir,
            transcripts_dir=transcripts_dir,
            transcriber=transcriber,
            captioner=captioner,
            force=force,
            scene_threshold=scene_threshold,
            min_scene_ms=min_scene_ms,
        )
        results.append(result)
        status = "skipped" if result.skipped else f"{result.scene_count} scenes"
        print(
            f"[ingest] {result.video_id}: {status} in {result.elapsed_s:.2f}s",
            file=sys.stderr,
        )
    return results


def load_manifest(workdir: str | Path, video_id: str) -> dict[str, Any]:
    path = Path(workdir) / "videos" / video_id / "manifest.json"
    if not path.exists():
        raise VidqaError(f"no manifest for video {video_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_scenes(workdir: str | Path, video_id: str) -> list[Scene]:
    data = load_manifest(workdir, video_id)
    return [Scene.from_dict(row) for row in data.get("scenes", [])]
