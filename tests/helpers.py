"""Shared deterministic fixtures built once per test process."""

from __future__ import annotations

import atexit
import functools
import shutil
import tempfile
from pathlib import Path
from typing import Any

from vidqa.corpus import build_corpus
from vidqa.ingest import ingest_many
from vidqa.pipeline import AskService


@functools.lru_cache(maxsize=1)
def demo_paths() -> tuple[Path, Path, Path]:
    root = Path(tempfile.mkdtemp(prefix="vidqa-tests-"))
    corpus = root / "demo"
    index = root / "index"
    build_corpus(
        corpus,
        seed=0,
        videos=2,
        scenes_per_video=2,
        scene_ms=2000,
        fps=8,
    )
    ingest_many(
        [str(corpus / "media")],
        workdir=index,
        transcripts_dir=corpus / "transcripts",
    )
    atexit.register(shutil.rmtree, root, True)
    return corpus, index, corpus / "benchmark.jsonl"


@functools.lru_cache(maxsize=1)
def service() -> AskService:
    _corpus, index, _benchmark = demo_paths()
    return AskService(index)


def specs() -> dict[str, Any]:
    import json

    corpus, _index, _benchmark = demo_paths()
    return json.loads((corpus / "specs.json").read_text(encoding="utf-8"))
