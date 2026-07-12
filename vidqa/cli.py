"""Command-line interface: corpus, ingest, ask, serve, eval."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import VidqaError, __version__
from .asr import FasterWhisperTranscriber, FixtureTranscriber
from .caption import GeminiCaptioner, ProviderError
from .corpus import build_corpus
from .evaluate import run_eval
from .ground import HTTPGrounder
from .index import LocalIndex
from .ingest import ingest_many
from .pipeline import AskService
from .qdrant import QdrantStore
from .synth import GeminiSynthesizer


def _format_ms(ms: int) -> str:
    seconds, millis = divmod(max(0, int(ms)), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def _build_synthesizer(kind: str, workdir: Path) -> Any:
    if kind == "local":
        return None
    if kind == "gemini":
        return GeminiSynthesizer(asset_root=workdir)
    raise VidqaError(f"unknown synthesizer: {kind}")


def _build_grounder(kind: str) -> Any:
    if kind == "local":
        return None
    url = os.environ.get("GROUNDER_URL", "")
    if kind != "http":
        raise VidqaError(f"unknown grounder: {kind}")
    if not url:
        raise VidqaError("--grounder http requires GROUNDER_URL")
    return HTTPGrounder(url)


def _build_qdrant(url: str | None, collection: str) -> QdrantStore | None:
    if not url:
        return None
    return QdrantStore(
        url=url,
        collection=collection,
        api_key=os.environ.get("QDRANT_API_KEY"),
    )


def _service(args: argparse.Namespace) -> AskService:
    workdir = Path(args.workdir)
    return AskService(
        workdir,
        synthesizer=_build_synthesizer(args.synth, workdir),
        grounder=_build_grounder(args.grounder),
        qdrant=_build_qdrant(getattr(args, "qdrant_url", None), getattr(args, "qdrant_collection", "vidqa_scenes")),
    )


def _cmd_corpus(args: argparse.Namespace) -> int:
    result = build_corpus(
        args.out,
        seed=args.seed,
        videos=args.videos,
        scenes_per_video=args.scenes,
        scene_ms=args.scene_ms,
        fps=args.fps,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    transcriber = None
    if args.asr == "faster-whisper":
        transcriber = FasterWhisperTranscriber()
    captioner = GeminiCaptioner() if args.captioner == "gemini" else None
    results = ingest_many(
        args.sources,
        workdir=args.workdir,
        transcripts_dir=args.transcripts,
        transcriber=transcriber,
        captioner=captioner,
        force=args.force,
        limit=args.limit,
        scene_threshold=args.scene_threshold,
    )
    qdrant = _build_qdrant(args.qdrant_url, args.qdrant_collection)
    if qdrant is not None:
        index = LocalIndex(args.workdir)
        qdrant.ensure_collection(recreate=args.recreate)
        count = qdrant.sync(index)
        print(f"[qdrant] synced {count} scenes to {args.qdrant_collection}", file=sys.stderr)
    print(
        json.dumps(
            {
                "videos": len(results),
                "scenes": sum(result.scene_count for result in results if not result.skipped),
                "skipped": sum(1 for result in results if result.skipped),
            },
            indent=2,
        )
    )
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    if args.sources:
        ingest_many(
            args.sources,
            workdir=args.workdir,
            transcripts_dir=args.transcripts,
            transcriber=FasterWhisperTranscriber() if args.asr == "faster-whisper" else None,
            force=args.force,
        )
    service = _service(args)
    answer = service.ask(args.question, k=args.k, video_id=args.video_id)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
        return 0
    top = answer.retrieval[0] if answer.retrieval else None
    if top:
        print(
            f"[query]    top scene: scene {top['scene_id']} "
            f"[{_format_ms(top['start_ms'])}-{_format_ms(top['end_ms'])}], "
            f"score {top['score']:.4f}",
            file=sys.stderr,
        )
    print(
        f"[ground]   refined window: "
        f"[{_format_ms(answer.grounding['start_ms'])}-{_format_ms(answer.grounding['end_ms'])}]",
        file=sys.stderr,
    )
    print(f"[synth]    {answer.provider}", file=sys.stderr)
    print(f"answer:    {answer.answer}")
    for citation in answer.citations:
        print(
            f"citations: [scene {citation.scene_id}: "
            f"{_format_ms(citation.start_ms)}-{_format_ms(citation.end_ms)}]"
        )
        if citation.frames:
            previews = ", ".join(_format_ms(frame["t_ms"]) for frame in citation.frames)
            print(f"           [frame preview at {previews}]")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    service = AskService(
        args.workdir,
        synthesizer=_build_synthesizer(args.synth, Path(args.workdir)),
        grounder=_build_grounder(args.grounder),
        qdrant=_build_qdrant(args.qdrant_url, args.qdrant_collection),
    )
    serve(service, host=args.host, port=args.port)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    service = _service(args)
    result = run_eval(
        service,
        args.suite,
        report_path=args.report,
        json_path=args.json_out,
    )
    print(result["markdown"])
    print(f"report: {args.report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidqa",
        description="Scene-indexed video QA with multi-vector retrieval and cited answers",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus", help="generate the deterministic demo corpus")
    corpus.add_argument("--out", default="data/demo")
    corpus.add_argument("--seed", type=int, default=7)
    corpus.add_argument("--videos", type=int, default=6)
    corpus.add_argument("--scenes", type=int, default=5)
    corpus.add_argument("--scene-ms", type=int, default=4000)
    corpus.add_argument("--fps", type=int, default=8)
    corpus.set_defaults(func=_cmd_corpus)

    ingest = sub.add_parser("ingest", help="index local videos or URLs")
    ingest.add_argument("sources", nargs="+")
    ingest.add_argument("--workdir", default="data/index")
    ingest.add_argument("--transcripts", default=None)
    ingest.add_argument("--asr", choices=("fixture", "faster-whisper"), default="fixture")
    ingest.add_argument("--captioner", choices=("local", "gemini"), default="local")
    ingest.add_argument("--force", action="store_true")
    ingest.add_argument("--limit", type=int, default=None)
    ingest.add_argument("--scene-threshold", type=float, default=27.0)
    ingest.add_argument("--qdrant-url", default=None)
    ingest.add_argument("--qdrant-collection", default="vidqa_scenes")
    ingest.add_argument("--recreate", action="store_true")
    ingest.set_defaults(func=_cmd_ingest)

    ask = sub.add_parser("ask", help="answer a question over the index")
    ask.add_argument("question")
    ask.add_argument("--workdir", default="data/index")
    ask.add_argument("--video-id", default=None)
    ask.add_argument("--k", type=int, default=5)
    ask.add_argument("--json", action="store_true")
    ask.add_argument("--synth", choices=("local", "gemini"), default="local")
    ask.add_argument("--grounder", choices=("local", "http"), default="local")
    ask.add_argument("--source", dest="sources", action="append", default=[])
    ask.add_argument("--transcripts", default=None)
    ask.add_argument("--asr", choices=("fixture", "faster-whisper"), default="fixture")
    ask.add_argument("--force", action="store_true")
    ask.add_argument("--qdrant-url", default=None)
    ask.add_argument("--qdrant-collection", default="vidqa_scenes")
    ask.set_defaults(func=_cmd_ask)

    serve = sub.add_parser("serve", help="serve the citation UI and JSON API")
    serve.add_argument("--workdir", default="data/index")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--synth", choices=("local", "gemini"), default="local")
    serve.add_argument("--grounder", choices=("local", "http"), default="local")
    serve.add_argument("--qdrant-url", default=None)
    serve.add_argument("--qdrant-collection", default="vidqa_scenes")
    serve.set_defaults(func=_cmd_serve)

    evaluate = sub.add_parser("eval", help="evaluate QA, retrieval, grounding")
    evaluate.add_argument("--suite", required=True)
    evaluate.add_argument("--workdir", default="data/index")
    evaluate.add_argument("--report", default="outputs/eval-report.md")
    evaluate.add_argument("--json-out", default="outputs/eval-report.json")
    evaluate.add_argument("--synth", choices=("local", "gemini"), default="local")
    evaluate.add_argument("--grounder", choices=("local", "http"), default="local")
    evaluate.add_argument("--qdrant-url", default=None)
    evaluate.add_argument("--qdrant-collection", default="vidqa_scenes")
    evaluate.set_defaults(func=_cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (VidqaError, ProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
