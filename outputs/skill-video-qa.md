# Skill: Video QA with timestamped citations

Use this project when a question needs an answer grounded in specific moments of a long video corpus, not a whole-video summary.

## Capabilities

- Index local MP4/MOV/MKV/WebM/AVI files, directories, or `yt-dlp` URLs.
- Detect scenes, extract keyframes and preview frames, align word-level transcripts, caption/analyze each scene, and store three named vectors per scene.
- Answer natural-language questions with `(video_id, start_ms, end_ms)` citations, frame previews, per-retriever evidence, and a grounded window.
- Evaluate QA accuracy, counting/action error, retrieval hit rate, citation validity, and temporal IoU on a JSONL suite.

## Workflow

```bash
# Build the index once. Add --force to re-index a source.
uv run python -m vidqa ingest <video-or-directory>... \
  --transcripts <word-timestamp-dir> --workdir data/index

# Ask. --json returns the machine contract below.
uv run python -m vidqa ask "How many red rectangles are visible in the scene about signal?" \
  --workdir data/index --json

# Interactive citation UI + JSON API.
uv run python -m vidqa serve --workdir data/index --port 8765

# Reproducible evaluation.
uv run python -m vidqa eval --suite data/demo/benchmark.jsonl --workdir data/index \
  --report outputs/eval-report.md --json-out outputs/eval-report.json
```

For a URL, install the optional downloader without a global install:
`uv run --with yt-dlp python -m vidqa ingest https://... --asr faster-whisper` (also needs
`--with faster-whisper`). Offline, provide a `*.words.json` sidecar next to the media or in
`--transcripts`; missing transcripts fail explicitly.

## Answer contract

```json
{
  "question": "…",
  "answer": "2 red rectangles visible in scene 1.",
  "category": "counting",
  "citations": [
    {
      "video_id": "v001",
      "scene_id": 1,
      "start_ms": 0,
      "end_ms": 4000,
      "frames": [{"t_ms": 2000, "path": "videos/v001/frames/…jpg", "url": "/media/…jpg"}]
    }
  ],
  "evidence": {"object_counts": {"red rectangle": 2}},
  "retrieval": [
    {"video_id": "v001", "scene_id": 1, "score": 0.1,
     "ranks": {"caption": 1, "keyframe": 2, "transcript": 1, "lexical": 1}}
  ],
  "confidence": 0.66,
  "provider": "local",
  "grounding": {"start_ms": 950, "end_ms": 2720, "score": 0.55, "method": "local"}
}
```

Rules:

- Cite only `video_id`/`scene_id` pairs present in `retrieval`; the server clips timestamps to the scene and drops unknown scenes.
- `frame.url` is directly servable under `/media/`; `frame.path` is relative to the workdir.
- `grounding` is the refined window used for action citations; counting/descriptive answers cite the full scene.
- `confidence` is an uncalibrated fusion score, not a probability.
- `provider` is `local`, `gemini`, or a local fallback after a provider error.

## Choosing a mode

- **Offline/local (default):** inspectable classical vision, hashed embeddings, local index, extractive synthesis. Best for deterministic CI and demos.
- **Gemini:** `--captioner gemini` at ingest and/or `--synth gemini` at ask; requires `GEMINI_API_KEY`. Model/base are `GEMINI_MODEL` / `GEMINI_API_BASE`. Validate citations after any model change.
- **faster-whisper:** `--asr faster-whisper`; downloads the `turbo` checkpoint and emits word timestamps. CPU int8 by default.
- **Qdrant:** ingest/ask with `--qdrant-url`; the local bundles remain necessary for frames and grounding.
- **TimeLens/VideoITG:** `GROUNDER_URL` + `--grounder http`; the service must return a window inside the supplied scene.

## Failure modes to report

- Missing transcript sidecar: the CLI exits with the searched paths; do not fabricate a transcript.
- No retrieved scenes: ask returns a 404-style “no indexed scenes matched” error.
- Counting mismatch: inspect `evidence.object_counts`; the local detector can merge same-color blobs or misclassify a compressed rectangle as a circle.
- Real benchmark claims: the bundled suite is synthetic. Public benchmark loaders require the datasets and videos, and no public score is measured here.
