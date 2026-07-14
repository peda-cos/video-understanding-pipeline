# Video Understanding Pipeline

Scene-indexed video QA: segment a video into shots, align word-level transcripts, caption and embed every scene, retrieve with a multi-vector hybrid index, refine a `(start_ms, end_ms)` window, and answer with validated citations plus frame previews.

This is the Capstone 12 implementation adapted for deterministic local execution. The default path uses no hosted model, GPU, or vector database: real ffmpeg decoding, classical frame analysis, feature-hashed text embeddings, a local named-vector index, and a local grounding/synthesis stack. Real adapters for Gemini, faster-whisper, Qdrant, and a TimeLens-style grounding service are included and documented below.

```
video / URL
   │ ingest: 720p normalize · ffprobe · atomic bundle
   ├─ scene detection (HSV adjacent-frame delta, 4 fps, 160×90)
   │     └─ keyframe + 3 preview frames → local analysis (colors, blobs, motion)
   │           └─ caption + structured facts + 80-d visual descriptor
   ├─ ASR (fixture word JSON or faster-whisper turbo)
   │     └─ word timestamps → per-scene transcript slices → 384-d text vector
   └─ bundle: manifest.json · scenes.jsonl · vectors.npz (caption/keyframe/transcript/thumbs)

query
   ├─ caption, keyframe, transcript dense retrieval + IDF lexical retrieval
   ├─ reciprocal-rank fusion (k=60, top-5) + rare-term anchoring
   ├─ temporal grounding inside the top scene (text + visual + length prior)
   └─ local extractive synthesizer or Gemini → answer + citations + frame URLs
        └─ CLI · JSON HTTP API · TypeScript UI with jump-to-frame
```

## Quickstart

Prerequisites: Python 3.14 + `uv`, `ffmpeg`/`ffprobe`, Bun 1.4, and a POSIX shell. `./verify.sh` installs pinned Python and UI dependencies into ignored local directories, typechecks/builds the TypeScript UI, and runs the test suite.

```bash
./verify.sh

# 1. Generate a deterministic 6-video / 30-scene / 100-query corpus (120 s, 384×216).
uv run python -m vidqa corpus --out data/demo --seed 7

# 2. Index it. Transcribe from the fixture word-timestamp sidecars.
uv run python -m vidqa ingest data/demo/media \
  --transcripts data/demo/transcripts --workdir data/index

# 3. Ask a counting, action, or descriptive question.
uv run python -m vidqa ask "How many red rectangles are visible in the scene about signal?" \
  --workdir data/index

# 4. Reproduce the measured benchmark.
uv run python -m vidqa eval --suite data/demo/benchmark.jsonl --workdir data/index

# 5. Open the citation UI/API.
uv run python -m vidqa serve --workdir data/index --port 8765
# http://127.0.0.1:8765
```

`ask --json` emits the full answer contract (answer, category, citations with frame paths, retrieval scores/ranks, grounding). Every citation is clipped to a retrieved scene; unsupported citations are dropped and replaced with the top scene.

## Build-it mapping

| Spec step | Implementation |
| --- | --- |
| 1. Ingest walker | `vidqa/ingest.py`: local files/directories, optional `yt-dlp` URLs, 720p normalization, catalog `{video_id, source, file_path}`. |
| 2. Scene segmentation | `vidqa/segment.py`: ContentDetector-style HSV deltas at 4 fps; thresholds 27 / 600 ms; keyframe at midpoint, previews at 12/50/88%. |
| 3. ASR pass | `vidqa/asr.py`: fixture `*.words.json` or `faster-whisper` word timestamps; midpoint assignment and scene clipping. |
| 4. VLM captioning | `vidqa/analyze.py`, `vision.py`: classical color/blob/motion analysis + template caption; optional Gemini keyframe captioner. |
| 5. Multi-vector index | `vidqa/index.py`, `qdrant.py`: named `caption`, `keyframe`, `transcript` vectors + payload `{video_id, scene_id, start_ms, end_ms, keyframe_url}`, plus facts/transcript. |
| 6. Query | `vidqa/pipeline.py`, `index.py`: three dense spaces + IDF lexical retrieval, weighted RRF (`k=60`), top-5 scenes. |
| 7. Temporal grounding | `vidqa/ground.py`: candidate windows from word boundaries/preview times, text/visual/length scoring; optional HTTP TimeLens/VideoITG adapter. |
| 8. VLM synth | `vidqa/synth.py`: local extractive answers with citations; optional Gemini JSON synthesizer with citation validation. |
| 9. Eval | `vidqa/evaluate.py`, `corpus.py`: 100-query synthetic suite (30 counting / 30 action / 40 descriptive), per-class accuracy, hallucination proxies, retrieval, IoU, latency. |
| UI | `web/`: strict TypeScript SPA served by Python; player, scene strip, timestamp links, evidence ranks, frame previews. |

## What runs offline

Everything in the quickstart and `./verify.sh` runs offline after dependency installation: scene detection, frame analysis, embeddings, retrieval, grounding, synthesis, evaluation, HTTP server, and the UI. Tests make no network calls.

The deterministic local stack is deliberately inspectable:

- **Scene analysis** measures actual pixels: HSV color classes, connected components, shape from bounding-box aspect, centroid displacement across three sampled frames. It is not a semantic VLM.
- **Text vectors** are 384-d signed feature hashes of unigrams/bigrams (blake2b), L2-normalized. Visual vectors are 80-d color/count/shape/motion/layout descriptors. A symbolic query encoder maps color/shape/direction words into the same visual space, standing in for CLIP.
- **Lexical retrieval** uses IDF over scene captions/transcripts and anchors queries containing rare terms (document frequency ≤ 2) to the scenes that contain them. This is a hybrid-retrieval safeguard for “the scene about *keyword*” questions; without it, hashed dense vectors alone misrank keyword queries.
- **Grounding** scores candidate windows with `0.45·text cosine + 0.25·lexical coverage + 0.20·visual cosine + 0.10·length prior`, with deterministic tie-breaking.
- **Synthesis** is extractive: counts come from detected object counts, action answers from tracked direction, descriptive answers from facts/caption. It says “insufficient evidence” rather than inventing.

## Measured results

Measured locally on the generated synthetic suite: 6 videos, 30 scenes, 120 s, 100 queries. Gold labels come from the scene specs, independently of the analyzer; this validates mechanics and citation behavior, not real-world QA accuracy.

| Metric | Result |
| --- | --- |
| QA accuracy (100 queries) | 97.0% |
| Counting accuracy / error (n=30) | 93.3% / 6.7% |
| Action accuracy / error (n=30) | 100.0% / 0.0% |
| Descriptive accuracy (n=40) | 97.5% |
| Retrieval scene hit@1 / hit@5 | 100.0% / 100.0% |
| Valid citations | 100.0% |
| Grounding mean IoU (action set, n=30) | 0.734 |
| Grounding IoU@0.3 / @0.5 | 100.0% / 100.0% |
| Ask latency median / p95 | 1.8 ms / 4.0 ms |
| Corpus generation | 1.5 s |
| Ingest (120 s / 30 scenes) | 23.5 s → 5.1× realtime, 1.28 scenes/s |

The three QA misses are real detector limitations on the fixture (two same-color rectangles merging in one count, one false red-circle presence). `outputs/eval-report.md` and `outputs/eval-report.json` are regenerated by the eval command. Ingest throughput was measured on 384×216 synthetic video with fixture ASR; 720p decoding, real ASR, and VLM/API calls will be slower.

## Real-provider / GPU mode

The local path is the default. Each adapter can be selected without changing the pipeline contract:

| Capability | Local default | Real adapter |
| --- | --- | --- |
| Captioning | classical analyzer | `--captioner gemini` with `GEMINI_API_KEY`; `GEMINI_MODEL` (default `gemini-2.5-pro`), `GEMINI_API_BASE` |
| Synthesis | extractive | `--synth gemini`; requires `GEMINI_API_KEY`; citations are validated/clipped after parsing |
| ASR | fixture sidecar | `uv run --with faster-whisper python -m vidqa ingest ... --asr faster-whisper` (model `turbo`, word timestamps; downloads checkpoint on first use) |
| Vector index | local npz/JSONL | `--qdrant-url http://127.0.0.1:6333 --qdrant-collection vidqa_scenes --recreate`; `ask` accepts `--qdrant-url`; local bundles remain the source for grounding/frame paths |
| Temporal grounding | local scorer | `GROUNDER_URL=http://host/ground --grounder http`; POST `{query, video_id, scene_id, start_ms, end_ms, words}` → `{start_ms, end_ms, score}`; invalid/failed responses fall back locally |

Gemini requests use `POST {GEMINI_API_BASE}/models/{model}:generateContent` with inline JPEG parts and `x-goog-api-key`; the adapter is unit-tested with an injected transport, not against a live key. Qdrant uses the named-vector collection schema and `/points/query`; tests exercise exact request/response bodies through a fake transport. A URL ingest requires `yt-dlp` (`uv run --with yt-dlp ...`) and a real ASR mode or sidecar.

For 100 hours, the design is bundle-per-video and resumable: existing bundles are skipped unless `--force`. Scenes are stored as JSONL and vectors as compressed npz; no global matrix rewrite per video. At 6k–8k scenes the local index fits comfortably in memory; switch to Qdrant when the corpus or write concurrency grows.

## Design decisions

- **RRF with `k=60`**, following Cormack et al.; equal-weight ranks from caption/keyframe/transcript plus an IDF lexical retriever are fused. The lexical ranker is a hybrid-retrieval safeguard: it anchors queries containing a rare term (document frequency ≤ 2) to scenes that contain it, because hashed dense vectors alone misranked “the scene about *keyword*” queries. Its rank is one input among four; no per-retriever weights are hand-tuned.
- **Scene boundary score** is the mean per-pixel HSV delta (`|ΔH| + 255|ΔS| + 255|ΔV|)/3` at 4 fps. On the fixture, cuts score ≈30–46 while in-scene motion scores ≈2–3, so threshold 27 separates them. Boundaries are quantized to the analysis frame.
- **Deterministic embeddings** avoid model downloads and network variance. The tradeoff is semantic range: hashed text vectors and handcrafted visual descriptors cannot match CLIP/VLM embeddings on open-domain video.
- **Named vectors, not a single pooled scene vector**, preserve the three evidence streams; the UI exposes per-stream ranks. Transcript vectors are omitted for silent scenes.
- **Atomic, portable bundles** (`manifest.json`, `scenes.jsonl`, `vectors.npz`) keep re-ingest idempotent and allow the same payload to be synced to Qdrant.
- **Static TypeScript UI** instead of Next.js 15: no framework runtime, no install needed to serve, and the required player/thumbnail/timestamp UX is small. It is still typechecked strictly and built by `verify.sh`.
- **The benchmark is synthetic and programmatically labelled.** Public benchmark loaders accept normalized JSONL (`video_id, question, answer, category?, start_ms?, end_ms?`) and skip unavailable videos; ActivityNet-QA/NeXT-GQA videos are not bundled and no public scores are claimed.

## Tests

`./verify.sh` runs 66 stdlib `unittest` tests plus strict TypeScript checking and a Bun bundle. Coverage includes HSV/scene cuts, blob counting/shape/motion, fixture ASR alignment and missing-sidecar errors, embedding determinism, RRF math and video filtering, lexical/rare-term retrieval, grounding/IoU and HTTP fallback, local/Gemini synthesis with citation validation, Qdrant request bodies, ingest→ask end-to-end with real extracted frames, re-ingest skipping, metric math/report generation, HTTP endpoints, range requests, and path-traversal rejection. Generated media lives in temp directories.

File layout: `vidqa/` pipeline, `web/` UI, `tests/` suite, `outputs/` measured report and skill document, `data/` ignored workdirs.

## Limitations

- The local analyzer understands colors, blob counts, rectangle/circle geometry, and coarse motion only. Complex actions, people, text, speech, and open-domain semantics require the Gemini adapters or a real VLM/CLIP stack.
- ASR is fixture-driven offline; speech content in the synthetic corpus is silent. Real Whisper accuracy/WER is not measured here.
- No public benchmark numbers: ActivityNet-QA and NeXT-GQA require external videos/datasets and are not downloaded in CI. The 97% result is on an easy, programmatically labelled 100-query set.
- 100-hour ingest, GPU throughput, and hosted API cost/latency are not measured. The reported 5.1× realtime is for 384×216 synthetic video with fixture ASR.
- Qdrant and TimeLens adapters are contract-tested but not live-tested here. Gemini requires a key and was not exercised against the live API.
- UI has no authentication and binds to localhost by default; do not expose it publicly.
- Exercises 1–5 (caption-model swap, pooled-embedding ablation, interactive count verification, cost benchmarking, speaker diarization) are intentionally out of scope.

## References

1. Qdrant — [Hybrid and multi-stage queries](https://qdrant.tech/documentation/search/hybrid-queries/): named-vector hybrid retrieval and RRF.
2. PySceneDetect — [Detector documentation](https://www.scenedetect.com/docs/latest/api/detectors.html): content-aware HSV shot detection.
3. SYSTRAN — [faster-whisper](https://github.com/SYSTRAN/faster-whisper): CTranslate2 Whisper and `word_timestamps=True`.
4. Cormack, Clarke, Büttcher — [Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), SIGIR 2009: RRF with `k=60`.
5. Tencent ARC — [TimeLens](https://github.com/TencentARC/TimeLens): temporal-grounding model family and benchmark; the HTTP adapter follows its window-refinement contract.

Source specification: <https://aiengineeringfromscratch.com/learn/capstone-projects/video-understanding-pipeline>.

## License

MIT © 2026 Pedro Monteiro.
