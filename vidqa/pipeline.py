"""End-to-end search and answer orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import VidqaError
from .caption import ProviderError
from .embed import TextEmbedder
from .ground import HTTPGrounder, LocalGrounder
from .index import LocalIndex, reciprocal_rank_fusion
from .models import Answer, Citation, RetrievalHit
from .qdrant import QdrantStore
from .synth import (
    GeminiSynthesizer,
    LocalExtractiveSynthesizer,
    citation_for_scene,
    validate_citations,
)
from .vision import visual_query

RRF_WEIGHTS = {
    "caption": 1.0,
    "keyframe": 0.3,
    "transcript": 1.2,
    "lexical": 4.0,
}


class NoResultsError(VidqaError):
    pass


class AskService:
    def __init__(
        self,
        workdir: str | Path,
        synthesizer: Any | None = None,
        grounder: Any | None = None,
        qdrant: QdrantStore | None = None,
    ) -> None:
        self.workdir = Path(workdir)
        self.index = LocalIndex(self.workdir)
        self.embedder = TextEmbedder()
        self.local_synth = LocalExtractiveSynthesizer()
        self.synthesizer = synthesizer or self.local_synth
        self.grounder = grounder or LocalGrounder(self.embedder)
        self.qdrant = qdrant

    def _search_space(
        self,
        space: str,
        question: str,
        video_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if space == "keyframe":
            vector = visual_query(question)
        else:
            vector = self.embedder.encode(question)
        if self.qdrant is not None:
            return self.qdrant.search(space, vector, limit=limit, video_id=video_id)
        return self.index.search(space, vector, k=limit, video_id=video_id)

    def search(
        self,
        question: str,
        k: int = 5,
        video_id: str | None = None,
    ) -> list[RetrievalHit]:
        limit = max(20, k * 4)
        rankings = {
            space: self._search_space(space, question, video_id, limit)
            for space in ("caption", "keyframe", "transcript")
        }
        rankings["lexical"] = self.index.lexical_search(
            question, k=limit, video_id=video_id
        )
        fused = reciprocal_rank_fusion(
            rankings, k=60, weights=RRF_WEIGHTS, top_k=max(k * 4, 20)
        )
        rare = self.index.rare_query_tokens(question, max_df=2)
        if rare:
            anchored = [
                row
                for row in fused
                if self.index.contains_tokens(row["video_id"], row["scene_id"], rare)
            ]
            if anchored:
                seen = {(row["video_id"], row["scene_id"]) for row in anchored}
                fused = anchored + [
                    row
                    for row in fused
                    if (row["video_id"], row["scene_id"]) not in seen
                ]
        fused = fused[:k]
        hits: list[RetrievalHit] = []
        for row in fused:
            try:
                scene = self.index.scene(row["video_id"], row["scene_id"])
            except VidqaError:
                continue
            hits.append(
                RetrievalHit(
                    scene=scene,
                    score=float(row["rrf_score"]),
                    ranks={k2: int(v) for k2, v in row["ranks"].items()},
                    scores={k2: float(v) for k2, v in row["scores"].items()},
                )
            )
        return hits

    def ask(
        self,
        question: str,
        k: int = 5,
        video_id: str | None = None,
    ) -> Answer:
        question = question.strip()
        if not question:
            raise VidqaError("question is empty")
        hits = self.search(question, k=k, video_id=video_id)
        if not hits:
            raise NoResultsError("no indexed scenes matched the question")
        top = hits[0]
        grounding = self.grounder.ground(
            question,
            top.scene,
            self.index.thumb_vectors(top.scene.video_id, top.scene.scene_id),
        )
        used_fallback = False
        try:
            synthesis = self.synthesizer.synthesize(question, hits, grounding)
        except ProviderError:
            synthesis = self.local_synth.synthesize(question, hits, grounding)
            used_fallback = True
        citations = validate_citations(
            [c.to_dict() for c in synthesis.citations],
            hits,
        )
        if not citations:
            citations = [
                citation_for_scene(top.scene, top.scene.start_ms, top.scene.end_ms)
            ]
        confidence = (
            synthesis.confidence
            if synthesis.confidence is not None
            else top.score / (top.score + 0.05)
        )
        evidence = dict(synthesis.evidence)
        if used_fallback:
            evidence["fallback"] = "provider error; local extractive synthesizer used"
        return Answer(
            question=question,
            answer=synthesis.answer,
            category=synthesis.category,
            citations=[Citation(**c.to_dict()) for c in citations],
            evidence=evidence,
            retrieval=[hit.to_dict() for hit in hits],
            confidence=float(max(0.0, min(1.0, confidence))),
            provider=synthesis.provider,
            grounding=grounding.to_dict(),
        )


def build_grounder(kind: str, url: str | None = None) -> Any:
    if kind == "local":
        return None
    if kind == "http":
        if not url:
            raise VidqaError("--grounder http requires GROUNDER_URL")
        return HTTPGrounder(url)
    raise VidqaError(f"unknown grounder: {kind}")


def build_synthesizer(kind: str) -> Any:
    if kind == "local":
        return None
    if kind == "gemini":
        return GeminiSynthesizer()
    raise VidqaError(f"unknown synthesizer: {kind}")
