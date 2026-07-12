"""Local persisted multi-vector index with reciprocal rank fusion."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import VidqaError
from .embed import content_tokens
from .models import Scene

SPACES = ("caption", "keyframe", "transcript")


class IndexLoadError(VidqaError):
    pass


def reciprocal_rank_fusion(
    rankings: dict[str, list[dict[str, Any]]],
    k: int = 60,
    weights: dict[str, float] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Fuse ranked lists of {video_id, scene_id, score} by 1/(k+rank)."""
    weights = weights or {}
    fused: dict[tuple[str, int], float] = defaultdict(float)
    ranks: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    raw_scores: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    payloads: dict[tuple[str, int], dict[str, Any]] = {}
    for space, items in rankings.items():
        weight = float(weights.get(space, 1.0))
        for rank, item in enumerate(items, start=1):
            key = (str(item["video_id"]), int(item["scene_id"]))
            fused[key] += weight / (k + rank)
            ranks[key][space] = rank
            raw_scores[key][space] = float(item.get("score", 0.0))
            payloads[key] = item
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    results: list[dict[str, Any]] = []
    for key, score in ordered[:top_k]:
        item = dict(payloads[key])
        item["video_id"], item["scene_id"] = key
        item["rrf_score"] = score
        item["ranks"] = ranks[key]
        item["scores"] = raw_scores[key]
        results.append(item)
    return results


class LocalIndex:
    def __init__(self, workdir: str | Path) -> None:
        self.workdir = Path(workdir)
        self.payloads: list[dict[str, Any]] = []
        self.arrays: dict[str, np.ndarray] = {}
        self.thumb_arrays: np.ndarray = np.zeros((0, 3, 80), dtype=np.float32)
        self.transcript_present: np.ndarray = np.zeros((0,), dtype=bool)
        self._doc_tokens: list[set[str]] = []
        self._idf: dict[str, float] = {}
        self._df: dict[str, int] = {}
        self._scene_cache: dict[str, dict[int, Scene]] = {}
        self._video_ids: list[str] = []
        self.reload()

    def reload(self) -> None:
        videos_dir = self.workdir / "videos"
        payloads: list[dict[str, Any]] = []
        matrices: dict[str, list[np.ndarray]] = {space: [] for space in SPACES}
        present: list[bool] = []
        thumbs: list[np.ndarray] = []
        if videos_dir.exists():
            for bundle in sorted(videos_dir.iterdir()):
                if not bundle.is_dir():
                    continue
                scenes_path = bundle / "scenes.jsonl"
                vectors_path = bundle / "vectors.npz"
                if not scenes_path.exists() or not vectors_path.exists():
                    raise IndexLoadError(f"incomplete index bundle: {bundle}")
                rows = [
                    json.loads(line)
                    for line in scenes_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                with np.load(vectors_path) as data:
                    arrays = {
                        space: np.asarray(data[space], dtype=np.float32)
                        for space in SPACES
                    }
                    present_array = np.asarray(data["transcript_present"], dtype=bool)
                    thumb_array = np.asarray(data["thumbs"], dtype=np.float32)
                if len(rows) != arrays["caption"].shape[0]:
                    raise IndexLoadError(f"scene/vector count mismatch in {bundle}")
                payloads.extend(rows)
                for space in SPACES:
                    matrices[space].append(arrays[space])
                present.extend(present_array.tolist())
                thumbs.append(thumb_array)
        self.payloads = payloads
        self._video_ids = [str(row["video_id"]) for row in payloads]
        self._scene_cache = {}
        dims = {"caption": 384, "keyframe": 80, "transcript": 384}
        self.arrays = {
            space: (
                np.concatenate(matrices[space], axis=0)
                if matrices[space]
                else np.zeros((0, dims[space]), dtype=np.float32)
            )
            for space in SPACES
        }
        self.thumb_arrays = (
            np.concatenate(thumbs, axis=0)
            if thumbs
            else np.zeros((0, 3, 80), dtype=np.float32)
        )
        self.transcript_present = np.array(present, dtype=bool)
        self._doc_tokens = [
            set(
                content_tokens(
                    str(row.get("caption", "")) + " " + str(row.get("transcript", ""))
                )
            )
            for row in payloads
        ]
        document_frequency: dict[str, int] = defaultdict(int)
        for tokens in self._doc_tokens:
            for token in tokens:
                document_frequency[token] += 1
        self._df = dict(document_frequency)
        total = max(1, len(payloads))
        self._idf = {
            token: math.log((total + 1) / (count + 1)) + 1.0
            for token, count in document_frequency.items()
        }

    @property
    def size(self) -> int:
        return len(self.payloads)

    def videos(self) -> list[dict[str, Any]]:
        path = self.workdir / "catalog.json"
        if not path.exists():
            return []
        return list(json.loads(path.read_text(encoding="utf-8")).get("videos", []))

    def payload(self, video_id: str, scene_id: int) -> dict[str, Any] | None:
        for row in self.payloads:
            if row["video_id"] == video_id and int(row["scene_id"]) == scene_id:
                return row
        return None

    def payloads_for_video(self, video_id: str) -> list[dict[str, Any]]:
        return [row for row in self.payloads if row["video_id"] == video_id]

    def scene(self, video_id: str, scene_id: int) -> Scene:
        if video_id not in self._scene_cache:
            manifest = self.workdir / "videos" / video_id / "manifest.json"
            if not manifest.exists():
                raise IndexLoadError(f"missing manifest for {video_id}")
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self._scene_cache[video_id] = {
                int(row["scene_id"]): Scene.from_dict(row)
                for row in data.get("scenes", [])
            }
        scene = self._scene_cache[video_id].get(int(scene_id))
        if scene is None:
            raise IndexLoadError(f"scene {video_id}:{scene_id} not found")
        return scene

    def row_index(self, video_id: str, scene_id: int) -> int:
        for index, row in enumerate(self.payloads):
            if row["video_id"] == video_id and int(row["scene_id"]) == scene_id:
                return index
        raise IndexLoadError(f"scene {video_id}:{scene_id} not indexed")

    def thumb_vectors(self, video_id: str, scene_id: int) -> np.ndarray:
        return self.thumb_arrays[self.row_index(video_id, scene_id)]

    def lexical_search(
        self,
        question: str,
        k: int = 20,
        video_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query_tokens = set(content_tokens(question))
        if not query_tokens:
            return []
        scored: list[tuple[float, int]] = []
        for index, tokens in enumerate(self._doc_tokens):
            row = self.payloads[index]
            if video_id is not None and row["video_id"] != video_id:
                continue
            overlap = query_tokens & tokens
            if not overlap:
                continue
            score = sum(self._idf.get(token, 1.0) for token in overlap)
            scored.append((score, index))
        scored.sort(
            key=lambda item: (
                -item[0],
                str(self.payloads[item[1]]["video_id"]),
                int(self.payloads[item[1]]["scene_id"]),
            )
        )
        results: list[dict[str, Any]] = []
        for score, index in scored[:k]:
            row = self.payloads[index]
            results.append(
                {
                    "video_id": str(row["video_id"]),
                    "scene_id": int(row["scene_id"]),
                    "score": float(score),
                }
            )
        return results

    def rare_query_tokens(self, question: str, max_df: int = 2) -> list[str]:
        tokens = set(content_tokens(question))
        return sorted(
            token
            for token in tokens
            if 0 < self._df.get(token, 0) <= max_df
        )

    def contains_tokens(self, video_id: str, scene_id: int, tokens: list[str]) -> bool:
        if not tokens:
            return True
        index = self.row_index(video_id, scene_id)
        return all(token in self._doc_tokens[index] for token in tokens)

    def search(
        self,
        space: str,
        query_vector: np.ndarray,
        k: int = 20,
        video_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if space not in self.arrays:
            raise IndexLoadError(f"unknown vector space: {space}")
        matrix = self.arrays[space]
        if matrix.shape[0] == 0:
            return []
        if query_vector.size != matrix.shape[1]:
            raise IndexLoadError(
                f"vector dimension mismatch for {space}: {query_vector.size} != {matrix.shape[1]}"
            )
        query = np.asarray(query_vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        query = query / norm
        norms = np.linalg.norm(matrix, axis=1)
        scores = np.zeros(matrix.shape[0], dtype=np.float32)
        valid = norms > 0
        scores[valid] = (matrix[valid] @ query) / norms[valid]
        if video_id is not None:
            candidate_indices = [
                i for i, vid in enumerate(self._video_ids) if vid == video_id
            ]
        else:
            candidate_indices = list(range(matrix.shape[0]))
        eligible: list[tuple[int, float]] = []
        for index in candidate_indices:
            if space == "transcript" and not bool(self.transcript_present[index]):
                continue
            if scores[index] <= 0:
                continue
            eligible.append((index, float(scores[index])))
        eligible.sort(
            key=lambda item: (
                -item[1],
                str(self.payloads[item[0]]["video_id"]),
                int(self.payloads[item[0]]["scene_id"]),
            )
        )
        results: list[dict[str, Any]] = []
        for index, score in eligible[:k]:
            row = self.payloads[index]
            results.append(
                {
                    "video_id": str(row["video_id"]),
                    "scene_id": int(row["scene_id"]),
                    "score": score,
                }
            )
        return results

    def points(self) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
        for index, payload in enumerate(self.payloads):
            vectors = {space: self.arrays[space][index] for space in SPACES}
            yield payload, vectors

    def stats(self) -> dict[str, Any]:
        return {"videos": len(self.videos()), "scenes": self.size, "spaces": list(SPACES)}
