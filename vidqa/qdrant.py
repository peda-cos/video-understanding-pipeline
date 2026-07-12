"""Qdrant REST adapter for the same named multi-vector scene schema."""

from __future__ import annotations

import json
import uuid
from typing import Any

import numpy as np

from . import VidqaError
from .caption import Transport, default_transport
from .index import LocalIndex

VECTOR_SIZES = {"caption": 384, "keyframe": 80, "transcript": 384}


class QdrantError(VidqaError):
    pass


class QdrantStore:
    def __init__(
        self,
        url: str,
        collection: str = "vidqa_scenes",
        api_key: str | None = None,
        transport: Transport | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not url.strip():
            raise QdrantError("Qdrant URL is empty")
        self.url = url.rstrip("/")
        self.collection = collection
        self.api_key = api_key
        self.transport = transport or default_transport
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        allow: tuple[int, ...] = (200, 201),
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        status, text = self.transport(
            method,
            f"{self.url}{path}",
            headers,
            body or {},
            self.timeout,
        )
        if status not in allow:
            raise QdrantError(f"Qdrant HTTP {status} for {method} {path}: {text[:400]}")
        if not text.strip():
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise QdrantError(f"Qdrant returned invalid JSON for {path}") from exc
        if not isinstance(parsed, dict):
            raise QdrantError(f"Qdrant returned unexpected payload for {path}")
        return parsed

    def ensure_collection(self, recreate: bool = False) -> None:
        path = f"/collections/{self.collection}"
        if recreate:
            self._request("DELETE", path, allow=(200, 201, 404))
        body = {
            "vectors": {
                name: {"size": size, "distance": "Cosine"}
                for name, size in VECTOR_SIZES.items()
            }
        }
        self._request("PUT", path, body, allow=(200, 201, 409))

    def upsert_scene(
        self,
        payload: dict[str, Any],
        vectors: dict[str, np.ndarray],
    ) -> None:
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"vidqa:{payload['video_id']}:{payload['scene_id']}",
            )
        )
        vector_payload: dict[str, list[float]] = {}
        for space, size in VECTOR_SIZES.items():
            vector = np.asarray(vectors[space], dtype=np.float32)
            if vector.size != size:
                raise QdrantError(f"{space} vector has {vector.size} dims, expected {size}")
            if space == "transcript" and float(np.linalg.norm(vector)) == 0.0:
                continue
            vector_payload[space] = vector.tolist()
        body = {
            "points": [
                {
                    "id": point_id,
                    "vector": vector_payload,
                    "payload": payload,
                }
            ]
        }
        self._request(
            "PUT",
            f"/collections/{self.collection}/points?wait=true",
            body,
            allow=(200, 201),
        )

    def sync(self, index: LocalIndex) -> int:
        count = 0
        for payload, vectors in index.points():
            self.upsert_scene(payload, vectors)
            count += 1
        return count

    def search(
        self,
        space: str,
        vector: np.ndarray,
        limit: int = 20,
        video_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if space not in VECTOR_SIZES:
            raise QdrantError(f"unknown vector space: {space}")
        body: dict[str, Any] = {
            "query": np.asarray(vector, dtype=np.float32).tolist(),
            "using": space,
            "limit": limit,
            "with_payload": True,
        }
        if video_id is not None:
            body["filter"] = {
                "must": [{"key": "video_id", "match": {"value": video_id}}]
            }
        payload = self._request(
            "POST",
            f"/collections/{self.collection}/points/query",
            body,
        )
        result = payload.get("result", {})
        if isinstance(result, dict):
            points = result.get("points", [])
        elif isinstance(result, list):
            points = result
        else:
            raise QdrantError("unexpected Qdrant query result")
        hits: list[dict[str, Any]] = []
        for point in points:
            scene_payload = point.get("payload") or {}
            if "video_id" not in scene_payload or "scene_id" not in scene_payload:
                continue
            hits.append(
                {
                    "video_id": str(scene_payload["video_id"]),
                    "scene_id": int(scene_payload["scene_id"]),
                    "score": float(point.get("score", 0.0)),
                }
            )
        return hits
