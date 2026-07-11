"""Local scene analysis wrapper used by ingest."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import vision


class LocalAnalyzer:
    def analyze(
        self, frames: list[np.ndarray], times_ms: list[int]
    ) -> tuple[str, dict[str, Any]]:
        facts = vision.analyze_scene(frames, times_ms)
        return str(facts["caption"]), facts
