import json
import unittest

import numpy as np

from vidqa.ground import HTTPGrounder, LocalGrounder, iou
from vidqa.index import LocalIndex
from vidqa.models import Scene, Thumb
from tests.helpers import demo_paths


class IoUTests(unittest.TestCase):
    def test_iou(self) -> None:
        self.assertAlmostEqual(iou(0, 100, 0, 100), 1.0)
        self.assertAlmostEqual(iou(0, 100, 50, 150), 1 / 3)
        self.assertEqual(iou(0, 100, 100, 200), 0.0)
        self.assertEqual(iou(0, 0, 0, 0), 0.0)


class GroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _corpus, index_path, _benchmark = demo_paths()
        cls.index = LocalIndex(index_path)
        cls.grounder = LocalGrounder()

    def test_window_stays_inside_scene_and_finds_action_words(self) -> None:
        scene = self.index.scene("v001", 1)
        thumbs = self.index.thumb_vectors("v001", 1)
        result = self.grounder.ground(
            "where does the red rectangle move left to right",
            scene,
            thumbs,
        )
        self.assertGreaterEqual(result.start_ms, scene.start_ms)
        self.assertLessEqual(result.end_ms, scene.end_ms)
        self.assertLess(result.start_ms, result.end_ms)
        self.assertGreater(result.score, 0.0)
        self.assertGreater(iou(result.start_ms, result.end_ms, 500, 1500), 0.2)

    def test_no_words_falls_back_to_scene(self) -> None:
        scene = Scene(
            video_id="v",
            scene_id=1,
            start_ms=0,
            end_ms=4000,
            thumbs=[Thumb(500, "a.jpg", "/a.jpg")],
        )
        result = self.grounder.ground("anything", scene, np.zeros((1, 80), dtype=np.float32))
        self.assertEqual((result.start_ms, result.end_ms), (0, 4000))

    def test_http_grounder_parses_and_validates(self) -> None:
        scene = self.index.scene("v001", 1)
        thumbs = self.index.thumb_vectors("v001", 1)

        def transport(method, url, headers, body, timeout):  # noqa: ANN001
            return 200, json.dumps({"start_ms": 700, "end_ms": 1500, "score": 0.9})

        result = HTTPGrounder("http://grounder.test", transport=transport).ground(
            "question", scene, thumbs
        )
        self.assertEqual(result.method, "http")
        self.assertEqual((result.start_ms, result.end_ms), (700, 1500))

    def test_http_grounder_falls_back_on_invalid_window(self) -> None:
        scene = self.index.scene("v001", 1)
        thumbs = self.index.thumb_vectors("v001", 1)

        def transport(method, url, headers, body, timeout):  # noqa: ANN001
            return 200, json.dumps({"start_ms": 5000, "end_ms": 6000})

        result = HTTPGrounder("http://grounder.test", transport=transport).ground(
            "question", scene, thumbs
        )
        self.assertEqual(result.method, "local")


if __name__ == "__main__":
    unittest.main()
