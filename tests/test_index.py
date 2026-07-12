import unittest

import numpy as np

from vidqa.embed import TextEmbedder
from vidqa.index import LocalIndex, reciprocal_rank_fusion
from tests.helpers import demo_paths


def item(video: str, scene: int, score: float = 1.0) -> dict:
    return {"video_id": video, "scene_id": scene, "score": score}


class RrfTests(unittest.TestCase):
    def test_expected_order_and_formula(self) -> None:
        rankings = {
            "caption": [item("v", 1), item("v", 2)],
            "transcript": [item("v", 1), item("v", 3)],
            "keyframe": [item("v", 3)],
        }
        fused = reciprocal_rank_fusion(rankings, k=60, top_k=3)
        self.assertEqual([(row["video_id"], row["scene_id"]) for row in fused], [("v", 1), ("v", 3), ("v", 2)])
        self.assertAlmostEqual(fused[0]["rrf_score"], 2 / 61)
        self.assertAlmostEqual(fused[1]["rrf_score"], 1 / 62 + 1 / 61)
        self.assertEqual(fused[0]["ranks"], {"caption": 1, "transcript": 1})

    def test_weights_and_top_k(self) -> None:
        rankings = {"caption": [item("a", 1)], "transcript": [item("b", 1)]}
        fused = reciprocal_rank_fusion(
            rankings, k=60, weights={"caption": 3.0, "transcript": 1.0}, top_k=1
        )
        self.assertEqual(fused[0]["video_id"], "a")
        self.assertAlmostEqual(fused[0]["rrf_score"], 3 / 61)


class LocalIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _corpus, cls.index_path, _benchmark = demo_paths()
        cls.index = LocalIndex(cls.index_path)

    def test_loads_every_scene_and_space(self) -> None:
        self.assertEqual(self.index.size, 4)
        for space, dim in (("caption", 384), ("keyframe", 80), ("transcript", 384)):
            self.assertEqual(self.index.arrays[space].shape, (4, dim))
        self.assertEqual(self.index.thumb_arrays.shape, (4, 3, 80))
        self.assertEqual(len(self.index.videos()), 2)

    def test_search_and_video_filter(self) -> None:
        embedder = TextEmbedder()
        vector = embedder.encode("orchard")
        hits = self.index.search("transcript", vector, k=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["video_id"], "v001")
        self.assertEqual(hits[0]["scene_id"], 1)
        filtered = self.index.search("transcript", vector, k=5, video_id="v002")
        self.assertTrue(all(hit["video_id"] == "v002" for hit in filtered))

    def test_scene_loads_words_and_thumbs(self) -> None:
        scene = self.index.scene("v001", 1)
        self.assertTrue(scene.words)
        self.assertEqual(len(scene.thumbs), 3)
        self.assertIn("orchard", scene.transcript)
        self.assertEqual(self.index.thumb_vectors("v001", 1).shape, (3, 80))

    def test_points_include_named_vectors(self) -> None:
        payload, vectors = next(self.index.points())
        self.assertIn("video_id", payload)
        self.assertEqual(set(vectors), {"caption", "keyframe", "transcript"})
        self.assertEqual(float(np.linalg.norm(vectors["keyframe"])), 1.0)

    def test_reload_from_disk_is_stable(self) -> None:
        second = LocalIndex(self.index_path)
        self.assertEqual(second.size, self.index.size)
        hits_a = self.index.search("caption", self.index.arrays["caption"][0], k=3)
        hits_b = second.search("caption", second.arrays["caption"][0], k=3)
        self.assertEqual(hits_a, hits_b)


if __name__ == "__main__":
    unittest.main()
