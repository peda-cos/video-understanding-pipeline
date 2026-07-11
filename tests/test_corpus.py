import unittest
from collections import Counter

from vidqa.corpus import _build_benchmark, _build_specs


class CorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.video_specs, self.scenes = _build_specs(6, 5, 4000, 384, 216)
        self.rows = _build_benchmark(self.scenes)

    def test_benchmark_composition_is_exactly_100(self) -> None:
        self.assertEqual(len(self.rows), 100)
        counts = Counter(row["category"] for row in self.rows)
        self.assertEqual(counts["counting"], 30)
        self.assertEqual(counts["action"], 30)
        self.assertEqual(counts["descriptive"], 40)

    def test_gold_windows_are_inside_scenes(self) -> None:
        by_key = {(s["video_id"], s["scene_id"]): s for s in self.scenes}
        for row in self.rows:
            scene = by_key[(row["video_id"], row["scene_id"])]
            start, end = row["window_ms"]
            self.assertGreaterEqual(start, scene["start_ms"])
            self.assertLessEqual(end, scene["end_ms"])
            self.assertLess(start, end)

    def test_keywords_are_unique(self) -> None:
        keywords = [scene["keyword"] for scene in self.scenes]
        self.assertEqual(len(keywords), len(set(keywords)))

    def test_action_queries_cover_yes_and_no(self) -> None:
        actions = [row for row in self.rows if row["category"] == "action"]
        self.assertTrue(any(row["answer"] == "yes" for row in actions))
        self.assertTrue(any(row["answer"] == "no" for row in actions))

    def test_specs_are_deterministic(self) -> None:
        again, _ = _build_specs(6, 5, 4000, 384, 216)
        self.assertEqual(again, self.video_specs)


if __name__ == "__main__":
    unittest.main()
