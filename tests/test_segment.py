import tempfile
import unittest
from pathlib import Path

from vidqa.corpus import build_corpus
from vidqa.segment import detect_scenes
from tests.helpers import demo_paths


class SegmentTests(unittest.TestCase):
    def test_detects_two_hard_cuts(self) -> None:
        corpus, _index, _benchmark = demo_paths()
        scenes = detect_scenes(str(corpus / "media" / "v001.mp4"))
        self.assertEqual(len(scenes), 2)
        self.assertLessEqual(abs(scenes[0][1] - 2000), 400)
        self.assertEqual(scenes[-1][1], 4000)
        self.assertEqual(scenes[0][0], 0)

    def test_single_scene_is_not_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "one"
            build_corpus(out, videos=1, scenes_per_video=1, scene_ms=2000, fps=8)
            scenes = detect_scenes(str(out / "media" / "v001.mp4"))
            self.assertEqual(scenes, [(0, 2000)])


if __name__ == "__main__":
    unittest.main()
