import tempfile
import unittest
from pathlib import Path

from vidqa import media
from tests.helpers import demo_paths


class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _corpus, cls.index_path, _benchmark = demo_paths()
        cls.source = cls.index_path / "videos" / "v001" / "source.mp4"

    def test_probe(self) -> None:
        info = media.probe(self.source)
        self.assertEqual((info["width"], info["height"]), (384, 216))
        self.assertEqual(info["duration_ms"], 4000)
        self.assertAlmostEqual(info["fps"], 8.0, places=2)
        self.assertTrue(info["has_audio"])

    def test_frame_stream(self) -> None:
        frames = list(media.iter_frames(self.source, fps=4, width=96, height=54))
        self.assertGreaterEqual(len(frames), 14)
        self.assertEqual(frames[0].shape, (54, 96, 3))

    def test_extract_and_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.jpg"
            media.extract_jpeg(self.source, 1000, path)
            self.assertTrue(path.exists())
            image = media.decode_image(path, width=192)
            self.assertEqual(image.shape, (108, 192, 3))

    def test_to_720p_copies_small_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "copy.mp4"
            media.to_720p(self.source, destination)
            self.assertEqual(destination.read_bytes(), self.source.read_bytes())

    def test_missing_media_is_explicit(self) -> None:
        with self.assertRaises(media.MediaError):
            media.probe("/tmp/definitely-not-a-video.mp4")


if __name__ == "__main__":
    unittest.main()
