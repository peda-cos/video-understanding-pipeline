import json
import tempfile
import unittest
from pathlib import Path

from vidqa import asr
from vidqa.models import Word


class AsrTests(unittest.TestCase):
    def test_slice_words_by_scene_midpoint_and_clipping(self) -> None:
        words = [
            Word("one", 100, 300),
            Word("two", 400, 650),
            Word("boundary", 950, 1100),
            Word("three", 1500, 1700),
        ]
        buckets = asr.slice_words_by_scenes(words, [(0, 1000), (1000, 2000)])
        self.assertEqual([w.text for w in buckets[0]], ["one", "two"])
        self.assertEqual([w.text for w in buckets[1]], ["boundary", "three"])
        self.assertEqual(buckets[1][0].start_ms, 1000)
        self.assertEqual(buckets[1][0].end_ms, 1100)

    def test_fixture_transcriber_reads_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "clip.mp4"
            media.write_bytes(b"not real media")
            payload = {
                "language": "en",
                "words": [
                    {"word": "hello", "start_ms": 0, "end_ms": 200},
                    {"word": "world", "start_ms": 250, "end_ms": 500},
                ],
            }
            (root / "clip.words.json").write_text(json.dumps(payload), encoding="utf-8")
            words = asr.FixtureTranscriber().transcribe(media)
            self.assertEqual([w.text for w in words], ["hello", "world"])

    def test_fixture_transcriber_accepts_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "clip.mp4"
            media.write_bytes(b"x")
            (root / "clip.words.json").write_text(
                json.dumps({"words": [{"word": "ok", "start": 0.5, "end": 1.25}]}),
                encoding="utf-8",
            )
            words = asr.FixtureTranscriber().transcribe(media)
            self.assertEqual(words[0].start_ms, 500)
            self.assertEqual(words[0].end_ms, 1250)

    def test_missing_sidecar_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "clip.mp4"
            media.write_bytes(b"x")
            with self.assertRaises(asr.TranscriptUnavailable) as ctx:
                asr.FixtureTranscriber(Path(tmp) / "elsewhere").transcribe(media)
            self.assertIn("no word-timestamp transcript", str(ctx.exception))

    def test_transcript_text(self) -> None:
        self.assertEqual(
            asr.transcript_text([Word(" one ", 0, 1), Word("two", 1, 2)]),
            "one two",
        )


if __name__ == "__main__":
    unittest.main()
