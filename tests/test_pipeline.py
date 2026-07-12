import unittest
from pathlib import Path

from vidqa.ingest import ingest_one
from tests.helpers import demo_paths, service


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus, cls.index_path, cls.benchmark = demo_paths()
        cls.service = service()

    def test_counting_question_is_grounded_and_cited(self) -> None:
        answer = self.service.ask(
            "How many green rectangles are visible in the scene about orchard?"
        )
        self.assertEqual(answer.category, "counting")
        self.assertTrue(answer.answer.startswith("2 "))
        self.assertEqual(len(answer.citations), 1)
        citation = answer.citations[0]
        self.assertEqual((citation.video_id, citation.scene_id), ("v001", 1))
        self.assertTrue(citation.frames)
        for frame in citation.frames:
            self.assertTrue((Path(self.index_path) / frame["path"]).exists())
        self.assertTrue(answer.retrieval)

    def test_action_question_uses_refined_window(self) -> None:
        answer = self.service.ask(
            "Does the red rectangle move from left to right in the scene about orchard?"
        )
        self.assertEqual(answer.category, "action")
        self.assertTrue(answer.answer.startswith("Yes"))
        grounding = answer.grounding
        self.assertGreaterEqual(grounding["start_ms"], 0)
        self.assertLessEqual(grounding["end_ms"], 4000)
        self.assertEqual(answer.citations[0].start_ms, grounding["start_ms"])
        self.assertEqual(answer.citations[0].end_ms, grounding["end_ms"])

    def test_retrieval_uses_unique_transcript_keyword(self) -> None:
        hits = self.service.search("meadow")
        self.assertTrue(hits)
        self.assertEqual((hits[0].scene.video_id, hits[0].scene.scene_id), ("v002", 2))
        self.assertIn("meadow", hits[0].scene.transcript)

    def test_reingest_without_force_skips(self) -> None:
        result = ingest_one(
            str(self.corpus / "media" / "v001.mp4"),
            workdir=self.index_path,
            transcripts_dir=self.corpus / "transcripts",
        )
        self.assertTrue(result.skipped)
        self.assertEqual(result.scene_count, 2)

    def test_provider_failure_falls_back_to_local(self) -> None:
        from vidqa.caption import ProviderError
        from vidqa.pipeline import AskService

        class FailingSynthesizer:
            provider = "gemini"

            def synthesize(self, question, hits, grounding):  # noqa: ANN001
                raise ProviderError("fake provider failure")

        fallback = AskService(self.index_path, synthesizer=FailingSynthesizer())
        answer = fallback.ask("How many green rectangles are visible in the scene about orchard?")
        self.assertEqual(answer.provider, "local")
        self.assertIn("fallback", answer.evidence)
        self.assertTrue(answer.answer.startswith("2 "))

    def test_retrieval_ranks_are_present(self) -> None:
        answer = self.service.ask("orchard red rectangle")
        ranks = answer.retrieval[0]["ranks"]
        self.assertTrue(ranks)
        self.assertTrue(set(ranks).issubset({"caption", "keyframe", "transcript", "lexical"}))
        self.assertIn("lexical", ranks)

    def test_rare_keyword_anchors_across_videos(self) -> None:
        hits = self.service.search("lantern", k=1)
        self.assertTrue(hits)
        self.assertEqual((hits[0].scene.video_id, hits[0].scene.scene_id), ("v001", 2))
        self.assertIn("lantern", hits[0].scene.transcript)


if __name__ == "__main__":
    unittest.main()
