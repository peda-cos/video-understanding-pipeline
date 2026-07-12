import json
import unittest

from vidqa.ground import Grounding
from vidqa.models import RetrievalHit, Scene, Thumb
from vidqa.caption import ProviderError
from vidqa.synth import (
    GeminiSynthesizer,
    LocalExtractiveSynthesizer,
    classify_question,
    validate_citations,
)


def make_scene() -> Scene:
    return Scene(
        video_id="v001",
        scene_id=1,
        start_ms=0,
        end_ms=4000,
        keyframe_path="",
        keyframe_url="",
        thumbs=[
            Thumb(500, "a.jpg", "/a.jpg"),
            Thumb(2000, "b.jpg", "/b.jpg"),
            Thumb(3500, "c.jpg", "/c.jpg"),
        ],
        caption="A medium scene with blue background. Visible: 2 red rectangles.",
        transcript="the orchard scene shows a red rectangle that moves left to right",
        words=[],
        facts={
            "background_colors": ["blue"],
            "object_counts": {"red rectangle": 2},
            "tracks": [
                {
                    "color": "red",
                    "shape": "rectangle",
                    "direction": "left-to-right",
                    "displacement": 0.6,
                    "start_t_ms": 500,
                    "end_t_ms": 3500,
                }
            ],
        },
    )


def hit(scene: Scene) -> RetrievalHit:
    return RetrievalHit(scene=scene, score=0.03, ranks={"caption": 1}, scores={"caption": 0.8})


class LocalSynthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = make_scene()
        self.hits = [hit(self.scene)]
        self.synth = LocalExtractiveSynthesizer()
        self.grounding = Grounding(1000, 2500, 0.5)

    def test_counting(self) -> None:
        result = self.synth.synthesize(
            "How many red rectangles are visible in the scene about orchard?",
            self.hits,
            self.grounding,
        )
        self.assertEqual(result.category, "counting")
        self.assertTrue(result.answer.startswith("2 "))
        self.assertEqual(result.citations[0].start_ms, 0)
        self.assertEqual(result.citations[0].end_ms, 4000)

    def test_action_yes_and_no(self) -> None:
        yes = self.synth.synthesize(
            "Does the red rectangle move from left to right in the scene about orchard?",
            self.hits,
            self.grounding,
        )
        self.assertEqual(yes.category, "action")
        self.assertTrue(yes.answer.startswith("Yes"))
        self.assertEqual((yes.citations[0].start_ms, yes.citations[0].end_ms), (1000, 2500))
        no = self.synth.synthesize(
            "Does the red rectangle move from right to left in the scene about orchard?",
            self.hits,
            self.grounding,
        )
        self.assertTrue(no.answer.startswith("No"))

    def test_presence_and_background(self) -> None:
        present = self.synth.synthesize(
            "Is there a red rectangle in the scene about orchard?",
            self.hits,
            self.grounding,
        )
        self.assertTrue(present.answer.startswith("Yes"))
        absent = self.synth.synthesize(
            "Is there a green circle in the scene about orchard?",
            self.hits,
            self.grounding,
        )
        self.assertTrue(absent.answer.startswith("No"))
        background = self.synth.synthesize(
            "What color dominates the background of the scene about orchard?",
            self.hits,
            self.grounding,
        )
        self.assertIn("blue", background.answer)

    def test_classification(self) -> None:
        self.assertEqual(classify_question("How many red cars?"), "counting")
        self.assertEqual(
            classify_question("Does the red rectangle move left to right?"), "action"
        )
        self.assertEqual(classify_question("What color is the background?"), "descriptive")

    def test_citation_validation_clips_and_drops_unknown(self) -> None:
        citations = validate_citations(
            [
                {"video_id": "v001", "scene_id": 1, "start_ms": -50, "end_ms": 99999},
                {"video_id": "missing", "scene_id": 1, "start_ms": 0, "end_ms": 10},
            ],
            self.hits,
        )
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].start_ms, 0)
        self.assertEqual(citations[0].end_ms, 4000)
        self.assertTrue(citations[0].frames)


class FakeGeminiClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.parts = None

    def generate(self, parts):  # noqa: ANN001
        self.parts = parts
        return "```json\n" + json.dumps(self.payload) + "\n```"


class GeminiSynthTests(unittest.TestCase):
    def test_parses_answer_and_citations(self) -> None:
        scene = make_scene()
        client = FakeGeminiClient(
            {
                "answer": "2 red rectangles are visible.",
                "category": "counting",
                "citations": [
                    {"video_id": "v001", "scene_id": 1, "start_ms": 0, "end_ms": 4000}
                ],
                "confidence": 0.8,
            }
        )
        synth = GeminiSynthesizer(client=client, asset_root=".")
        result = synth.synthesize("How many red rectangles?", [hit(scene)], Grounding(0, 4000, 0.1))
        self.assertEqual(result.answer, "2 red rectangles are visible.")
        self.assertEqual(result.citations[0].end_ms, 4000)
        self.assertEqual(result.provider, "gemini")
        self.assertIn("Question:", client.parts[0]["text"])

    def test_invalid_citation_is_rejected(self) -> None:
        scene = make_scene()
        client = FakeGeminiClient(
            {
                "answer": "Yes.",
                "category": "action",
                "citations": [{"video_id": "nope", "scene_id": 9, "start_ms": 0, "end_ms": 1}],
            }
        )
        with self.assertRaises(ProviderError):
            GeminiSynthesizer(client=client).synthesize(
                "Does it move?", [hit(scene)], Grounding(0, 4000, 0.1)
            )


if __name__ == "__main__":
    unittest.main()
