import json
import tempfile
import unittest
from pathlib import Path

from vidqa.evaluate import answers_match, load_suite, run_eval
from vidqa.models import Answer, Citation


SCENE = {
    "video_id": "v001",
    "scene_id": 1,
    "start_ms": 0,
    "end_ms": 4000,
    "caption": "A scene.",
    "transcript": "orchard",
    "keyframe_url": "/media/a.jpg",
    "thumbs": [{"t_ms": 1000, "path": "a.jpg", "url": "/media/a.jpg"}],
}

FRAME = [{"t_ms": 1000, "path": "a.jpg", "url": "/media/a.jpg"}]


def make_answer(question: str, text: str, category: str, start: int, end: int) -> Answer:
    retrieval = [{**SCENE, "score": 0.03, "ranks": {"caption": 1}, "scores": {"caption": 0.9}}]
    return Answer(
        question=question,
        answer=text,
        category=category,
        citations=[Citation("v001", 1, start, end, FRAME)],
        evidence={},
        retrieval=retrieval,
        confidence=0.5,
        provider="test",
        grounding={"start_ms": start, "end_ms": end, "method": "local"},
    )


class DummyService:
    def __init__(self) -> None:
        self.answers = {
            "two?": make_answer("two?", "2 red rectangles", "counting", 0, 4000),
            "yes?": make_answer("yes?", "Yes, it moves.", "action", 1000, 2000),
            "wrong?": make_answer("wrong?", "green", "descriptive", 0, 4000),
        }

    def ask(self, question: str, video_id=None):  # noqa: ANN001
        if question == "missing?":
            raise RuntimeError("not indexed")
        return self.answers[question]


class EvalTests(unittest.TestCase):
    def test_answer_matching(self) -> None:
        self.assertTrue(answers_match("2 red rectangles are visible.", "2"))
        self.assertFalse(answers_match("3 red rectangles", "2"))
        self.assertTrue(answers_match("Yes, it moves", "yes"))
        self.assertTrue(answers_match("No, it stays still.", "no"))
        self.assertTrue(answers_match("The background is blue.", "blue"))
        self.assertFalse(answers_match("red", "blue"))

    def test_metrics_and_report(self) -> None:
        rows = [
            {
                "query_id": "q1",
                "question": "two?",
                "category": "counting",
                "video_id": "v001",
                "scene_id": 1,
                "answer": "2",
                "window_ms": [0, 4000],
            },
            {
                "query_id": "q2",
                "question": "yes?",
                "category": "action",
                "video_id": "v001",
                "scene_id": 1,
                "answer": "yes",
                "window_ms": [1000, 2000],
            },
            {
                "query_id": "q3",
                "question": "wrong?",
                "category": "descriptive",
                "video_id": "v001",
                "scene_id": 1,
                "answer": "blue",
                "window_ms": [0, 4000],
            },
            {
                "query_id": "q4",
                "question": "missing?",
                "category": "counting",
                "video_id": "v999",
                "scene_id": 1,
                "answer": "1",
                "window_ms": [0, 1000],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp) / "suite.jsonl"
            suite.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = Path(tmp) / "report.md"
            json_out = Path(tmp) / "report.json"
            result = run_eval(
                DummyService(),
                suite,
                report_path=report,
                json_path=json_out,
            )
            self.assertEqual(load_suite(suite)[0]["category"], "counting")
            metrics = result["metrics"]
            self.assertEqual(metrics["queries_evaluated"], 3)
            self.assertEqual(metrics["queries_skipped"], 1)
            self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
            self.assertAlmostEqual(metrics["per_category"]["counting"]["accuracy"], 1.0)
            self.assertAlmostEqual(metrics["per_category"]["descriptive"]["accuracy"], 0.0)
            self.assertAlmostEqual(metrics["grounding"]["mean_iou"], 1.0)
            self.assertAlmostEqual(metrics["citations_valid"], 1.0)
            self.assertTrue(report.exists())
            self.assertTrue(json_out.exists())
            self.assertIn("Per-class accuracy", report.read_text(encoding="utf-8"))
            self.assertIn("action", report.read_text(encoding="utf-8"))

    def test_suite_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text('{"question": "q"}\n', encoding="utf-8")
            with self.assertRaises(Exception):
                load_suite(path)


if __name__ == "__main__":
    unittest.main()
