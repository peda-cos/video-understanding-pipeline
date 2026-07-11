import unittest

import numpy as np

from vidqa import vision
from vidqa.embed import cosine


def scene_frame() -> np.ndarray:
    frame = np.zeros((216, 384, 3), dtype=np.uint8)
    frame[:] = (30, 60, 120)  # blue background
    frame[80:120, 80:140] = (230, 60, 60)  # red rectangle
    frame[30:60, 250:294] = (70, 200, 90)  # green rectangle
    frame[150:180, 280:324] = (70, 200, 90)  # green rectangle
    return frame


class VisionTests(unittest.TestCase):
    def test_detect_objects_counts_and_shapes(self) -> None:
        objects = vision.detect_objects(scene_frame())
        colors = [(obj["color"], obj["shape"]) for obj in objects]
        self.assertIn(("red", "rectangle"), colors)
        self.assertEqual(colors.count(("green", "rectangle")), 2)
        self.assertNotIn("blue", [obj["color"] for obj in objects])

    def test_analysis_produces_counts_and_motion(self) -> None:
        first = scene_frame()
        middle = scene_frame()
        last = scene_frame()
        # move the red rectangle right across the sampled frames
        middle[80:120, 80:140] = (30, 60, 120)
        middle[80:120, 140:200] = (230, 60, 60)
        last[80:120, 80:140] = (30, 60, 120)
        last[80:120, 250:310] = (230, 60, 60)
        facts = vision.analyze_scene([first, middle, last], [0, 1000, 2000])
        self.assertEqual(facts["object_counts"].get("red rectangle"), 1)
        self.assertEqual(facts["background_colors"][0], "blue")
        self.assertEqual(facts["motion"], "high")
        track = next(item for item in facts["tracks"] if item["color"] == "red")
        self.assertEqual(track["direction"], "left-to-right")
        self.assertIn("red rectangle", facts["caption"])

    def test_visual_descriptor_aligns_query(self) -> None:
        frame = scene_frame()
        facts = vision.analyze_scene([frame, frame, frame], [0, 500, 1000])
        descriptor = vision.visual_descriptor([frame, frame, frame], facts)
        self.assertEqual(descriptor.shape, (80,))
        self.assertAlmostEqual(float(np.linalg.norm(descriptor)), 1.0, places=5)
        red_query = vision.visual_query("red rectangle")
        yellow_query = vision.visual_query("yellow circle")
        self.assertGreater(
            cosine(red_query, descriptor),
            cosine(yellow_query, descriptor),
        )

    def test_parse_visual_terms(self) -> None:
        terms = vision.parse_visual_terms(
            "Does the green circle move from left to right in the scene?"
        )
        self.assertEqual(terms["colors"], ["green"])
        self.assertEqual(terms["shapes"], ["circle"])
        self.assertEqual(terms["direction"], "left-to-right")
        self.assertFalse(terms["count_intent"])

    def test_hsv_primaries(self) -> None:
        rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        hsv = vision.rgb_to_hsv(rgb)
        self.assertAlmostEqual(float(hsv[0, 0, 0]), 0.0, places=3)
        self.assertAlmostEqual(float(hsv[0, 1, 0]), 120.0, places=3)
        self.assertAlmostEqual(float(hsv[0, 2, 0]), 240.0, places=3)


if __name__ == "__main__":
    unittest.main()
