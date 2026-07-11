import unittest

import numpy as np

from vidqa.embed import TextEmbedder, content_tokens, cosine


class EmbeddingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedder = TextEmbedder()

    def test_deterministic_and_normalized(self) -> None:
        a = self.embedder.encode("the red rectangle moves left to right")
        b = self.embedder.encode("the red rectangle moves left to right")
        np.testing.assert_allclose(a, b)
        self.assertEqual(a.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)

    def test_related_query_scores_higher(self) -> None:
        query = self.embedder.encode("red rectangle moves left to right")
        relevant = self.embedder.encode(
            "the red rectangle moves left to right near the marker"
        )
        unrelated = self.embedder.encode("a quiet blue harbor at night")
        self.assertGreater(cosine(query, relevant), cosine(query, unrelated))

    def test_stopwords_and_empty_text(self) -> None:
        self.assertEqual(content_tokens("How many of the red rectangles?"), ["red", "rectangles"])
        empty = self.embedder.encode("how many of the")
        self.assertEqual(float(np.linalg.norm(empty)), 0.0)
        self.assertEqual(cosine(empty, empty), 0.0)

    def test_bigrams_change_vector(self) -> None:
        one = self.embedder.encode("red rectangle")
        two = self.embedder.encode("rectangle red")
        self.assertLess(cosine(one, two), 0.99)


if __name__ == "__main__":
    unittest.main()
