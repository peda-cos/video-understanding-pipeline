import json
import unittest

import numpy as np

from vidqa.caption import GeminiCaptioner, GeminiClient, ProviderError, extract_json
from vidqa.qdrant import QdrantStore
from tests.helpers import demo_paths


class FakeTransport:
    def __init__(self, responses: list[tuple[int, str]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, method, url, headers, body, timeout):  # noqa: ANN001
        self.calls.append({"method": method, "url": url, "headers": headers, "body": body})
        if not self.responses:
            return 500, "no response queued"
        return self.responses.pop(0)


def gemini_response(text: str) -> str:
    return json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]})


class GeminiTests(unittest.TestCase):
    def test_extract_json_from_fence(self) -> None:
        parsed = extract_json('```json\n{"caption": "a red scene"}\n```')
        self.assertEqual(parsed["caption"], "a red scene")
        with self.assertRaises(ProviderError):
            extract_json("no json here")

    def test_client_sends_image_and_parses_answer(self) -> None:
        _corpus, index, _benchmark = demo_paths()
        image = next((index / "videos" / "v001" / "frames").glob("*.jpg"))
        transport = FakeTransport([(200, gemini_response('{"caption": "red rectangle"}'))])
        client = GeminiClient(api_key="fake-key-only", transport=transport)
        captioner = GeminiCaptioner(client=client)
        caption = captioner.caption(str(image), {})
        self.assertEqual(caption, "red rectangle")
        call = transport.calls[0]
        self.assertIn(":generateContent", call["url"])
        self.assertIn("x-goog-api-key", call["headers"])
        parts = call["body"]["contents"][0]["parts"]
        self.assertEqual(parts[0]["text"].find("Caption"), 0)
        self.assertIn("inline_data", parts[1])

    def test_client_http_error(self) -> None:
        transport = FakeTransport([(403, "denied")])
        client = GeminiClient(api_key="fake-key-only", transport=transport)
        with self.assertRaises(ProviderError):
            client.generate([{"text": "hi"}])


class QdrantTests(unittest.TestCase):
    def make_store(self, responses: list[tuple[int, str]]) -> tuple[QdrantStore, FakeTransport]:
        transport = FakeTransport(responses)
        store = QdrantStore("http://qdrant.test", transport=transport, api_key="fake")
        return store, transport

    def test_collection_and_upsert_payload(self) -> None:
        store, transport = self.make_store(
            [(404, ""), (200, "{}"), (200, '{"result": {"status": "completed"}}')]
        )
        store.ensure_collection(recreate=True)
        self.assertEqual(transport.calls[0]["method"], "DELETE")
        self.assertEqual(transport.calls[1]["method"], "PUT")
        vectors = transport.calls[1]["body"]["vectors"]
        self.assertEqual(vectors["keyframe"]["size"], 80)
        self.assertEqual(vectors["caption"]["distance"], "Cosine")
        payload = {"video_id": "v001", "scene_id": 1, "start_ms": 0, "end_ms": 10}
        store.upsert_scene(
            payload,
            {
                "caption": np.ones(384, dtype=np.float32),
                "keyframe": np.ones(80, dtype=np.float32),
                "transcript": np.zeros(384, dtype=np.float32),
            },
        )
        points = transport.calls[2]["body"]["points"]
        self.assertEqual(len(points), 1)
        self.assertIn("caption", points[0]["vector"])
        self.assertNotIn("transcript", points[0]["vector"])
        self.assertRegex(points[0]["id"], r"^[0-9a-f-]{36}$")

    def test_search_parses_result(self) -> None:
        response = {
            "result": {
                "points": [
                    {"id": "x", "score": 0.8, "payload": {"video_id": "v001", "scene_id": 2}}
                ]
            }
        }
        store, transport = self.make_store([(200, json.dumps(response))])
        hits = store.search(
            "caption", np.ones(384, dtype=np.float32), limit=5, video_id="v001"
        )
        self.assertEqual(hits, [{"video_id": "v001", "scene_id": 2, "score": 0.8}])
        body = transport.calls[0]["body"]
        self.assertEqual(body["using"], "caption")
        self.assertEqual(body["filter"]["must"][0]["key"], "video_id")


if __name__ == "__main__":
    unittest.main()
