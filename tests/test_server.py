import http.client
import json
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request

from vidqa.server import VidqaServer
from tests.helpers import service


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = service()
        cls.server = VidqaServer(("127.0.0.1", 0), cls.service)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def get(self, path: str, headers: dict[str, str] | None = None) -> tuple[int, bytes, dict]:
        request = urllib.request.Request(self.base + path, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as exc:
            body = exc.read()
            headers = dict(exc.headers)
            exc.close()
            return exc.code, body, headers

    def test_health_and_videos(self) -> None:
        status, body, _ = self.get("/api/health")
        self.assertEqual(status, 200)
        health = json.loads(body)
        self.assertTrue(health["ok"])
        self.assertEqual(health["index"]["scenes"], 4)
        status, body, _ = self.get("/api/videos")
        self.assertEqual(status, 200)
        videos = json.loads(body)["videos"]
        self.assertEqual(len(videos), 2)
        self.assertTrue(videos[0]["video_url"].startswith("/media/"))

    def test_scenes_and_ask(self) -> None:
        status, body, _ = self.get("/api/videos/v001/scenes")
        self.assertEqual(status, 200)
        scenes = json.loads(body)["scenes"]
        self.assertEqual(len(scenes), 2)
        query = urllib.parse.quote(
            "How many green rectangles are visible in the scene about orchard?"
        )
        status, body, _ = self.get(f"/api/ask?q={query}&k=3")
        self.assertEqual(status, 200)
        answer = json.loads(body)
        self.assertTrue(answer["answer"].startswith("2"))
        self.assertLessEqual(len(answer["retrieval"]), 3)
        self.assertTrue(answer["citations"][0]["frames"])

    def test_bad_query_and_unknown_routes(self) -> None:
        status, _body, _ = self.get("/api/ask?q=")
        self.assertEqual(status, 400)
        status, _body, _ = self.get("/api/nope")
        self.assertEqual(status, 404)
        status, _body, _ = self.get("/api/videos/nope/scenes")
        self.assertEqual(status, 404)

    def test_range_request(self) -> None:
        status, body, headers = self.get(
            "/media/videos/v001/source.mp4", headers={"Range": "bytes=0-99"}
        )
        self.assertEqual(status, 206)
        self.assertEqual(len(body), 100)
        self.assertIn("bytes 0-99/", headers.get("Content-Range", ""))

    def test_traversal_is_blocked(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request("GET", "/media/%2e%2e/catalog.json")
        response = connection.getresponse()
        self.assertEqual(response.status, 404)
        connection.close()

    def test_static_index(self) -> None:
        status, body, _ = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"SCENE INDEX", body)


if __name__ == "__main__":
    unittest.main()
