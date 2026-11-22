import unittest

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app, embed
from app.security import APIKeyAuth


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(api_keys="test-key", rate_limit_per_minute=100)
        self.client = TestClient(create_app(settings=self.settings))

    def test_health_is_public(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertIn("X-Request-ID", response.headers)

    def test_embed_requires_api_key(self):
        self.assertEqual(self.client.post("/v1/embed", json={"text": "hi"}).status_code, 401)
        response = self.client.post(
            "/v1/embed", json={"text": "hi"}, headers={"X-API-Key": "wrong"}
        )
        self.assertEqual(response.status_code, 401)

    def test_embed_returns_vector(self):
        response = self.client.post(
            "/v1/embed",
            json={"text": "hello world"},
            headers={"X-API-Key": "test-key"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["dim"], self.settings.embedding_dim)
        self.assertEqual(len(body["vector"]), self.settings.embedding_dim)

    def test_ask_is_grounded_and_cites_sources(self):
        response = self.client.post(
            "/v1/ask",
            json={"question": "How does rate limiting work?", "top_k": 2},
            headers={"X-API-Key": "test-key"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("Rate limiting", body["answer"])
        self.assertTrue(body["sources"])
        self.assertIn("api-guide.txt", [s["document"] for s in body["sources"]])

    def test_validation_error(self):
        response = self.client.post(
            "/v1/embed", json={"text": ""}, headers={"X-API-Key": "test-key"}
        )
        self.assertEqual(response.status_code, 422)

    def test_rate_limit(self):
        settings = Settings(api_keys="limited", rate_limit_per_minute=1)
        client = TestClient(create_app(settings=settings))
        headers = {"X-API-Key": "limited"}
        self.assertEqual(client.get("/health", headers=headers).status_code, 200)
        first = client.post("/v1/embed", json={"text": "a"}, headers=headers)
        second = client.post("/v1/embed", json={"text": "b"}, headers=headers)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)

    def test_embedding_normalized(self):
        vector = embed("semantic search", 64)
        self.assertAlmostEqual(sum(v * v for v in vector), 1.0, places=6)

    def test_bad_rate_limit(self):
        with self.assertRaises(ValueError):
            APIKeyAuth(Settings(api_keys="x", rate_limit_per_minute=0))


if __name__ == "__main__":
    unittest.main()
