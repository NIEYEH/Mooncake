#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vllm_warmup


class _FakeVllmHandler(BaseHTTPRequestHandler):
    post_payloads: list[dict] = []
    invalid_completion = False

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/status":
            self._send_json(200, {"status": "ok"})
            return
        if self.path != "/v1/models":
            self.send_error(404)
            return
        self._send_json(200, {"data": [{"id": "test-model"}]})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).post_payloads.append(payload)
        if type(self).invalid_completion:
            self._send_json(200, {"choices": []})
            return
        self._send_json(
            200,
            {
                "choices": [{"message": {"content": "ready"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            },
        )

    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class VllmWarmupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVllmHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def setUp(self) -> None:
        _FakeVllmHandler.post_payloads = []
        _FakeVllmHandler.invalid_completion = False

    def _config(self) -> vllm_warmup.WarmupConfig:
        return vllm_warmup.WarmupConfig(
            base_url=self.base_url,
            ready_path="/status",
            model="test-model",
            prompt_token_counts=(8, 16),
            requests=3,
            concurrency=2,
            min_duration_seconds=0,
            batch_interval_seconds=0,
            max_tokens=2,
            request_timeout_seconds=2,
            ready_timeout_seconds=2,
            settle_seconds=0,
            namespace="unit-test",
        )

    def test_run_warmup_runs_serial_probes_and_concurrent_load(self) -> None:
        summary = vllm_warmup.run_warmup(self._config())

        self.assertTrue(summary["success"])
        self.assertEqual(summary["compile_probe_requests"], 2)
        self.assertEqual(summary["load_requests"], 3)
        self.assertEqual(summary["requests"], 5)
        self.assertEqual(len(_FakeVllmHandler.post_payloads), 5)
        self.assertTrue(
            all(item["stream"] is False for item in _FakeVllmHandler.post_payloads)
        )
        self.assertTrue(
            all(
                "mooncake-vllm-warmup-unit-test" in item["messages"][0]["content"]
                for item in _FakeVllmHandler.post_payloads
            )
        )

    def test_invalid_completion_fails_warmup(self) -> None:
        _FakeVllmHandler.invalid_completion = True
        with self.assertRaisesRegex(RuntimeError, "invalid completion"):
            vllm_warmup.run_warmup(self._config())

    def test_load_minimum_duration_starts_after_compile_probes(self) -> None:
        calls = 0

        def fake_request(*_args: object, **_kwargs: object) -> dict:
            nonlocal calls
            calls += 1
            if calls <= len(self._config().prompt_token_counts):
                time.sleep(0.03)
            return {
                "latency_seconds": 0.001,
                "prompt_tokens": 1,
                "completion_tokens": 1,
            }

        config = replace(
            self._config(),
            requests=1,
            concurrency=1,
            min_duration_seconds=0.04,
            batch_interval_seconds=0.005,
        )
        with patch.object(
            vllm_warmup, "send_inference_request", side_effect=fake_request
        ):
            summary = vllm_warmup.run_warmup(config)

        self.assertGreaterEqual(summary["load_duration_seconds"], 0.035)
        self.assertGreater(summary["load_requests"], 1)

    def test_parse_prompt_token_counts(self) -> None:
        self.assertEqual(
            vllm_warmup.parse_prompt_token_counts("1, 64,256"), (1, 64, 256)
        )
        with self.assertRaises(Exception):
            vllm_warmup.parse_prompt_token_counts("32,0")


if __name__ == "__main__":
    unittest.main()
