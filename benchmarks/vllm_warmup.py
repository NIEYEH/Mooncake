#!/usr/bin/env python3
"""Warm up a vLLM OpenAI-compatible endpoint before timed benchmarks.

The vLLM HTTP readiness endpoint only proves that the server has started.  It
does not execute the scheduler/attention path, so Triton kernels such as
``_compute_slot_mapping_kernel`` may otherwise compile during the first timed
request. This helper sends completed inference requests for every configured
input/output shape, then keeps a concurrent load running for a minimum
duration.

The script intentionally uses only the Python standard library so it can run
inside the same environment as vLLM without installing benchmark-only
dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_PROMPT_TOKEN_COUNTS = (1024, 4096)
DEFAULT_OUTPUT_TOKEN_COUNTS = (6, 256)


def _open_direct(request: urllib.request.Request, timeout: float):
    # Benchmark endpoints are normally on localhost or a private fabric.  Do
    # not accidentally route them through HTTP_PROXY; create an opener per
    # request so concurrent warmup workers do not share handler state.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


@dataclass(frozen=True)
class WarmupConfig:
    base_url: str
    ready_path: str
    model: str
    prompt_token_counts: tuple[int, ...]
    requests: int
    concurrency: int
    min_duration_seconds: float
    batch_interval_seconds: float
    # Retained for compatibility with callers that construct WarmupConfig
    # directly. When set, it overrides output_token_counts just like the
    # deprecated --max-tokens CLI option.
    max_tokens: int | None
    request_timeout_seconds: float
    ready_timeout_seconds: float
    settle_seconds: float
    namespace: str
    output_token_counts: tuple[int, ...] = DEFAULT_OUTPUT_TOKEN_COUNTS
    compile_probe_repetitions: int = 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_token_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "token counts must be comma-separated integers"
        ) from exc
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError(
            "token counts must contain positive integers"
        )
    return counts


# Keep the public helper name used by older callers.
parse_prompt_token_counts = parse_token_counts


def _join_url(base_url: str, path: str) -> str:
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def wait_until_ready(base_url: str, ready_path: str, timeout_seconds: float) -> None:
    """Wait until the selected vLLM or proxy readiness endpoint returns 200."""

    ready_url = _join_url(base_url, ready_path)
    deadline = time.monotonic() + timeout_seconds
    last_error = "no response"
    while True:
        try:
            request = urllib.request.Request(ready_url, method="GET")
            with _open_direct(request, min(5.0, timeout_seconds)) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except Exception as exc:  # Readiness failures are reported at timeout.
            last_error = repr(exc)
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"vLLM did not become ready within {timeout_seconds:.1f}s: "
                f"{last_error}"
            )
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def make_prompt(namespace: str, request_id: int, approximate_tokens: int) -> str:
    """Create a deterministic, cache-isolated prompt of roughly N tokens."""

    # Put the unique marker first so prefix caching cannot turn a compile probe
    # into a full-prefix hit left behind by an unrelated benchmark run.
    marker = f"mooncake-vllm-warmup-{namespace}-{request_id}"
    return marker + " " + "warmup " * approximate_tokens


def build_payload(
    model: str,
    namespace: str,
    request_id: int,
    prompt_tokens: int,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": make_prompt(namespace, request_id, prompt_tokens),
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        # vLLM extension also used by benchmark_serving.py. Without it a
        # generated EOS can end a nominal 256-token probe early, leaving the
        # longest decode path unexercised until the timed run.
        "ignore_eos": True,
        "stream": False,
    }


def send_inference_request(
    api_url: str, payload: dict[str, Any], timeout_seconds: float
) -> dict[str, Any]:
    """Send one inference request and require a completed OpenAI response."""

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with _open_direct(request, timeout_seconds) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(
            f"warmup request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"warmup request failed: {exc!r}") from exc

    latency = time.perf_counter() - started
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"warmup request returned non-JSON HTTP {status}: "
            f"{raw.decode('utf-8', 'replace')[:1000]}"
        ) from exc
    if status != 200 or body.get("error") or not body.get("choices"):
        raise RuntimeError(
            f"warmup request returned an invalid completion: "
            f"HTTP {status}, body={body!r}"
        )
    return {
        "latency_seconds": latency,
        "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (body.get("usage") or {}).get("completion_tokens"),
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _run_batch(
    config: WarmupConfig,
    api_url: str,
    request_ids: Iterable[int],
    shapes: Sequence[tuple[int, int]],
) -> list[dict[str, Any]]:
    ids = list(request_ids)
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        futures = {}
        for request_id in ids:
            prompt_tokens, output_tokens = shapes[request_id % len(shapes)]
            payload = build_payload(
                config.model,
                config.namespace,
                request_id,
                prompt_tokens,
                output_tokens,
            )
            future = pool.submit(
                send_inference_request,
                api_url,
                payload,
                config.request_timeout_seconds,
            )
            futures[future] = (request_id, prompt_tokens, output_tokens)

        results = []
        for future in as_completed(futures):
            request_id, prompt_tokens, output_tokens = futures[future]
            result = future.result()
            result.update(
                {
                    "request_id": request_id,
                    "approximate_prompt_tokens": prompt_tokens,
                    "requested_output_tokens": output_tokens,
                }
            )
            results.append(result)
        return results


def run_warmup(config: WarmupConfig) -> dict[str, Any]:
    """Run serial compile probes followed by a bounded concurrent warmup."""

    wait_until_ready(config.base_url, config.ready_path, config.ready_timeout_seconds)
    api_url = _join_url(config.base_url, "/v1/chat/completions")
    total_started = time.monotonic()
    results: list[dict[str, Any]] = []
    request_id = 0
    output_token_counts = (
        (config.max_tokens,) if config.max_tokens is not None
        else config.output_token_counts
    )
    shapes = tuple(product(config.prompt_token_counts, output_token_counts))

    # Execute every input/output shape serially first. Repeating each shape is
    # important behind a round-robin PD proxy: the caller sets repetitions to
    # max(prefill instances, decode instances), so every participating engine
    # sees every shape before benchmark_serving.py starts its timer.
    for prompt_tokens, output_tokens in shapes:
        for _ in range(config.compile_probe_repetitions):
            payload = build_payload(
                config.model,
                config.namespace,
                request_id,
                prompt_tokens,
                output_tokens,
            )
            result = send_inference_request(
                api_url, payload, config.request_timeout_seconds
            )
            result.update(
                {
                    "request_id": request_id,
                    "approximate_prompt_tokens": prompt_tokens,
                    "requested_output_tokens": output_tokens,
                }
            )
            results.append(result)
            request_id += 1

    # The minimum duration applies to the concurrent steady-state phase, not
    # to the serial compile probes above.  A slow first Triton compilation
    # must not consume the load-warmup budget.
    load_started = time.monotonic()
    load_requests_completed = 0
    while (
        load_requests_completed < config.requests
        or time.monotonic() - load_started < config.min_duration_seconds
    ):
        remaining = max(0, config.requests - load_requests_completed)
        batch_size = (
            min(config.concurrency, remaining) if remaining else config.concurrency
        )
        batch_ids = range(request_id, request_id + batch_size)
        results.extend(_run_batch(config, api_url, batch_ids, shapes))
        request_id += batch_size
        load_requests_completed += batch_size
        elapsed = time.monotonic() - load_started
        if config.batch_interval_seconds > 0 and elapsed < config.min_duration_seconds:
            time.sleep(config.batch_interval_seconds)

    load_duration = time.monotonic() - load_started
    if config.settle_seconds:
        time.sleep(config.settle_seconds)
    total_duration = time.monotonic() - total_started
    latencies = [float(item["latency_seconds"]) for item in results]
    return {
        "success": True,
        "base_url": config.base_url,
        "ready_path": config.ready_path,
        "model": config.model,
        "namespace": config.namespace,
        "compile_probe_prompt_token_counts": list(config.prompt_token_counts),
        "compile_probe_output_token_counts": list(output_token_counts),
        "compile_probe_shapes": [
            {"prompt_tokens": prompt, "output_tokens": output}
            for prompt, output in shapes
        ],
        "requests": len(results),
        "compile_probe_repetitions": config.compile_probe_repetitions,
        "compile_probe_requests": (
            len(shapes) * config.compile_probe_repetitions
        ),
        "load_requests": load_requests_completed,
        "concurrency": config.concurrency,
        "load_duration_seconds": load_duration,
        "total_duration_seconds": total_duration,
        "settle_seconds": config.settle_seconds,
        "latency_avg_seconds": statistics.fmean(latencies),
        "latency_p50_seconds": _percentile(latencies, 0.50),
        "latency_p99_seconds": _percentile(latencies, 0.99),
        "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in results),
        "completion_tokens": sum(
            int(item.get("completion_tokens") or 0) for item in results
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="vLLM or PD proxy base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--ready-path",
        default="/v1/models",
        help="HTTP readiness path exposed by the selected endpoint",
    )
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument(
        "--prompt-token-counts",
        type=parse_token_counts,
        default=DEFAULT_PROMPT_TOKEN_COUNTS,
        help="comma-separated approximate prompt lengths (default: 1024,4096)",
    )
    parser.add_argument(
        "--output-token-counts",
        type=parse_token_counts,
        default=DEFAULT_OUTPUT_TOKEN_COUNTS,
        help="comma-separated generated lengths to compile (default: 6,256)",
    )
    parser.add_argument(
        "--compile-probe-repetitions",
        type=_positive_int,
        default=1,
        help=(
            "serial requests per input/output shape; behind a round-robin "
            "proxy use max(prefill instances, decode instances)"
        ),
    )
    parser.add_argument(
        "--requests",
        type=_non_negative_int,
        default=16,
        help=(
            "minimum concurrent-load requests after serial probes "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=16,
        help="concurrent warmup requests (default: %(default)s)",
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=_non_negative_float,
        default=60.0,
        help="minimum active warmup duration (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-interval-seconds",
        type=_non_negative_float,
        default=1.0,
        help="pause between warmup batches while satisfying minimum duration",
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=None,
        help=(
            "deprecated compatibility override: use one output length for "
            "all requests instead of --output-token-counts"
        ),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=_non_negative_float,
        default=300.0,
        help="individual inference timeout (default: %(default)s)",
    )
    parser.add_argument(
        "--ready-timeout-seconds",
        type=_non_negative_float,
        default=360.0,
        help="API readiness timeout (default: %(default)s)",
    )
    parser.add_argument(
        "--settle-seconds",
        type=_non_negative_float,
        default=15.0,
        help="quiet period after successful inference warmup (default: %(default)s)",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help=(
            "prompt namespace; reuse it across vLLM restarts to exercise "
            "external-cache restore during warmup"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for the JSON warmup summary",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> WarmupConfig:
    namespace = args.namespace or f"{int(time.time())}-{os.getpid()}"
    return WarmupConfig(
        base_url=args.base_url,
        ready_path=args.ready_path,
        model=args.model,
        prompt_token_counts=tuple(args.prompt_token_counts),
        requests=args.requests,
        concurrency=args.concurrency,
        min_duration_seconds=args.min_duration_seconds,
        batch_interval_seconds=args.batch_interval_seconds,
        max_tokens=args.max_tokens,
        request_timeout_seconds=args.request_timeout_seconds,
        ready_timeout_seconds=args.ready_timeout_seconds,
        settle_seconds=args.settle_seconds,
        namespace=namespace,
        output_token_counts=tuple(args.output_token_counts),
        compile_probe_repetitions=args.compile_probe_repetitions,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.request_timeout_seconds == 0 or args.ready_timeout_seconds == 0:
        parser.error("request and readiness timeouts must be greater than zero")
    try:
        summary = run_warmup(config_from_args(args))
    except Exception as exc:
        print(f"vLLM warmup failed: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
