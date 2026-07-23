#!/usr/bin/env python3
"""Collect synchronized GDS baseline samples with explicit availability."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_header(config_path: Path, executable_path: Path) -> dict[str, Any]:
    config = config_path.resolve()
    executable = executable_path.resolve()
    return {
        "type": "header",
        "wall_time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "config_path": str(config),
        "config_sha256": _sha256(config),
        "executable_path": str(executable),
        "executable_sha256": _sha256(executable),
    }


def _collect_gpu() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        output = subprocess.run(
            [
                executable,
                "--query-gpu=index,utilization.gpu,utilization.memory,"
                "memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
        rows = []
        for line in output.splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            rows.append(
                {
                    "index": int(fields[0]),
                    "gpu_util_pct": float(fields[1]),
                    "memory_util_pct": float(fields[2]),
                    "memory_used_mib": float(fields[3]),
                    "memory_total_mib": float(fields[4]),
                    "power_w": float(fields[5]),
                }
            )
        return {"available": True, "gpus": rows}
    except Exception as exc:
        return {"available": False, "reason": f"nvidia-smi failed: {exc}"}


def _collect_nvme(block_device: str | None) -> dict[str, Any]:
    if not block_device:
        return {"available": False, "reason": "block device not configured"}
    stat_path = Path("/sys/block") / block_device / "stat"
    if not stat_path.is_file():
        return {
            "available": False,
            "reason": f"block metrics unavailable for {block_device}",
        }
    try:
        fields = [int(value) for value in stat_path.read_text().split()]
        return {
            "available": True,
            "device": block_device,
            "reads_completed": fields[0],
            "sectors_read": fields[2],
            "read_time_ms": fields[3],
            "writes_completed": fields[4],
            "sectors_written": fields[6],
            "write_time_ms": fields[7],
            "ios_in_progress": fields[8],
            "io_time_ms": fields[9],
            "weighted_io_time_ms": fields[10],
        }
    except Exception as exc:
        return {"available": False, "reason": f"block stat failed: {exc}"}


def _collect_endpoint(url: str | None) -> dict[str, Any]:
    if not url:
        return {"available": False, "reason": "endpoint not configured"}
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=2) as response:
            payload = json.loads(response.read())
        return {"available": True, "payload": payload}
    except Exception as exc:
        return {"available": False, "reason": f"endpoint failed: {exc}"}


def collect_sample(
    block_device: str | None,
    endpoints: dict[str, str],
) -> dict[str, Any]:
    # One timestamp pair labels all source reads in this sample. Individual
    # source availability is never represented by a fabricated numeric zero.
    sample: dict[str, Any] = {
        "type": "sample",
        "wall_time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "gpu": _collect_gpu(),
        "nvme": _collect_nvme(block_device),
    }
    for source in ("vllm", "runtime", "kv_restore", "inference"):
        sample[source] = _collect_endpoint(endpoints.get(source))
    return sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--block-device")
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--vllm-endpoint")
    parser.add_argument("--runtime-endpoint")
    parser.add_argument("--kv-restore-endpoint")
    parser.add_argument("--inference-endpoint")
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.interval_seconds <= 0:
        parser.error("duration and interval must be greater than zero")

    endpoints = {
        "vllm": args.vllm_endpoint,
        "runtime": args.runtime_endpoint,
        "kv_restore": args.kv_restore_endpoint,
        "inference": args.inference_endpoint,
    }
    deadline = time.monotonic() + args.duration_seconds
    with args.output.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(build_header(args.config, args.executable),
                       sort_keys=True)
            + "\n"
        )
        while True:
            stream.write(
                json.dumps(
                    collect_sample(args.block_device, endpoints),
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(args.interval_seconds, remaining))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
