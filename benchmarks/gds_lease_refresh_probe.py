#!/usr/bin/env python3
"""Evaluate whether indirect BatchGetReplicaList lease refresh is safe.

Production refresh remains disabled. This tool compares synchronized baseline
and experimental JSONL captures and exits nonzero until every Master-load,
lease-expiry, grouped-key, and operation-shape gate passes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_probe_jsonl(path: str | Path) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_type = record.get("type")
            if record_type == "operation":
                operations.append(record)
            elif record_type == "summary":
                if summary is not None:
                    raise ValueError(f"{path}: multiple summary records")
                summary = record
            elif record_type != "header":
                raise ValueError(
                    f"{path}:{line_number}: unsupported record type"
                )
    if summary is None:
        raise ValueError(f"{path}: missing summary record")
    return {"summary": summary, "operations": operations}


def evaluate_refresh_probe(sample: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if int(sample.get("operation_count", 0)) < 80:
        reasons.append("operation_count")
    if int(sample.get("invalid_key_count_operations", 0)) != 0:
        reasons.append("key_count_shape")
    if float(sample.get("master_cpu_delta_pct", math.inf)) >= 5.0:
        reasons.append("master_cpu")
    if float(sample.get("batch_get_p99_delta_pct", math.inf)) >= 10.0:
        reasons.append("batch_get_p99")
    if int(sample.get("post_terminal_refreshes", -1)) != 0:
        reasons.append("post_terminal_refresh")

    lease_ttl_ms = float(sample.get("lease_ttl_ms", 0.0))
    expiry_ms = float(sample.get("expiry_after_last_refresh_ms", math.inf))
    if lease_ttl_ms <= 0.0 or expiry_ms > lease_ttl_ms * 1.25:
        reasons.append("lease_expiry")
    if float(sample.get("grouped_key_amplification", math.inf)) > 1.10:
        reasons.append("group_amplification")
    return {"passed": not reasons, "reasons": reasons, "sample": sample}


def compare_probe_runs(
    baseline: dict[str, Any], refresh: dict[str, Any]
) -> dict[str, Any]:
    baseline_summary = baseline["summary"]
    refresh_summary = refresh["summary"]
    operations = refresh["operations"]
    key_counts = [int(operation.get("key_count", -1)) for operation in operations]
    valid_shapes = {126, 192}
    invalid_shape_count = sum(count not in valid_shapes for count in key_counts)
    if not valid_shapes.issubset(set(key_counts)):
        invalid_shape_count += 1

    baseline_p99 = float(baseline_summary["batch_get_p99_us"])
    refresh_p99 = float(refresh_summary["batch_get_p99_us"])
    p99_delta_pct = (
        math.inf
        if baseline_p99 <= 0.0
        else (refresh_p99 - baseline_p99) * 100.0 / baseline_p99
    )
    post_terminal_refreshes = int(
        refresh_summary.get("post_terminal_refreshes", 0)
    ) + sum(
        int(operation.get("post_terminal_refreshes", 0))
        for operation in operations
    )
    sample = {
        "operation_count": len(operations),
        "invalid_key_count_operations": invalid_shape_count,
        "master_cpu_delta_pct": float(refresh_summary["master_cpu_pct"])
        - float(baseline_summary["master_cpu_pct"]),
        "batch_get_p99_delta_pct": p99_delta_pct,
        "post_terminal_refreshes": post_terminal_refreshes,
        "expiry_after_last_refresh_ms": float(
            refresh_summary.get("expiry_after_last_refresh_ms", math.inf)
        ),
        "lease_ttl_ms": float(refresh_summary["lease_ttl_ms"]),
        "grouped_key_amplification": float(
            refresh_summary.get("grouped_key_amplification", math.inf)
        ),
        # BatchGet refresh necessarily changes Get-family metrics. Preserve
        # the measured value so benchmark consumers do not misread hit-rate
        # changes as workload behavior.
        "metric_side_effect_batch_get_requests": int(
            refresh_summary.get("metric_side_effect_batch_get_requests", 0)
        ),
    }
    return evaluate_refresh_probe(sample)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--refresh", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = compare_probe_runs(
        load_probe_jsonl(args.baseline), load_probe_jsonl(args.refresh)
    )
    rendered = json.dumps(result, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
