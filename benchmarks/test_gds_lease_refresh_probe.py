import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gds_lease_refresh_probe import (
    compare_probe_runs,
    evaluate_refresh_probe,
    load_probe_jsonl,
)


def valid_sample():
    return {
        "operation_count": 80,
        "invalid_key_count_operations": 0,
        "master_cpu_delta_pct": 2.0,
        "batch_get_p99_delta_pct": 4.0,
        "post_terminal_refreshes": 0,
        "expiry_after_last_refresh_ms": 95.0,
        "lease_ttl_ms": 100.0,
        "grouped_key_amplification": 1.02,
    }


def test_evaluate_refresh_probe_accepts_all_gates():
    result = evaluate_refresh_probe(valid_sample())
    assert result["passed"]
    assert result["reasons"] == []


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("master_cpu_delta_pct", 5.0, "master_cpu"),
        ("batch_get_p99_delta_pct", 10.0, "batch_get_p99"),
        ("post_terminal_refreshes", 1, "post_terminal_refresh"),
        ("expiry_after_last_refresh_ms", 130.0, "lease_expiry"),
        ("grouped_key_amplification", 1.2, "group_amplification"),
        ("operation_count", 79, "operation_count"),
        ("invalid_key_count_operations", 1, "key_count_shape"),
    ],
)
def test_evaluate_refresh_probe_rejects_each_gate(field, value, reason):
    sample = valid_sample()
    sample[field] = value
    result = evaluate_refresh_probe(sample)
    assert not result["passed"]
    assert reason in result["reasons"]


def test_compare_probe_runs_computes_cpu_and_p99_deltas():
    baseline = {
        "summary": {
            "master_cpu_pct": 40.0,
            "batch_get_p99_us": 1000.0,
            "lease_ttl_ms": 100.0,
        },
        "operations": [
            {"operation_id": i, "key_count": 126 if i % 2 == 0 else 192}
            for i in range(80)
        ],
    }
    refresh = {
        "summary": {
            "master_cpu_pct": 42.0,
            "batch_get_p99_us": 1050.0,
            "lease_ttl_ms": 100.0,
            "expiry_after_last_refresh_ms": 100.0,
            "grouped_key_amplification": 1.0,
            "post_terminal_refreshes": 0,
        },
        "operations": baseline["operations"],
    }
    result = compare_probe_runs(baseline, refresh)
    assert result["passed"]
    assert result["sample"]["master_cpu_delta_pct"] == 2.0
    assert result["sample"]["batch_get_p99_delta_pct"] == 5.0


def test_load_probe_jsonl_requires_one_summary(tmp_path):
    path = tmp_path / "probe.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"type": "operation", "operation_id": 1, "key_count": 126}
                ),
                json.dumps(
                    {
                        "type": "summary",
                        "master_cpu_pct": 1.0,
                        "batch_get_p99_us": 2.0,
                        "lease_ttl_ms": 100.0,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_probe_jsonl(path)
    assert loaded["summary"]["lease_ttl_ms"] == 100.0
    assert len(loaded["operations"]) == 1
