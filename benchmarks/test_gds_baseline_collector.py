import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gds_baseline_collector


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "mooncake-transfer-engine" / "tent" / "config"


def test_prometheus_parser_keeps_only_fixed_batch_get_counters():
    payload = """
# HELP mooncake_batch_get_calls_total Total calls
mooncake_batch_get_calls_total 2
mooncake_batch_get_requested_keys_total 80
mooncake_batch_get_selected_gds_keys_total 12
mooncake_batch_get_available_gds_keys_total{cluster_id="test"} 7
unrelated_metric{route="dynamic"} 99
mooncake_batch_get_bad_value_total not-a-number
"""
    counters = gds_baseline_collector.parse_prometheus_counters(payload)
    assert counters == {
        "mooncake_batch_get_calls_total": 2,
        "mooncake_batch_get_requested_keys_total": 80,
        "mooncake_batch_get_selected_gds_keys_total": 12,
        "mooncake_batch_get_available_gds_keys_total": 7,
    }


def test_missing_store_endpoint_is_explicitly_unavailable():
    result = gds_baseline_collector._collect_prometheus_endpoint(None)
    assert result["available"] is False
    assert result["reason"] == "endpoint not configured"
    assert "counters" not in result


def test_baseline_and_default_configs_are_fixed_and_conservative():
    for name in ("tent-gds.json", "tent-gds-baseline.json"):
        config = json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))
        assert config["runtime_queue"]["gds_scheduler_mode"] == "fixed"
        assert config["runtime_queue"]["gds_shared_physical_tokens"] == 16
        assert config["runtime_queue"]["max_dispatch_read_owners"] == 16
        assert config["runtime_queue"]["max_dispatch_write_owners"] == 1
        assert config["transports"]["gds"]["read_worker_threads"] == 16
        assert config["transports"]["gds"]["max_inflight_reads"] == 16
        assert config["transports"]["gds"]["write_worker_threads"] == 4
        assert config["transports"]["gds"]["max_inflight_writes"] == 1
        assert config["transports"]["gds"]["adaptive_concurrency"] is False
        assert config["transports"]["gds"]["batch_api"] is False
        assert config["transports"]["gds"]["async_api"] is False


def test_weighted_config_uses_shared_tokens_and_completed_byte_quanta():
    config = json.loads(
        (CONFIG_DIR / "tent-gds-weighted.json").read_text(encoding="utf-8")
    )
    queue = config["runtime_queue"]
    assert queue["gds_scheduler_mode"] == "weighted_fair"
    assert queue["gds_shared_physical_tokens"] == 16
    assert queue["gds_read_quantum_bytes"] == 8 << 20
    assert queue["gds_write_quantum_bytes"] == 2 << 20
    assert queue["gds_contended_write_tokens"] == 1
    assert queue["gds_write_standalone_tokens"] == 2
    assert config["transports"]["gds"]["adaptive_concurrency"] is False


def test_write4_target_config_matches_all_three_concurrency_layers():
    config = json.loads(
        (CONFIG_DIR / "tent-gds-write4.json").read_text(encoding="utf-8")
    )
    queue = config["runtime_queue"]
    gds = config["transports"]["gds"]
    assert queue["max_dispatch_write_owners"] == 4
    assert queue["gds_shared_physical_tokens"] == 16
    assert queue["gds_read_standalone_tokens"] == 16
    assert queue["gds_write_standalone_tokens"] == 4
    assert queue["gds_contended_write_tokens"] == 1
    assert queue["gds_primary_read_tokens"] == 16
    assert gds["write_worker_threads"] == 4
    assert gds["shared_device_tokens"] == 16
    assert gds["max_inflight_writes"] == 4
    assert gds["adaptive_concurrency"] is False


def test_missing_gpu_and_block_tools_are_explicitly_unavailable(tmp_path):
    with patch.object(
        gds_baseline_collector.shutil, "which", return_value=None
    ):
        sample = gds_baseline_collector.collect_sample(
            block_device="nvme9n9",
            endpoints={},
        )
    assert sample["gpu"]["available"] is False
    assert sample["gpu"]["reason"] == "nvidia-smi not found"
    assert sample["nvme"]["available"] is False
    assert "zero" not in sample["nvme"].get("reason", "").lower()
    assert sample["finished_monotonic_ns"] >= sample["monotonic_ns"]
    for source in (
        "gpu",
        "nvme",
        "vllm",
        "runtime",
        "kv_restore",
        "inference",
        "store",
    ):
        assert (
            sample[source]["source_finished_monotonic_ns"]
            >= sample[source]["source_started_monotonic_ns"]
        )


def test_header_fingerprints_exact_config_and_binary(tmp_path):
    config = tmp_path / "config.json"
    binary = tmp_path / "server"
    config.write_text('{"mode":"fixed"}\n', encoding="utf-8")
    binary.write_bytes(b"binary-id")
    header = gds_baseline_collector.build_header(config, binary)
    assert header["type"] == "header"
    assert header["config_path"] == str(config.resolve())
    assert len(header["config_sha256"]) == 64
    assert header["executable_path"] == str(binary.resolve())
    assert len(header["executable_sha256"]) == 64
