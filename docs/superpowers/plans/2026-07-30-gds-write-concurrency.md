# GDS Write Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe opt-in four-WRITE GDS configuration, actionable token validation, truthful dispatch diagnostics, and lossless external-cache lookup metrics without changing the conservative production default.

**Architecture:** `TransferEngineImpl` remains the runtime scheduling authority, `GdsOperationScheduler` owns physical-token reservations, and `GdsTransport` enforces worker/direction/shared execution limits. A pure dispatch-decision helper makes the actual WRITE limit and block reason testable. Store lookup observations flow through the existing `ClientMetric` endpoint so the synchronized collector can distinguish metadata availability, selected route, and GDS submission.

**Tech Stack:** C++17, existing standalone C++ scheduler tests, GoogleTest Store metrics tests, Python 3/pytest collectors, JSON TENT configuration.

## Global Constraints

- Production `tent-gds.json` and `tent-gds-baseline.json` retain WRITE concurrency 1 until target-host gates pass.
- Opt-in `tent-gds-write4.json` uses runtime WRITE owners 4, standalone WRITE tokens 4, WRITE workers/inflight 4, contended WRITE tokens 1, and shared tokens 16.
- READ plus WRITE reservations never exceed 16 shared physical tokens.
- Fixed-mode READ pressure pauses only new WRITEs; submitted WRITEs drain normally.
- cuFile Batch, Async, adaptive concurrency, request merging, and lease behavior remain unchanged.
- Metrics must not contain object keys, segment URIs, prompt content, or unbounded labels.

---

### Task 1: Actionable scheduler token validation

**Files:**
- Modify: `mooncake-transfer-engine/tent/src/runtime/gds_operation_scheduler.cpp`
- Modify: `mooncake-transfer-engine/tent/tests/gds_operation_scheduler_test.cpp`

**Interfaces:**
- Consumes: `GdsOperationSchedulerConfig`.
- Produces: field-specific `Status::InvalidArgument` messages and preserves `GdsOperationScheduler::status()`.

- [ ] **Step 1: Add failing validation-message tests**

Add a string assertion helper and four table-driven invalid configurations:

```cpp
void expectContains(const std::string& text, const std::string& needle) {
    EXPECT_TRUE(text.find(needle) != std::string::npos);
}

void testInvalidTokenRelationshipsNameValues() {
    struct Case {
        const char* relationship;
        void (*mutate)(GdsOperationSchedulerConfig&);
    };
    const std::vector<Case> cases = {
        {"read_standalone_tokens <= shared_tokens",
         [](GdsOperationSchedulerConfig& c) {
             c.read_standalone_tokens = 17;
         }},
        {"write_standalone_tokens <= shared_tokens",
         [](GdsOperationSchedulerConfig& c) {
             c.write_standalone_tokens = 17;
         }},
        {"contended_write_tokens <= write_standalone_tokens",
         [](GdsOperationSchedulerConfig& c) {
             c.contended_write_tokens = 3;
         }},
        {"primary_read_tokens <= shared_tokens",
         [](GdsOperationSchedulerConfig& c) {
             c.primary_read_tokens = 17;
         }},
    };
    for (const auto& item : cases) {
        auto config = weightedConfig();
        item.mutate(config);
        const auto status = GdsOperationScheduler(config).status();
        EXPECT_TRUE(!status.ok());
        expectContains(status.ToString(), item.relationship);
        expectContains(status.ToString(), "shared_tokens=16");
        expectContains(status.ToString(), "read_standalone_tokens=");
        expectContains(status.ToString(), "write_standalone_tokens=");
        expectContains(status.ToString(), "contended_write_tokens=");
        expectContains(status.ToString(), "primary_read_tokens=");
    }
}
```

Add `testInvalidTokenRelationshipsNameValues()` to `main()`.

- [ ] **Step 2: Run the scheduler test and verify RED**

Run:

```bash
cmake --build build --target tent_gds_operation_scheduler_test -j
ctest --test-dir build -R tent_gds_operation_scheduler_test --output-on-failure
```

Expected: FAIL because the current error is only `GDS direction/operation tokens exceed shared tokens`.

- [ ] **Step 3: Implement field-specific validation**

In `validateConfig()`, build a bounded value suffix:

```cpp
const auto token_values = [&] {
    return " [shared_tokens=" + std::to_string(config_.shared_tokens) +
           ", read_standalone_tokens=" +
           std::to_string(config_.read_standalone_tokens) +
           ", write_standalone_tokens=" +
           std::to_string(config_.write_standalone_tokens) +
           ", contended_write_tokens=" +
           std::to_string(config_.contended_write_tokens) +
           ", primary_read_tokens=" +
           std::to_string(config_.primary_read_tokens) + "]";
};
```

Return one error per violated relationship in the same order as the tests.
Do not add a `primary_read_tokens + contended_write_tokens` startup check:
these are caps over one shared pool, not reserved partitions.

- [ ] **Step 4: Add a valid 16/4/1 and shared-pool regression**

Add:

```cpp
void testWriteFourConfigSharesTokensWithRead() {
    auto config = weightedConfig();
    config.write_standalone_tokens = 4;
    GdsOperationScheduler scheduler(config);
    EXPECT_TRUE(scheduler.status().ok());
    // Existing contended selection must remain 15 READ + 1 WRITE.
    // Assert snapshot.global_reserved_tokens == 16 and never 17.
}
```

- [ ] **Step 5: Run the scheduler test and verify GREEN**

Run the Task 1 command again. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mooncake-transfer-engine/tent/src/runtime/gds_operation_scheduler.cpp mooncake-transfer-engine/tent/tests/gds_operation_scheduler_test.cpp
git commit -m "fix: report invalid GDS token relationships"
```

---

### Task 2: Truthful WRITE dispatch decisions and READ-pressure accounting

**Files:**
- Modify: `mooncake-transfer-engine/tent/include/tent/transport/gds/gds_fifo_dispatch.h`
- Modify: `mooncake-transfer-engine/tent/tests/gds_fifo_dispatch_test.cpp`
- Modify: `mooncake-transfer-engine/tent/include/tent/runtime/transport.h`
- Modify: `mooncake-transfer-engine/tent/include/tent/transport/gds/gds_transport.h`
- Modify: `mooncake-transfer-engine/tent/src/transport/gds/gds_transport.cpp`
- Modify: `mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp`
- Modify: `mooncake-transfer-engine/tent/tests/runtime_queue_dispatch_test.cpp`

**Interfaces:**
- Produces:

```cpp
enum class GdsWriteDispatchBlockReason : uint8_t {
    None,
    SharedTokens,
    WriteDirectionLimit,
    WritePausedForRead,
    WorkerPool,
    FifoFront,
    Count,
};

struct GdsWriteDispatchDecision {
    size_t configured_limit;
    size_t current_limit;
    size_t runtime_limit;
    bool read_pressure;
    GdsWriteDispatchBlockReason reason;
};

GdsWriteDispatchDecision gdsWriteDispatchDecision(
    size_t configured_limit, size_t current_limit,
    size_t contended_limit, bool pause_for_read,
    bool read_pressure);
```

- Extends `Transport::updateRuntimeQueueDepth` with
  `reserved_read_tokens`.

- [ ] **Step 1: Add failing pure decision tests**

Add cases proving:

```cpp
EXPECT_EQ(gdsWriteDispatchDecision(4, 4, 1, true, false).runtime_limit, 4u);
const auto paused = gdsWriteDispatchDecision(4, 4, 1, true, true);
EXPECT_EQ(paused.configured_limit, 4u);
EXPECT_EQ(paused.current_limit, 4u);
EXPECT_EQ(paused.runtime_limit, 0u);
EXPECT_EQ(paused.reason,
          GdsWriteDispatchBlockReason::WritePausedForRead);
EXPECT_EQ(gdsWriteDispatchDecision(4, 4, 1, false, true).runtime_limit, 1u);
```

- [ ] **Step 2: Run FIFO tests and verify RED**

Run:

```bash
cmake --build build --target tent_gds_fifo_dispatch_test -j
ctest --test-dir build -R tent_gds_fifo_dispatch_test --output-on-failure
```

Expected: compile failure because the decision API does not exist.

- [ ] **Step 3: Implement the pure decision helper**

Keep configured/current values unchanged. Compute only the runtime limit:

```cpp
if (pause_for_read && read_pressure) {
    return {configured_limit, current_limit, 0, true,
            GdsWriteDispatchBlockReason::WritePausedForRead};
}
const size_t limit =
    gdsFifoEffectiveWriteLimit(current_limit, contended_limit,
                               read_pressure);
return {configured_limit, current_limit, limit, read_pressure,
        limit == 0 ? GdsWriteDispatchBlockReason::WriteDirectionLimit
                   : GdsWriteDispatchBlockReason::None};
```

- [ ] **Step 4: Add a failing runtime reserved-READ pressure test**

Extend the fake transport in `runtime_queue_dispatch_test.cpp` to capture the
third `updateRuntimeQueueDepth` argument. Enqueue a READ owner selected into the
dispatch window but not completed and assert `reserved_read_tokens > 0` is
reported even when admitted queued READ owners are zero.

- [ ] **Step 5: Pass reserved READ tokens across the runtime boundary**

Change the virtual interface to:

```cpp
virtual void updateRuntimeQueueDepth(
    size_t queued_reads, size_t queued_writes,
    size_t reserved_read_tokens) {}
```

In `TransferEngineImpl::updateRuntimeQueueMetrics()`:

```cpp
const auto scheduler = runtime_queue_->gdsSchedulerSnapshot();
gds_transport->updateRuntimeQueueDepth(
    queued_gds_reads, queued_gds_writes,
    scheduler.reserved_tokens[0]);
```

Store the value in `GdsTransport::runtime_reserved_read_tokens_`. Define READ
pressure as runtime queued READs, runtime reserved READ tokens, transport
pending READs, or transport inflight READs.

- [ ] **Step 6: Use one locked decision path**

Add a private `writeDispatchDecisionLocked()` method. Use it from:

- `runtimeQueueDispatchLimit(WRITE)`;
- `dispatchPendingIoLocked()`;
- `maybeLogIoSummaryLocked()`.

This avoids the current misleading assignment:

```cpp
runtime_dispatch_limit = write_adaptive_.current_limit
```

The summary prints configured, current, runtime, READ pressure, pause policy,
and reserved READ tokens separately.

- [ ] **Step 7: Count bounded block reasons**

When dispatch stops, increment fixed-array interval counters for:
shared limit, READ/WRITE direction limit, READ pause, missing worker pool, and
FIFO front. Reset them with the one-second IO summary. Do not add dynamic
labels or one log per IO.

- [ ] **Step 8: Run runtime and FIFO tests**

Run:

```bash
cmake --build build --target tent_gds_fifo_dispatch_test tent_runtime_queue_dispatch_test -j
ctest --test-dir build -R 'tent_gds_fifo_dispatch_test|tent_runtime_queue_dispatch_test' --output-on-failure
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add mooncake-transfer-engine/tent/include/tent/transport/gds/gds_fifo_dispatch.h mooncake-transfer-engine/tent/tests/gds_fifo_dispatch_test.cpp mooncake-transfer-engine/tent/include/tent/runtime/transport.h mooncake-transfer-engine/tent/include/tent/transport/gds/gds_transport.h mooncake-transfer-engine/tent/src/transport/gds/gds_transport.cpp mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp mooncake-transfer-engine/tent/tests/runtime_queue_dispatch_test.cpp
git commit -m "fix: expose effective GDS write dispatch limits"
```

---

### Task 3: Opt-in WRITE=4 target configuration

**Files:**
- Create: `mooncake-transfer-engine/tent/config/tent-gds-write4.json`
- Modify: `benchmarks/test_gds_baseline_collector.py`

**Interfaces:**
- Produces one explicit target-validation config; does not change production
  defaults.

- [ ] **Step 1: Add a failing configuration test**

Keep the existing baseline assertions at WRITE=1. Add:

```python
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
```

- [ ] **Step 2: Run the Python test and verify RED**

Run:

```bash
python -m pytest benchmarks/test_gds_baseline_collector.py -q
```

Expected: FAIL with `FileNotFoundError` for `tent-gds-write4.json`.

- [ ] **Step 3: Create the target configuration**

Copy the fixed baseline structure, changing only:

```json
"max_dispatch_write_owners": 4,
"gds_write_standalone_tokens": 4,
"max_inflight_writes": 4
```

Retain `write_worker_threads=4`, `shared_device_tokens=16`,
`gds_contended_write_tokens=1`,
`gds_pause_writes_while_reads_pending=true`,
`adaptive_concurrency=false`, `batch_api=false`, and `async_api=false`.

- [ ] **Step 4: Run the Python test and verify GREEN**

Run the Task 3 command again. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mooncake-transfer-engine/tent/config/tent-gds-write4.json benchmarks/test_gds_baseline_collector.py
git commit -m "feat: add opt-in four-write GDS config"
```

---

### Task 4: Lossless BatchGet route metrics

**Files:**
- Modify: `mooncake-store/include/client_metric.h`
- Modify: `mooncake-store/src/client_metric.cpp`
- Modify: `mooncake-store/include/client_service.h`
- Modify: `mooncake-store/src/real_client.cpp`
- Modify: `mooncake-store/tests/client_metrics_test.cpp`

**Interfaces:**
- Produces:

```cpp
struct BatchGetLookupObservation {
    uint64_t calls{0};
    uint64_t requested_keys{0};
    uint64_t metadata_hit_keys{0};
    uint64_t metadata_miss_keys{0};
    uint64_t query_error_keys{0};
    uint64_t available_memory_keys{0};
    uint64_t available_gds_keys{0};
    uint64_t available_local_keys{0};
    uint64_t selected_memory_keys{0};
    uint64_t selected_gds_keys{0};
    uint64_t selected_local_keys{0};
    uint64_t skipped_gds_due_memory_keys{0};
    uint64_t skipped_gds_due_local_keys{0};
    uint64_t gds_submit_attempt_keys{0};
    uint64_t gds_submit_failure_keys{0};
};
```

- Exports fixed-name cumulative Prometheus counters with the
  `mooncake_batch_get_` prefix.

- [ ] **Step 1: Add failing metric serialization and classification tests**

Construct an observation where availability is multi-label but selected route
is mutually exclusive:

```cpp
BatchGetLookupObservation observation{
    .calls = 1,
    .requested_keys = 4,
    .metadata_hit_keys = 3,
    .metadata_miss_keys = 1,
    .available_memory_keys = 2,
    .available_gds_keys = 2,
    .selected_memory_keys = 1,
    .selected_gds_keys = 1,
    .selected_local_keys = 1,
    .skipped_gds_due_memory_keys = 1,
    .gds_submit_attempt_keys = 1,
};
metrics.ObserveBatchGetLookup(observation);
```

Assert literal counter values and serialization names. Assert serialized text
does not contain `segment_uri`, object keys, or dynamic route labels.

- [ ] **Step 2: Run Store metrics tests and verify RED**

Run:

```bash
cmake --build build --target client_metrics_test -j
ctest --test-dir build -R client_metrics_test --output-on-failure
```

Expected: compile failure because the observation API does not exist.

- [ ] **Step 3: Implement fixed counters in `ClientMetric`**

Add `BatchGetLookupMetric` containing one `ylt::metric::counter_t` per field.
Implement `Observe`, `serialize`, and a compact cumulative summary. Add it to
`ClientMetric::serialize()` and `summary_metrics()`. Add forwarding:

```cpp
void ClientService::ObserveBatchGetLookup(
    const BatchGetLookupObservation& observation) {
    if (metrics_) metrics_->ObserveBatchGetLookup(observation);
}
```

- [ ] **Step 4: Record metadata availability and selected route**

In both `batch_get_into_internal` implementations:

- initialize `calls=1` and `requested_keys=keys.size()`;
- classify `OBJECT_NOT_FOUND` and `REPLICA_IS_NOT_READY` as metadata miss,
  and other failed queries as query error;
- for successful queries, scan every COMPLETE replica and increment each
  availability counter at most once per key;
- increment exactly one selected-route counter only after buffer validation;
- increment a skipped-GDS reason only when GDS was available and a non-GDS
  route was selected;
- tag each valid operation with its selected route;
- before `Client::BatchGet`, add selected GDS keys to submit attempts;
- after the result, count failed GDS keys by matching the tagged operation;
- emit the observation once on every post-query return path.

Do not infer selected route from availability.

- [ ] **Step 5: Run Store metrics tests and verify GREEN**

Run the Task 4 command again. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mooncake-store/include/client_metric.h mooncake-store/src/client_metric.cpp mooncake-store/include/client_service.h mooncake-store/src/real_client.cpp mooncake-store/tests/client_metrics_test.cpp
git commit -m "feat: expose lossless BatchGet route metrics"
```

---

### Task 5: Collector support and abnormal reservation coverage

**Files:**
- Modify: `benchmarks/gds_baseline_collector.py`
- Modify: `benchmarks/test_gds_baseline_collector.py`
- Modify: `mooncake-transfer-engine/tent/tests/gds_operation_scheduler_test.cpp`
- Modify: `mooncake-transfer-engine/tent/tests/runtime_queue_dispatch_test.cpp`

**Interfaces:**
- Adds `--store-metrics-endpoint` for Prometheus text and emits parsed
  cumulative counters in the `store` sample source.
- Strengthens terminal reconciliation tests without changing public APIs.

- [ ] **Step 1: Add failing Prometheus parser tests**

Use a literal fixture:

```python
payload = """
mooncake_batch_get_calls_total 2
mooncake_batch_get_requested_keys_total 80
mooncake_batch_get_selected_gds_keys_total 12
"""
assert parse_prometheus_counters(payload)["mooncake_batch_get_calls_total"] == 2
```

Also assert comments, labeled unrelated metrics, malformed values, and missing
endpoints do not fabricate numeric zero.

- [ ] **Step 2: Implement Store endpoint collection**

Add `_collect_prometheus_endpoint`, `--store-metrics-endpoint`, and source
`store`. Preserve source start/finish timestamps and explicit
`available:false` behavior.

- [ ] **Step 3: Add terminal-status table and next-dispatch assertions**

For `FAILED`, `CANCELED`, and `TIMEOUT`, reserve one token, complete it with
zero or partial bytes, assert all reservation fields return to zero, then
enqueue/select another owner. Keep the existing duplicate-completion assertion
and verify it does not underflow accounting.

Extend fake-transport runtime tests so immediate submit failure and engine
shutdown leave no dispatch-inflight owner and allow a later fresh engine/queue
to dispatch. Reuse the existing fake transport; do not mock cuFile.

- [ ] **Step 4: Run affected tests**

Run:

```bash
python -m pytest benchmarks/test_gds_baseline_collector.py -q
cmake --build build --target tent_gds_operation_scheduler_test tent_runtime_queue_dispatch_test -j
ctest --test-dir build -R 'tent_gds_operation_scheduler_test|tent_runtime_queue_dispatch_test' --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/gds_baseline_collector.py benchmarks/test_gds_baseline_collector.py mooncake-transfer-engine/tent/tests/gds_operation_scheduler_test.cpp mooncake-transfer-engine/tent/tests/runtime_queue_dispatch_test.cpp
git commit -m "test: cover GDS diagnostics and terminal reconciliation"
```

---

### Task 6: Full local verification and target-host handoff

**Files:**
- Modify only if verification exposes a defect in files already listed above.

- [ ] **Step 1: Run Python regressions**

```bash
python -m pytest benchmarks/test_gds_baseline_collector.py benchmarks/test_gds_lease_refresh_probe.py benchmarks/test_vllm_warmup.py -q
```

- [ ] **Step 2: Run affected C++ regressions**

```bash
cmake --build build --target tent_gds_operation_scheduler_test tent_gds_fifo_dispatch_test tent_runtime_queue_dispatch_test client_metrics_test -j
ctest --test-dir build -R 'tent_gds_operation_scheduler_test|tent_gds_fifo_dispatch_test|tent_runtime_queue_dispatch_test|client_metrics_test' --output-on-failure
```

- [ ] **Step 3: Run static checks**

```bash
git diff --check
python -m json.tool mooncake-transfer-engine/tent/config/tent-gds-write4.json
```

- [ ] **Step 4: Prepare the target command**

Use:

```bash
export TENT_CONFIG_PATH=/mnt/extra_new/l50058871/Mooncake/mooncake-transfer-engine/tent/config/tent-gds-write4.json
export TAKE_OVER_EXISTING=1
```

With those variables, launch the same 80-client, 10-turn runner that produced
the supplied statistics. Collect client Store metrics from
`http://127.0.0.1:9300/metrics` and TENT metrics from the configured TENT
endpoint.

- [ ] **Step 5: Apply target-host gates**

Run four independent WRITE owners with sustained non-overlapping backlog at
limits 1, 2, and 4. Require:

```text
throughput_2 >= 1.5 * throughput_1
throughput_4 >= 1.25 * throughput_2
cuFile failures = 0
all reserved and inflight counters = 0 after drain
mixed READ throughput >= 0.95 * baseline
mixed READ p99 <= 1.10 * baseline
submitted_requests = completed_http_200 = 800
client_timeouts = server_5xx = cancelled = 0
max_input_num_turns = 10
```

Do not promote the fixed/default config if any gate fails. Return the
WRITE=1/2/4 summaries and one-second GDS/Store metrics for the next diagnosis.
