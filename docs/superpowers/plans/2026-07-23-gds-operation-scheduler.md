# GDS Operation Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace duplicate READ-first scheduling with one operation-level runtime scheduler that uses completed-byte WDRR plus outstanding reservations, fills a primary BatchGet operation across bounded segments, and ships safe fixed-concurrency defaults and lease-validation tooling.

**Architecture:** `LocalTransferAdmissionQueue` remains the owner/status store, while a new pure `GdsOperationScheduler` owns only GDS selection, operation windows, WDRR rounds, and reservation reconciliation. `TransferEngineImpl` asks GDS for a side-effect-free physical plan, dispatches bounded same-operation groups, and completes reservations with actual bytes; GDS consumes selected work in FIFO order under shared physical tokens and never applies a second READ/WRITE policy.

**Tech Stack:** C++17, GoogleTest, TENT runtime/transport interfaces, cuFile synchronous single-IO APIs, Mooncake Store Master tests, Python 3 benchmark collectors, JSON configuration.

## Global Constraints

- `batch_token` is an operation-level owner and must never be logged or named as a conversation identifier.
- cuFile Batch stays disabled; production requests never probe Batch or Async.
- Baseline defaults are READ 16 workers/16 inflight, WRITE 4 workers/1 inflight, adaptive disabled.
- Weighted-fair shared physical tokens are 16; READ:WRITE WDRR is 4:1 by actual completed physical bytes.
- Positive spendable credit is capped at two direction quanta and idle directions cannot accumulate credit.
- Runtime is the only READ/WRITE policy scheduler; GDS enforces resources and consumes runtime order.
- Segment construction stops before the first request that would exceed 8 original requests or 16 MiB.
- A primary READ operation may reserve at most 16 physical tokens and 48 MiB; a secondary starts only when the primary cannot use otherwise idle capacity.
- Indirect lease refresh through `BatchGetReplicaList` is disabled by default until expiry, group-amplification, metric-side-effect, and Master-load gates pass.
- Existing user-staged `.agents/skills/**` files are out of scope and must not enter any implementation commit.

---

### Task 1: Pure completed-byte WDRR reservation policy

**Files:**
- Create: `mooncake-transfer-engine/tent/include/tent/runtime/gds_operation_scheduler.h`
- Create: `mooncake-transfer-engine/tent/src/runtime/gds_operation_scheduler.cpp`
- Create: `mooncake-transfer-engine/tent/tests/gds_operation_scheduler_test.cpp`
- Modify: `mooncake-transfer-engine/tent/tests/CMakeLists.txt`

**Interfaces:**
- Produces: `GdsOperationSchedulerConfig`, `GdsDispatchEntry`, `GdsDispatchReservation`, `GdsOperationScheduler::select`, `GdsOperationScheduler::complete`, and `GdsOperationScheduler::cancelOperation`.
- `select` consumes queued immutable entries plus global owner/byte budgets and returns reservations in runtime dispatch order.
- `complete(reservation_id, actual_bytes, terminal_status)` releases exactly one reservation and reconciles actual completed bytes.

- [ ] **Step 1: Write failing reservation and WDRR tests**

```cpp
TEST(GdsOperationScheduler, OutstandingReservationPreventsFreeResubmit) {
    constexpr size_t kMiB = 1UL << 20;
    GdsOperationScheduler scheduler(weightedConfig());
    scheduler.enqueue(readEntry(1, 101, 8 * kMiB));
    scheduler.enqueue(readEntry(1, 102, 8 * kMiB));
    scheduler.enqueue(writeEntry(2, 201, 2 * kMiB));
    auto first = scheduler.select({16, 64 * kMiB});
    ASSERT_FALSE(first.empty());
    EXPECT_LE(scheduler.snapshot().reserved_bytes[READ_INDEX], 16 * kMiB);
    auto without_completion = scheduler.select({16, 64 * kMiB});
    EXPECT_TRUE(without_completion.empty());
}

TEST(GdsOperationScheduler, PartialCompletionRefundsUnusedReservation) {
    constexpr size_t kMiB = 1UL << 20;
    GdsOperationScheduler scheduler(weightedConfig());
    scheduler.enqueue(readEntry(1, 101, 8 * kMiB));
    auto reservation = scheduler.select({16, 64 * kMiB}).front();
    ASSERT_TRUE(scheduler.complete(reservation.id, 3 * kMiB, FAILED).ok());
    EXPECT_EQ(scheduler.snapshot().completed_bytes[READ_INDEX], 3 * kMiB);
    EXPECT_EQ(scheduler.snapshot().reserved_bytes[READ_INDEX], 0u);
}
```

- [ ] **Step 2: Build the new test and confirm the missing-header failure**

Run: `cmake --build build --target tent_gds_operation_scheduler_test -j`

Expected: compilation fails because `tent/runtime/gds_operation_scheduler.h` does not exist.

- [ ] **Step 3: Implement the scheduler state and checked reservation ledger**

```cpp
struct GdsDirectionState {
    int64_t deficit_bytes{0};
    size_t outstanding_reserved_bytes{0};
    size_t outstanding_reserved_tokens{0};
    size_t completed_bytes{0};
    uint64_t last_granted_round{0};
};

int64_t GdsOperationScheduler::spendable(Direction d) const {
    const auto& s = directions_[index(d)];
    return s.deficit_bytes -
           static_cast<int64_t>(s.outstanding_reserved_bytes);
}

Status GdsOperationScheduler::complete(uint64_t id, size_t actual,
                                       TransferStatusEnum status) {
    auto it = reservations_.find(id);
    if (it == reservations_.end() || it->second.reconciled)
        return Status::InvalidEntry("duplicate or missing reservation" LOC_MARK);
    if (actual > it->second.bytes)
        return Status::InvalidArgument("completion exceeds reservation" LOC_MARK);
    auto& direction = directions_[index(it->second.direction)];
    direction.outstanding_reserved_bytes -= it->second.bytes;
    direction.outstanding_reserved_tokens -= it->second.tokens;
    direction.deficit_bytes -= static_cast<int64_t>(actual);
    direction.completed_bytes += actual;
    it->second.reconciled = true;
    return Status::OK();
}
```

- [ ] **Step 4: Add unequal-size, idle-credit, global-round, oversized-head, duplicate-completion, and underflow tests**

Run: `ctest --test-dir build -R tent_gds_operation_scheduler_test --output-on-failure`

Expected: all scheduler tests pass, including 2.25 MiB versus 15 MiB actual-byte accounting.

- [ ] **Step 5: Commit the isolated policy**

```bash
git add mooncake-transfer-engine/tent/include/tent/runtime/gds_operation_scheduler.h mooncake-transfer-engine/tent/src/runtime/gds_operation_scheduler.cpp mooncake-transfer-engine/tent/tests/gds_operation_scheduler_test.cpp mooncake-transfer-engine/tent/tests/CMakeLists.txt
git commit -m "feat: add reservation-aware GDS operation scheduler"
```

### Task 2: Operation-focused queue selection and bounded multi-segment windows

**Files:**
- Modify: `mooncake-transfer-engine/tent/include/tent/runtime/admission_queue.h`
- Modify: `mooncake-transfer-engine/tent/src/runtime/admission_queue.cpp`
- Modify: `mooncake-transfer-engine/tent/tests/admission_queue_test.cpp`
- Modify: `mooncake-transfer-engine/tent/include/tent/runtime/gds_operation_scheduler.h`
- Modify: `mooncake-transfer-engine/tent/src/runtime/gds_operation_scheduler.cpp`

**Interfaces:**
- Consumes: Task 1 scheduler.
- Produces: `QueuePhysicalPlan { physical_ios, physical_bytes }`, `QueueDispatchBudget`, and `LocalTransferAdmissionQueue::complete(owner_id, actual_bytes, status)`.

- [ ] **Step 1: Add failing operation-focus tests**

```cpp
TEST(AdmissionQueue, OneReadOperationFillsSixteenTokensAcrossSegments) {
    constexpr size_t kMiB = 1UL << 20;
    LocalTransferAdmissionQueue queue(limits(), weightedSchedulerConfig());
    admitOwners(queue, 77, Request::READ, 32, 2359296, 1);
    auto first = queue.pickForDispatch(dispatchBudget(16, 48 * kMiB));
    ASSERT_EQ(first.size(), 16u);
    EXPECT_TRUE(std::all_of(first.begin(), first.end(),
        [&](QueueOwnerId id) { return operationOf(queue, id) == 77; }));
    EXPECT_EQ(segmentCount(first, 8, 16 * kMiB), 3u);
}

TEST(AdmissionQueue, DoesNotSpreadWhilePrimaryCanDispatch) {
    constexpr size_t kMiB = 1UL << 20;
    LocalTransferAdmissionQueue queue(limits(), weightedSchedulerConfig());
    admitOwners(queue, 77, Request::READ, 32, 2359296, 1);
    admitOwners(queue, 88, Request::READ, 32, 2359296, 1);
    auto picked = queue.pickForDispatch(dispatchBudget(16, 48 * kMiB));
    EXPECT_TRUE(allBelongTo(picked, queue, 77));
}
```

- [ ] **Step 2: Run admission tests and verify the old READ-first picker fails operation focus**

Run: `cmake --build build --target admission_queue_test -j && ctest --test-dir build -R admission_queue_test --output-on-failure`

Expected: new tests fail because the queue alternates by FIFO/READ preference and lacks reservations.

- [ ] **Step 3: Store physical plans per owner and delegate GDS selection to the scheduler**

```cpp
struct QueueOwnerInput {
    size_t owner_task_id{0};
    std::vector<size_t> derived_task_ids;
    Request request{};
    TransportType transport{UNSPEC};
    QueueOwnerKind kind{QueueOwnerKind::User};
    QueuePhysicalPlan physical_plan{1, 0};
};
```

Non-GDS owners retain FIFO selection. GDS entries use the pure scheduler, and the queue marks every returned reservation `Dispatching` atomically.

- [ ] **Step 4: Cover exact secondary-operation activation**

Add tests where the primary has no ready work, hits 48 MiB while tokens remain, has only one segment inflight, and is canceled. Only the first two cases may activate operation two.

Run: `ctest --test-dir build -R 'admission_queue_test|tent_gds_operation_scheduler_test' --output-on-failure`

Expected: both targets pass; a 192-owner operation contributes no more than the token/byte window.

- [ ] **Step 5: Commit operation-focused admission**

```bash
git add mooncake-transfer-engine/tent/include/tent/runtime/admission_queue.h mooncake-transfer-engine/tent/src/runtime/admission_queue.cpp mooncake-transfer-engine/tent/tests/admission_queue_test.cpp mooncake-transfer-engine/tent/include/tent/runtime/gds_operation_scheduler.h mooncake-transfer-engine/tent/src/runtime/gds_operation_scheduler.cpp
git commit -m "feat: schedule bounded GDS operation windows"
```

### Task 3: Physical planning and grouped runtime segment dispatch

**Files:**
- Modify: `mooncake-transfer-engine/tent/include/tent/runtime/transport.h`
- Modify: `mooncake-transfer-engine/tent/include/tent/transport/gds/gds_transport.h`
- Modify: `mooncake-transfer-engine/tent/src/transport/gds/gds_transport.cpp`
- Modify: `mooncake-transfer-engine/tent/include/tent/runtime/transfer_engine_impl.h`
- Modify: `mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp`
- Modify: `mooncake-transfer-engine/tent/tests/runtime_queue_dispatch_test.cpp`

**Interfaces:**
- Produces: `Transport::RuntimeQueuePlan`, `Transport::planRuntimeQueueRequest`, and exact actual-byte completion into Task 2.
- A GDS dispatch group contains one batch token, one opcode, at most 8 original owners, and at most 16 MiB.

- [ ] **Step 1: Replace single-owner expectations with failing bounded-group tests**

```cpp
TEST(RuntimeQueueDispatch, GroupsOneOperationIntoBoundedGdsSegments) {
    constexpr size_t kMiB = 1UL << 20;
    submitReadOperation(engine, 32, 2359296);
    EXPECT_EQ(fake_gds->submitted_requests.load(), 16u);
    EXPECT_EQ(fake_gds->submit_calls.load(), 3u);
    EXPECT_LE(fake_gds->max_requests_per_submit.load(), 8u);
    EXPECT_LE(fake_gds->max_bytes_per_submit.load(), 16 * kMiB);
}
```

Add a second batch and assert no request from the second operation is in the initial 16-token window.

- [ ] **Step 2: Run the runtime dispatch test and verify old one-request submits fail**

Run: `cmake --build build --target tent_runtime_queue_dispatch_test -j && ctest --test-dir build -R tent_runtime_queue_dispatch_test --output-on-failure`

Expected: bounded-group tests fail with `submit_calls == submitted_requests`.

- [ ] **Step 3: Add side-effect-free physical plan API**

```cpp
struct RuntimeQueuePlan {
    size_t physical_ios{1};
    size_t physical_bytes{0};
};

virtual Status planRuntimeQueueRequest(const Request& request,
                                       RuntimeQueuePlan& plan) {
    plan = {1, request.length};
    return Status::OK();
}
```

GDS computes the same registered/unregistered split size used by `submitTransferTasks`. It validates that the submitted expansion equals the admitted plan before mutating pending queues.

- [ ] **Step 4: Group selected GDS owners by operation/opcode and reconcile actual bytes**

Change `finishQueuedOwner` to accept `size_t actual_transferred_bytes`. Pass `task_status.transferred_bytes` from every terminal poll and pass zero on pre-submit failure. Preserve non-GDS grouping behavior.

- [ ] **Step 5: Run runtime, admission, failover, and GDS planning tests**

Run: `ctest --test-dir build -R 'tent_runtime_queue_dispatch_test|admission_queue_test|tent_engine_failover_e2e_test|tent_gds_block_io_test' --output-on-failure`

Expected: all selected tests pass with GDS `max_requests_per_submit <= 8`, no cuFile Batch call, and exact partial-byte reconciliation.

- [ ] **Step 6: Commit runtime integration**

```bash
git add mooncake-transfer-engine/tent/include/tent/runtime/transport.h mooncake-transfer-engine/tent/include/tent/transport/gds/gds_transport.h mooncake-transfer-engine/tent/src/transport/gds/gds_transport.cpp mooncake-transfer-engine/tent/include/tent/runtime/transfer_engine_impl.h mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp mooncake-transfer-engine/tent/tests/runtime_queue_dispatch_test.cpp
git commit -m "feat: dispatch bounded GDS operation segments"
```

### Task 4: Remove the second GDS scheduling policy and enforce shared tokens

**Files:**
- Modify: `mooncake-transfer-engine/tent/include/tent/transport/gds/gds_transport.h`
- Modify: `mooncake-transfer-engine/tent/src/transport/gds/gds_transport.cpp`
- Modify: `mooncake-transfer-engine/tent/tests/gds_block_io_test.cpp`

**Interfaces:**
- Consumes: runtime-selected FIFO order from Task 3.
- Produces: a unified pending FIFO, `shared_physical_tokens`, fixed direction caps, active-worker metrics, and WRITE boost hysteresis resource limits without direction selection.

- [ ] **Step 1: Add failing shared-token and FIFO tests**

```cpp
TEST(GdsResourcePolicy, RuntimeOrderIsPreserved) {
    GdsResourcePolicy policy(configWithSharedTokens(16));
    policy.enqueue(WRITE, 1);
    policy.enqueue(READ, 2);
    EXPECT_EQ(policy.next().sequence, 1u);
}

TEST(GdsResourcePolicy, ContendedWriteUsesOneToken) {
    auto limits = effectiveLimits(/*read_backlog=*/true,
                                  /*write_backlog=*/true,
                                  WritePressureState::NORMAL);
    EXPECT_EQ(limits.write, 1u);
    EXPECT_EQ(limits.shared, 16u);
}
```

- [ ] **Step 2: Run the GDS test and verify READ-first behavior fails FIFO**

Run: `cmake --build build --target tent_gds_block_io_test -j && ctest --test-dir build -R tent_gds_block_io_test --output-on-failure`

Expected: FIFO test fails because `dispatchPendingIoLocked` selects READ first.

- [ ] **Step 3: Replace direction deques with one runtime-ordered pending deque**

Keep separate worker pools and inflight maps. Dispatch the head only when its direction cap, shared token, and worker slot are available; do not bypass a blocked head with the other direction.

- [ ] **Step 4: Make fixed mode the default and add WRITE hysteresis limits**

Adaptive concurrency remains compiled but disabled in fixed and weighted modes. WRITE-only may use configured limit 2; contended mode uses 1, promotes to 2 only after three one-second pressure windows, demotes after five healthy windows or READ P99 above 75 ms, then enters ten-second cooldown.

- [ ] **Step 5: Run GDS unit and target-device tests**

Run: `ctest --test-dir build -R tent_gds_block_io_test --output-on-failure`

Expected: pure policy tests pass. On a CUDA/GDS host, complete and partial synchronous single-IO integration cases pass without `CU_FILE_INTERNAL_ERROR(5030)`.

- [ ] **Step 6: Commit GDS resource enforcement**

```bash
git add mooncake-transfer-engine/tent/include/tent/transport/gds/gds_transport.h mooncake-transfer-engine/tent/src/transport/gds/gds_transport.cpp mooncake-transfer-engine/tent/tests/gds_block_io_test.cpp
git commit -m "fix: consume runtime-selected GDS work under shared tokens"
```

### Task 5: Operation timeline and scheduler observability

**Files:**
- Modify: `mooncake-transfer-engine/tent/include/tent/metrics/tent_metrics.h`
- Modify: `mooncake-transfer-engine/tent/src/metrics/tent_metrics.cpp`
- Modify: `mooncake-transfer-engine/tent/include/tent/runtime/transfer_engine_impl.h`
- Modify: `mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp`
- Modify: `mooncake-transfer-engine/tent/tests/runtime_queue_dispatch_test.cpp`

**Interfaces:**
- Produces: structured operation start/periodic/terminal logs and gauges/counters for waiting operations, active operations, reserved bytes/tokens, completed bytes, queue wait, actual workers, cuFile latency, and total latency.

- [ ] **Step 1: Add a failing exactly-once terminal timeline test**

Create a 16-key fake GDS BatchGet, complete it through three transport groups, poll twice, and assert one start record, one terminal record, 16 completed requests, exact logical/physical bytes, and `distinct_conversation_count=unknown`.

- [ ] **Step 2: Implement `RuntimeOperationTimeline` keyed by `batch_token`**

Record queue-enter, first-dispatch, first-completion, terminal time, segment count, request counts, reserved/completed/failed bytes, and terminal cause. Erase only after the terminal log and reservation count reach zero.

- [ ] **Step 3: Add metrics and startup fingerprint**

The startup log includes scheduler mode, all token/operation/segment limits, adaptive/Batch/Async/merge states, canonical config path when available, and environment allowlist values without secrets.

- [ ] **Step 4: Run metrics and runtime tests**

Run: `ctest --test-dir build -R 'tent_runtime_queue_dispatch_test|metrics_config_loader_test' --output-on-failure`

Expected: tests pass and repeated polling does not duplicate terminal metrics.

- [ ] **Step 5: Commit observability**

```bash
git add mooncake-transfer-engine/tent/include/tent/metrics/tent_metrics.h mooncake-transfer-engine/tent/src/metrics/tent_metrics.cpp mooncake-transfer-engine/tent/include/tent/runtime/transfer_engine_impl.h mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp mooncake-transfer-engine/tent/tests/runtime_queue_dispatch_test.cpp
git commit -m "feat: trace GDS operations and reservations"
```

### Task 6: Lease safety tests and disabled indirect-refresh experiment

**Files:**
- Modify: `mooncake-store/tests/master_service_test.cpp`
- Modify: `mooncake-store/tests/master_metrics_test.cpp`
- Create: `benchmarks/gds_lease_refresh_probe.py`
- Create: `benchmarks/test_gds_lease_refresh_probe.py`

**Interfaces:**
- Produces: repeatable proof that `BatchGetReplicaList` uses a bounded timestamp lease, measures grouped-key amplification and metric side effects, and refuses to recommend indirect refresh when load gates fail.

- [ ] **Step 1: Add short-TTL Master tests**

```cpp
TEST_F(MasterServiceTest, RepeatedBatchGetLeaseExpiresAfterLastRefresh) {
    auto config = MasterServiceConfig::builder()
                      .set_default_kv_lease_ttl(40)
                      .build();
    MasterService service(config);
    [[maybe_unused]] const auto segment = PrepareSimpleSegment(service);
    const UUID client_id = generate_uuid();
    ReplicateConfig replica_config;
    replica_config.replica_num = 1;
    PutCompletedObject(service, client_id, "lease-key", replica_config);
    ASSERT_TRUE(service.BatchGetReplicaList(
                            client_id, {"lease-key"}, "default")[0]
                    .has_value());
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    ASSERT_TRUE(service.BatchGetReplicaList(
                            client_id, {"lease-key"}, "default")[0]
                    .has_value());
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    EXPECT_TRUE(service.Remove("lease-key", "default").has_value());
}
```

Add a grouped-key test that records extended member count/bytes and a metrics test showing refresh traffic changes Get/cache-hit counters.

- [ ] **Step 2: Run Master tests and document observed side effects**

Run: `cmake --build build --target master_service_test master_metrics_test -j && ctest --test-dir build -R 'master_service_test|master_metrics_test' --output-on-failure`

Expected: expiry is bounded by one TTL after the last call; grouped refresh and Get metrics are observable.

- [ ] **Step 3: Implement the load-probe evaluator**

```python
def evaluate_refresh_probe(sample: dict) -> dict:
    reasons = []
    if sample["master_cpu_delta_pct"] >= 5.0:
        reasons.append("master_cpu")
    if sample["batch_get_p99_delta_pct"] >= 10.0:
        reasons.append("batch_get_p99")
    if sample["post_terminal_refreshes"] != 0:
        reasons.append("post_terminal_refresh")
    return {"passed": not reasons, "reasons": reasons}
```

The CLI accepts baseline and refresh JSONL files, validates an 80-operation 126/192-key run, and exits nonzero unless every gate passes.

- [ ] **Step 4: Test the evaluator**

Run: `python -m pytest benchmarks/test_gds_lease_refresh_probe.py -q`

Expected: passing sample exits zero; CPU, P99, post-terminal, and expiry violations each fail with a distinct reason.

- [ ] **Step 5: Commit lease validation**

```bash
git add mooncake-store/tests/master_service_test.cpp mooncake-store/tests/master_metrics_test.cpp benchmarks/gds_lease_refresh_probe.py benchmarks/test_gds_lease_refresh_probe.py
git commit -m "test: gate indirect GDS lease refresh"
```

### Task 7: Reproducible baseline/weighted configurations and synchronized collector

**Files:**
- Create: `mooncake-transfer-engine/tent/config/tent-gds-baseline.json`
- Create: `mooncake-transfer-engine/tent/config/tent-gds-weighted.json`
- Modify: `mooncake-transfer-engine/tent/config/tent-gds.json`
- Create: `benchmarks/gds_baseline_collector.py`
- Create: `benchmarks/test_gds_baseline_collector.py`
- Modify: `benchmarks/vllm_warmup.py`
- Modify: `benchmarks/test_vllm_warmup.py`

**Interfaces:**
- Produces: fixed baseline defaults, opt-in weighted mode, timestamped JSONL collection, config SHA256/executable identity, and a warmup completion gate that runs required Triton kernels before load.

- [ ] **Step 1: Add failing configuration and collector tests**

Assert baseline WRITE workers/inflight are 4/1, adaptive false, Batch/Async false; weighted shared tokens are 16; missing `nvidia-smi` or block metrics emit `available:false` instead of zero.

- [ ] **Step 2: Write exact configuration modes**

`tent-gds.json` and `tent-gds-baseline.json` use fixed mode with READ 16/16 and WRITE 4/1. `tent-gds-weighted.json` enables completed-byte WDRR, primary operation focus, merge shadow, and keeps adaptive/Batch/Async disabled.

- [ ] **Step 3: Implement synchronized JSONL collection**

Each sample contains one monotonic timestamp and independently marked GPU, NVMe, vLLM, runtime, KV restore, and inference fields. The header contains canonical config path/SHA256 and executable/build identity.

- [ ] **Step 4: Keep warmup from returning before compile coverage**

Extend the existing warmup manifest check so `_compute_slot_mapping_kernel` coverage is mandatory when the vLLM connector exposes it. A missing required kernel fails warmup rather than starting the formal benchmark.

- [ ] **Step 5: Run Python tests**

Run: `python -m pytest benchmarks/test_gds_baseline_collector.py benchmarks/test_vllm_warmup.py -q`

Expected: all tests pass without NVIDIA/NVMe tools installed, using explicit unavailable records.

- [ ] **Step 6: Commit reproducible operating modes**

```bash
git add mooncake-transfer-engine/tent/config/tent-gds.json mooncake-transfer-engine/tent/config/tent-gds-baseline.json mooncake-transfer-engine/tent/config/tent-gds-weighted.json benchmarks/gds_baseline_collector.py benchmarks/test_gds_baseline_collector.py benchmarks/vllm_warmup.py benchmarks/test_vllm_warmup.py
git commit -m "feat: add reproducible GDS baseline tooling"
```

### Task 8: Full regression and target-host handoff

**Files:**
- Modify: `docs/source/design/gds-ssd/spec.md`
- Modify: `docs/superpowers/specs/2026-07-23-gds-operation-scheduler-design.md`

**Interfaces:**
- Consumes: Tasks 1-7.
- Produces: verified documentation and a target-host command sequence with explicit hardware-only gates.

- [ ] **Step 1: Run formatting and static checks**

Run: `git diff --check && cmake --build build --target format-check -j`

Expected: no whitespace errors and no formatting violations.

- [ ] **Step 2: Run all affected C++ tests**

Run: `ctest --test-dir build -R 'gds|admission_queue|runtime_queue|master_service|master_metrics|failover|metrics_config' --output-on-failure`

Expected: all affected host-independent tests pass.

- [ ] **Step 3: Run all affected Python tests**

Run: `python -m pytest benchmarks/test_gds_baseline_collector.py benchmarks/test_gds_lease_refresh_probe.py benchmarks/test_vllm_warmup.py -q`

Expected: all tests pass.

- [ ] **Step 4: Run the CUDA/GDS target matrix**

Run baseline, weighted, WRITE-only inflight 1/2/3/4, and READ token/operation-window scans with the same 80-session workload. Reject any result with 5030, wholesale 126/192-key failure, non-draining backlog, data mismatch, lower BatchGet completion rate, or lower generation throughput.

- [ ] **Step 5: Update documentation with verified defaults and gates**

Document that merge remains shadow, Async remains startup-probe-only and disabled, indirect lease refresh remains disabled until the probe passes, and adaptive replacement remains disabled until synchronized baseline data exists.

- [ ] **Step 6: Commit documentation**

```bash
git add docs/source/design/gds-ssd/spec.md docs/superpowers/specs/2026-07-23-gds-operation-scheduler-design.md
git commit -m "docs: document operation-level GDS scheduling"
```
