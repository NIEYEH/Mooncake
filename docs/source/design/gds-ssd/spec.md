# GDS SSD Operation Scheduling

## Scope

This document describes the runtime scheduling contract for TENT GDS access
to raw block or NVMe-oF storage. The checked-in defaults are intentionally
conservative: they establish a reproducible baseline before enabling
weighted-fair scheduling or increasing physical concurrency.

`batch_token` is an operation-level owner identifier. It may cover one
BatchGet or BatchPut operation, but it is not treated as a conversation
identifier. Operation logs report distinct upstream request and conversation
counts as `unknown` until explicit correlation identifiers are available.

## Data-path contract

- cuFile Batch and Async APIs are disabled. Production requests use only
  synchronous, single-entry `cuFileRead` and `cuFileWrite` calls.
- READ and WRITE use separate pre-created worker pools.
- The runtime queue is the only direction and operation policy scheduler.
  GDS consumes runtime-selected work in FIFO order and enforces worker,
  direction, and shared-device token limits.
- Admission reserves planned physical IO tokens and bytes before dispatch.
  Completion releases the full reservation and charges WDRR using actual
  completed physical bytes, including partial and failed completion handling.
- One runtime segment contains at most 8 original requests or 16 MiB,
  whichever limit would be reached first. A large operation remains grouped
  by `batch_token`, but undispatched keys do not occupy the device window.
- The scheduler focuses one primary READ operation. It activates a secondary
  READ operation only when the primary has no immediately dispatchable
  physical IO or has reached its 16-token/48-MiB operation window while a
  shared token remains usable. A momentary global inflight count below 16 is
  not sufficient to activate another operation.

The GDS execution queue does not repeat WDRR, READ priority, or operation
selection. This prevents the runtime and transport layers from applying
conflicting policies.

## Checked-in operating modes

### Trusted baseline

Use `mooncake-transfer-engine/tent/config/tent-gds-baseline.json`:

| Setting | Value |
| --- | ---: |
| scheduler | fixed |
| shared physical tokens | 16 |
| READ workers / inflight | 16 / 16 |
| WRITE workers / inflight | 4 / 1 |
| adaptive concurrency | disabled |
| cuFile Batch / Async | disabled / disabled |
| physical merge | shadow statistics only |
| indirect lease refresh | disabled |

Four WRITE workers are pre-created to avoid pool construction in the hot path,
but the baseline permits only one physical WRITE IO at a time. WRITE-only
inflight values 2, 3, and 4 are experiment points, not production defaults.

### Weighted-fair experiment

Use `mooncake-transfer-engine/tent/config/tent-gds-weighted.json` only after
the fixed baseline is stable:

- READ and WRITE receive 8-MiB and 2-MiB completed-byte quanta, respectively.
- Idle directions do not accumulate credit. Spendable credit is capped at two
  direction quanta.
- With only READ backlog, READ may borrow all 16 tokens.
- With only WRITE backlog, WRITE may use two tokens in the checked-in
  experiment; scan values 1-4 independently before changing this cap.
- With both backlogs, WRITE normally uses at most one token, leaving up to 15
  for READ. Sustained WRITE pressure may promote WRITE to two tokens, leaving
  up to 14 for READ.
- Promotion requires three consecutive pressure windows. Demotion requires
  five healthy windows or occurs immediately when READ P99 exceeds the guard.
  A ten-window cooldown prevents 1-to-2 oscillation.
- WDRR chooses scheduling opportunities. Direction token caps still determine
  instantaneous physical concurrency.

The existing GDS adaptive concurrency controller remains disabled in both
checked-in modes. The WRITE pressure boost changes only the weighted mode's
contended WRITE cap; it does not change worker counts or enable an advanced
path.

## Backpressure and operation lifecycle

The runtime admission queue bounds outstanding owners and bytes, then streams
large 126/192-key operations through bounded segments. It must not reject an
otherwise valid operation merely because all of its keys cannot enter the
dispatch window at once.

An operation can wait, dispatch bounded work, drain inflight IO, and finish or
fail. Cancellation prevents new dispatch. Already-running synchronous cuFile
calls drain normally. Each physical result maps back to its original request;
a key failure affects only mapped keys, and the operation terminal event is
latched exactly once. A request failure does not disable the GDS transport.

Reservations are reconciled exactly once:

1. Reserve planned physical tokens and bytes before transport submission.
2. On submission failure, release the reservation with zero completed bytes.
3. On completion, release the entire reservation and record actual transferred
   bytes, even for a partial or error result.
4. Reject duplicate or unknown completion instead of underflowing accounting.

## Observability

Startup logs identify the canonical config path, executable path, effective
scheduler and worker limits, adaptive state, and the permanently disabled
Batch/Async paths. The benchmark collector also records config and executable
SHA256 values.

Operation start, segment, periodic, and terminal records include:

- operation owner, direction, logical request/byte totals;
- planned, reserved, and settled physical IO/bytes;
- successful and failed mapped owners;
- queue wait, execution, and total operation time;
- active/waiting operation counts and queue lengths;
- actual active workers, cuFile latency, and end-to-end IO latency;
- distinct request/conversation counts, or `unknown` when not supplied.

Use `benchmarks/gds_baseline_collector.py` to collect GPU, NVMe, vLLM,
runtime, KV restore, and inference metrics on a shared timestamp. Missing
sources are marked `available:false`; they are never emitted as zero-valued
measurements.

## Warmup and lease safety

Run `benchmarks/vllm_warmup.py` before formal load. When the connector exports
a Triton kernel manifest, warmup fails unless
`_compute_slot_mapping_kernel` is covered. Without a manifest, the result
explicitly says coverage is unavailable and must not be presented as proof of
kernel coverage.

Periodic `BatchGetReplicaList` calls are not used as an implicit lease-renewal
loop. The RPC changes lease and metric state and can amplify group pinning.
`indirect_lease_refresh` therefore remains disabled until
`benchmarks/gds_lease_refresh_probe.py` passes all of these target-host gates:

- at least 80 operations containing both 126-key and 192-key samples;
- Master CPU increase below 5 percentage points;
- BatchGet P99 increase below 10 percent;
- zero refresh calls after operation terminal;
- expiry no later than 1.25 lease TTL after the final refresh;
- grouped refresh amplification no greater than 1.10.

## Target-host acceptance

Run the same 80-session workload and data set for fixed baseline, weighted
mode, WRITE-only inflight 1/2/3/4, and selected READ token/operation-window
points. Collect synchronized GPU, NVMe, KV restore, operation, and inference
metrics.

Reject a candidate if any of the following occurs:

- `CU_FILE_INTERNAL_ERROR(5030)`;
- wholesale failure of a normal 126/192-key operation;
- data mismatch or incorrect partial/error key mapping;
- a runtime or GDS backlog that does not drain after load;
- worse BatchGet operation completion rate or generation throughput;
- lease-refresh probe failure;
- Triton compilation during the formal benchmark.

Success requires shorter BatchGet operation completion time, more vLLM
`Running` requests, fewer `Waiting`/`Deferred` requests, higher GPU utilization
and generation throughput, and no regression in actual cuFile READ throughput.
No single metric is sufficient.

Physical merge remains shadow-only, Async remains disabled without a
startup-only independent capability probe, and indirect lease refresh remains
disabled until their correctness and target-host gates pass.
