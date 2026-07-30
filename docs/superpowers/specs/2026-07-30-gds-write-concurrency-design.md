# GDS Write Concurrency and Diagnostics Design

## Problem

The fixed GDS configuration serializes every WRITE at three independent
layers:

- the runtime dispatch window admits one WRITE owner;
- the operation scheduler grants one standalone WRITE token;
- the GDS transport permits one inflight WRITE.

Production logs confirm the resulting behavior: `peak_active_workers=1`,
`effective_limit=1`, and approximately 338 MiB/s while dozens of WRITE owners
wait in the runtime queue. An attempted increase also produced this startup
error:

```text
InvalidArgument: GDS direction/operation tokens exceed shared tokens
```

That error is a configuration-invariant failure before any cuFile operation.
It does not show that concurrent cuFile writes failed. The current error text
does not identify the invalid field or print the effective values, which makes
safe tuning unnecessarily difficult.

The vLLM shutdown traceback reported after a benchmark run is outside this
change. External prefix-cache hit rate remains a diagnostic concern because
the supplied run showed WRITE traffic but no GDS READ traffic.

## Goals

- Allow up to four concurrent GDS WRITEs when no READ work is pending.
- Preserve strict READ priority and a one-token contended WRITE limit.
- Keep the shared physical device limit at 16.
- Reject inconsistent token configurations with field-specific values and
  relationships.
- Report the actual runtime WRITE limit and the reason dispatch is restricted.
- Distinguish a missing external-cache lookup from a lookup that found no GDS
  replicas.
- Provide host-independent regression tests plus explicit target-host
  acceptance checks.

## Non-goals

- Do not remove shared-token or direction-token safety checks.
- Do not silently clamp invalid configurations.
- Do not enable cuFile Batch or Async APIs.
- Do not enable adaptive concurrency.
- Do not change the READ limit, the 16-token shared limit, request merging, or
  lease behavior.
- Do not modify vLLM request timeout or shutdown handling.

## Configuration

The default and fixed-baseline configurations use these matched limits:

| Setting | Value |
| --- | ---: |
| `runtime_queue/max_dispatch_write_owners` | 4 |
| `runtime_queue/gds_shared_physical_tokens` | 16 |
| `runtime_queue/gds_read_standalone_tokens` | 16 |
| `runtime_queue/gds_write_standalone_tokens` | 4 |
| `runtime_queue/gds_contended_write_tokens` | 1 |
| `runtime_queue/gds_primary_read_tokens` | 16 |
| `transports/gds/write_worker_threads` | 4 |
| `transports/gds/shared_device_tokens` | 16 |
| `transports/gds/max_inflight_reads` | 16 |
| `transports/gds/max_inflight_writes` | 4 |

The weighted-fair configuration remains opt-in and keeps its existing
standalone WRITE limit of two.

The scheduler validates these relationships:

```text
shared_tokens >= read_standalone_tokens
shared_tokens >= write_standalone_tokens
shared_tokens >= primary_read_tokens
write_standalone_tokens >= contended_write_tokens
```

The runtime also retains its transport-capacity checks. Invalid configuration
is a startup error; the error names the failed relationship and prints all
five scheduler token values.

## Dispatch Behavior

The existing scheduling authority remains unchanged.

1. `TransferEngineImpl` admits a maximum of four WRITE owners in a write-only
   fixed-mode dispatch window.
2. `GdsOperationScheduler` reserves at most four standalone WRITE tokens while
   enforcing the global 16-token limit.
3. `GdsTransport` runs at most four cuFile WRITE calls on its four WRITE worker
   threads.
4. When READ pressure exists, fixed mode continues to pause new WRITEs. If a
   mode permits contended WRITEs, their cap remains one.
5. Completion releases reservations exactly once and wakes runtime dispatch.

Client concurrency is not a device-token setting. An 80-client benchmark must
not configure 80 WRITE tokens.

## Observability

### Startup validation

The scheduler startup failure includes:

- scheduler mode;
- shared, standalone READ, standalone WRITE, contended WRITE, and primary READ
  token values;
- the exact failed relationship.

The normal startup fingerprint prints the same values alongside the transport
worker and inflight limits.

### GDS one-second summary

The WRITE summary separates:

- configured transport limit;
- adaptive/current transport limit;
- actual runtime dispatch limit;
- current READ-pressure state;
- whether WRITEs are paused for READ;
- inflight, active, peak-active, internal-queued, and runtime-queued counts.

The current summary incorrectly labels the current transport limit as
`runtime_dispatch_limit`; this change reports the computed dispatch limit.

Dispatch-window logs record a reason from this bounded set:

- `shared_tokens`;
- `read_direction_limit`;
- `write_direction_limit`;
- `write_paused_for_read`;
- `worker_pool`;
- `fifo_front`.

Counters are accumulated over the one-second interval so logging does not add
one record per IO.

### External-cache lookup

The Store BatchGet boundary emits a sampled lookup summary even when zero keys
resolve to GDS. It reports requested keys and counts classified as metadata
miss, memory replica, GDS replica, local/offload replica, or query error.
Consequently:

- no lookup summary means vLLM did not call the Store BatchGet path;
- a summary with only metadata misses means keys were requested but absent;
- GDS replicas with no READ dispatch identify a later Store/TENT failure.

No object key or prompt content is logged.

## Error Handling

- Invalid token relationships fail before Transfer Engine construction with
  actionable values.
- Runtime WRITE concurrency is never silently raised above a worker, direction,
  or shared-token limit.
- cuFile failures retain per-IO errno, result, offsets, size, device, and
  latency. Any target-host cuFile failure fails acceptance and requires a
  reduction to two WRITE tokens based on evidence.
- External-cache diagnostic classification must not change BatchGet results or
  retry behavior.

## Testing

Host-independent tests cover:

- the fixed/default JSON files contain the matched 4/16/1 limits;
- each invalid scheduler relationship returns a field-specific error with all
  effective values;
- valid READ=16, WRITE=4, contended WRITE=1, primary READ=16 configuration is
  accepted;
- write-only dispatch exposes four tokens;
- READ pressure reduces the actual WRITE dispatch limit according to fixed-mode
  policy;
- summary snapshots distinguish configured, current, and runtime limits;
- BatchGet lookup classification covers metadata misses, GDS replicas, other
  replicas, and query errors without exposing keys.

Target-host verification uses the existing destructive GDS test controls and
the 80-client, 10-turn benchmark:

1. Run independent concurrent WRITEs at limits 1, 2, and 4 on non-overlapping,
   explicitly confirmed block-device ranges.
2. Verify byte-for-byte data, zero cuFile failures, and terminal completion.
3. Under write-only load, verify `runtime_dispatch_limit=4` and
   `peak_active_workers` reaches at least two; four is expected under sustained
   backlog.
4. Run mixed READ/WRITE load and verify READ pressure prevents four concurrent
   WRITEs.
5. Run all 800 conversation requests and require 800 terminal client results,
   zero GDS transfer failures, and a drained runtime backlog.

The local environment cannot prove target NVMe/GPU throughput. Completion is
therefore split into host-independent test success and a clearly reported
target-host acceptance result.

