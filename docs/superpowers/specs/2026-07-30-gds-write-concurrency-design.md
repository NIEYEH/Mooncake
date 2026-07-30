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

- Add an explicit target-validation configuration that allows up to four
  concurrent GDS WRITEs when no READ work is pending.
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

The new `tent-gds-write4.json` target-validation configuration uses these
matched limits:

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

The production default and fixed-baseline configurations retain a WRITE limit
of one until the target-host acceptance gates pass. The weighted-fair
configuration remains opt-in and keeps its existing standalone WRITE limit of
two. After a successful target-host run, promoting the fixed/default files to
four is a separate, evidence-backed configuration change.

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

Standalone and direction limits are caps, not reserved partitions. Every
reservation consumes the same shared pool:

```text
reserved_read_tokens + reserved_write_tokens <= shared_physical_tokens
```

The scheduler already enforces this invariant. Under weighted contention, one
reserved WRITE token leaves at most 15 of the 16 shared tokens available for
READ. `primary_read_tokens=16` is the standalone operation cap; it does not
permit 16 READ reservations alongside a WRITE reservation.

## Dispatch Behavior

The existing scheduling authority remains unchanged.

1. With `tent-gds-write4.json`, `TransferEngineImpl` admits a maximum of four
   WRITE owners in a write-only fixed-mode dispatch window.
2. `GdsOperationScheduler` reserves at most four standalone WRITE physical-IO
   tokens while enforcing the global 16-token limit.
3. `GdsTransport` runs at most four cuFile WRITE calls on its four WRITE worker
   threads.
4. When READ pressure exists, fixed mode continues to pause new WRITEs. If a
   mode permits contended WRITEs, their cap remains one.
5. Completion releases reservations exactly once and wakes runtime dispatch.

Client concurrency is not a device-token setting. An 80-client benchmark must
not configure 80 WRITE tokens.

One scheduler token represents one physical IO slot, not one conversation,
operation, or logical owner. A logical owner may charge multiple tokens when
its transport plan contains multiple physical IOs, bounded by direction,
operation, and shared limits. The observed workload uses approximately one
2.25 MiB physical IO per logical owner, so four independent owners are needed
to demonstrate four active workers reliably.

READ pressure is:

```text
queued_read_owners > 0
|| reserved_read_tokens > 0
|| transport_pending_reads > 0
|| transport_inflight_reads > 0
```

When READ pressure appears, fixed mode stops dispatching new WRITEs. It does
not cancel a WRITE already submitted to cuFile. Existing WRITEs complete and
release their reservations exactly once, after which READ consumes the
released shared capacity. When every READ-pressure source returns to zero,
the runtime wakes dispatch and the validation configuration's write-only
limit returns to four. Configured and current transport limits remain four
while the actual runtime limit is zero due to `write_paused_for_read`.

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

The Store BatchGet boundary records lossless cumulative counters in
`ClientMetric` and exports them through the existing client `/metrics`
endpoint. The synchronized benchmark collector samples the cumulative values
once per second and writes interval deltas, including zero-activity intervals.
Counters include:

- BatchGet calls and requested keys;
- metadata hits, misses, and query errors;
- multi-label replica availability for memory, GDS, and local/offload;
- mutually exclusive selected routes for memory, GDS, and local/offload;
- GDS skipped because memory or local was selected;
- GDS submit attempts and submit failures.

This supports bounded diagnostic conclusions:

- `batch_get_calls=0` means vLLM did not call the Store BatchGet path during
  that interval;
- calls with zero metadata hits mean keys were requested but absent or errored;
- GDS available with GDS selected zero may be normal route preference;
- GDS selected above zero with zero GDS submit attempts identifies a break
  before the Store/TENT submit boundary;
- GDS submit attempts above zero with zero transport READ dispatch identifies
  a later submit/admission issue;
- transport READ dispatch with no completion identifies the GDS execution or
  completion path.

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

- the production fixed/default JSON files retain the conservative one-WRITE
  limit;
- `tent-gds-write4.json` contains the matched 4/16/1 validation limits;
- each invalid scheduler relationship returns a field-specific error with all
  effective values;
- valid READ=16, WRITE=4, contended WRITE=1, primary READ=16 configuration is
  accepted;
- contended reservation never exceeds 15 READ plus one WRITE token, and the
  combined reservation never exceeds the shared pool;
- write-only dispatch exposes four tokens;
- existing WRITEs drain without cancellation when READ pressure appears;
- queued, reserved, pending, or inflight READ pressure prevents new WRITE
  dispatch, and clearing all pressure wakes WRITE dispatch back to four;
- summary snapshots distinguish configured, current, and runtime limits;
- BatchGet interval accounting is lossless and separates multi-label available
  replicas from a mutually exclusive selected route;
- scheduler/runtime fault paths cover immediate submit rejection, failed,
  canceled, timeout, duplicate, and partial completion. Each in-scope path
  verifies reservations and inflight counts return to zero, the terminal
  transition occurs once, the runtime is woken, and the next batch can
  dispatch. Shutdown with outstanding work remains outside this change, as
  stated in the scope above.

Target-host verification uses the existing destructive GDS test controls and
the 80-client, 10-turn benchmark:

1. Run four independent WRITE operations/owners, each with sustained,
   non-overlapping IO backlog, at limits 1, 2, and 4 on explicitly confirmed
   block-device ranges.
2. Verify byte-for-byte data, zero cuFile failures, and terminal completion.
3. Under write-only load, verify `runtime_dispatch_limit=4` and
   `peak_active_workers` reaches at least two; four is expected under sustained
   backlog.
4. Run mixed READ/WRITE load and verify READ pressure prevents four concurrent
   WRITEs. Relative to the one-WRITE baseline, READ throughput may decline by
   no more than 5% and READ P99 may increase by no more than 10%.
5. Require WRITE-only throughput at limit 2 to reach at least 1.5 times limit
   1, and limit 4 to reach at least 1.25 times limit 2. Runtime queue-wait P99
   must decrease, cuFile P99 must remain bounded, and all reservations must
   return to zero.
6. Run the complete conversation workload and require:

```text
submitted_requests = 800
completed_http_200 = 800
client_timeouts = 0
server_5xx = 0
cancelled = 0
max_input_num_turns = 10
GDS failed_requests = 0
active_gds_operations = 0
queued_owners = 0
dispatch_inflight_owners = 0
reserved_read_tokens = 0
reserved_write_tokens = 0
```

7. Only after gates 1 through 6 pass may the fixed/default configuration be
   promoted from one to four WRITE tokens.

The local environment cannot prove target NVMe/GPU throughput. Completion is
therefore split into host-independent test success and a clearly reported
target-host acceptance result.
