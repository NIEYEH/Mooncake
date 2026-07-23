# GDS Operation-Level Scheduling and Data Path Design

## Purpose

This design turns the GDS path into a measurable, operation-focused pipeline.
The primary goal is to complete a small number of BatchGet operations quickly
enough for vLLM requests to return to `Running`, while keeping the physical
NVMe-oF path inside a stable concurrency envelope.

`batch_token` is an operation-level identifier only. Logs, metrics, code, and
configuration must call it `operation_owner_id` or `operation`; they must not
claim that it identifies a conversation. A later change may introduce an
explicit conversation identifier only after production evidence demonstrates
that the distinction is necessary.

## Scope and rollout phases

### Phase 1: trusted fixed-concurrency baseline

- Keep READ and WRITE worker pools separate.
- Use 16 READ workers with 16 maximum physical READ IOs.
- Pre-create 4 WRITE workers but allow only 1 physical WRITE IO by default.
- Disable adaptive concurrency and weighted-fair scheduling.
- Permit an explicit WRITE-only scan over inflight values 1, 2, 3, and 4;
  this scan is not the default baseline.
- Record the configuration source and SHA256, executable identity, build ID,
  relevant environment allowlist, cuFile driver properties, registered GPU
  bytes, logical bytes, planned physical bytes, and actual transferred bytes.
- Collect GPU, NVMe, vLLM scheduler, KV restore, inference latency, and
  generation-throughput samples on one timestamped timeline.

### Phase 2: operation-focused weighted-fair scheduling

- The runtime queue is the only policy scheduler.
- GDS consumes runtime-selected physical work FIFO within each direction. It
  enforces physical tokens, worker availability, and safety limits, but does
  not run a second READ-first or WDRR policy. An entry blocked solely by its
  direction cap may yield to already-selected work from the other direction,
  avoiding cross-direction head-of-line blocking.
- Restore grouped transport submission only for a bounded segment belonging
  to one operation. cuFile Batch remains disabled; GDS executes individual
  `cuFileRead` and `cuFileWrite` calls.
- Keep generic runtime logical merging disabled for GDS until partial/error
  completion can map exact byte subranges back to every original key.
- Limit active operations and prefer completing the primary READ operation
  before spreading work across more operations.

### Phase 3: physical IO granularity

- Add a shadow merge planner first. Shadow mode records what could be merged
  but executes the original requests.
- Enable actual merging only after mapping, partial-completion, registered
  buffer, boundary, cancellation, and failure-injection tests pass.
- Start with an 8 MiB physical merge cap and scan up to 16 MiB.
- Profile worker queue contention and allocation costs before restructuring
  queues or object ownership.

### Phase 4: advanced paths and automatic control

- Add a multi-signal adaptive controller only after fixed and weighted-fair
  baselines exist. Inputs include completed throughput, P99, backlog, and an
  explicit GPU starvation signal when one becomes available.
- Async is experimental and capability-gated by a startup-only independent
  probe.
- cuFile Batch remains disabled by default and is never probed on production
  requests.
- Finish with worker, direction-limit, active-operation, shared-token, and IO
  size scans.

## Configuration modes

Two checked-in example configurations define reproducible modes.

### Baseline mode

| Setting | Default |
| --- | ---: |
| scheduler mode | `fixed` |
| adaptive concurrency | `false` |
| READ worker threads | 16 |
| READ physical inflight | 16 |
| WRITE worker threads | 4 |
| WRITE physical inflight | 1 |
| waiting high-watermark | 1024 owners / 2 GiB |
| cuFile Batch | disabled |
| Async | disabled |
| merge mode | `shadow` (statistics only) |

### Weighted-fair mode

| Setting | Default |
| --- | ---: |
| shared physical tokens | 16 |
| READ protected capacity during normal/boosted contention | 15 / 14 |
| READ standalone maximum | 16 |
| WRITE minimum reservation | 1 |
| WRITE standalone maximum | 2, configurable 1-4 |
| WRITE maximum while READ and WRITE backlog exist | 1 |
| temporary WRITE pressure maximum | 2 |
| READ:WRITE WDRR weights | 4:1 |
| WDRR base quantum | 2 MiB |
| READ quantum | 8 MiB |
| WRITE quantum | 2 MiB |
| per-direction credit cap | 2 direction quanta |
| target active READ operations | 1 |
| hard active READ operation maximum | 2 |
| hard active WRITE operation maximum | 1 |
| primary READ operation physical-token window | 16 |
| primary READ operation byte window | 48 MiB |
| secondary READ operation inflight segments | 1 |
| segment request maximum | 8 original requests |
| segment byte maximum | 16 MiB |
| merge mode | `shadow` |
| physical merge maximum | 8 MiB, configurable to 16 MiB |

Segment construction stops when adding the next original request would exceed
either 8 requests or 16 MiB. The first limit reached wins. A single oversized
request is split into safe physical IOs; it does not bypass physical limits.

## Scheduling authority and data flow

1. Store creates one transfer-engine batch for a GDS BatchGet or BatchPut.
2. TENT assigns its `batch_token` as the `operation_owner_id`.
3. Runtime admission stores per-operation queued requests without publishing
   the whole operation into the dispatch window.
4. Runtime chooses a direction using completed-byte WDRR when both directions
   have backlog.
5. Runtime chooses an eligible active operation in that direction.
6. GDS produces a side-effect-free physical plan summary for the candidate
   owner. Runtime records total physical IOs, reserves the full byte range,
   and charges only the maximum concurrent physical tokens allowed by the
   active direction/shared window.
7. Runtime builds one bounded operation segment and submits the selected plan
   to GDS. A primary READ operation may have multiple bounded segments inflight
   while it remains inside its operation token and byte windows.
8. GDS expands the selected group into safe physical IOs, consumes shared
   tokens, and executes synchronous single-entry cuFile calls.
9. GDS reports physical completion bytes and per-subrange status.
10. Runtime releases reservations, updates completed-byte service debt,
    advances request/key state, and selects the next segment. A failed
    runtime-queued GDS attempt terminates without automatic cross-transport
    failover until fallback can build a new route-specific reservation.
11. Store resolves individual key results and the operation result.

The physical plan summary is not a second scheduling decision. The current
implementation carries deterministic physical IO count and byte totals, pins
the route chosen during admission, and revalidates the same request/buffer
properties before GDS expands it. A future executable merge path must add an
immutable plan identifier, split boundaries, and subrange mapping and must
consume that exact plan or reject it before any IO starts.

The runtime must never submit all 192 keys merely because they share an
operation. Global tokens, bounded segments, and per-operation byte/token
windows are independent hard bounds.

## Completed-byte WDRR

WDRR is active only while READ and WRITE both have dispatchable backlog.
Single-direction load bypasses WDRR and borrows idle shared capacity up to its
configured standalone maximum.

- Service accounting uses actual physical `transferred_bytes`, not request
  count, logical key count, or submitted bytes.
- Runtime maintains `outstanding_reserved_bytes` and concurrent
  `outstanding_reserved_tokens` per direction as well as globally. An owner
  with more physical sub-IOs than the active token window reserves the window,
  not every sub-IO atomically; GDS streams its remaining sub-IOs through that
  window.
- A direction's dispatch-time spendable deficit is
  `deficit_bytes - outstanding_reserved_bytes`. Runtime must use this value,
  not completed service alone, for every scheduling decision.
- Dispatch creates a bounded byte/token reservation before handing work to GDS.
  The reservation immediately reduces spendable deficit but does not finalize
  the completed-byte service debit.
- Completion releases the full reservation and subtracts actual completed
  physical bytes from `deficit_bytes`. Algebraically this refunds
  `reserved_bytes - actual_completed_bytes`.
- Partial completion debits only the completed prefix.
- A terminal failed IO with zero completed bytes has zero completed-byte
  service; it cannot loop because failed physical IOs are not retried
  implicitly.
- A direction with neither queued work nor outstanding reservations has its
  credit reset to zero.
- Credit is not accrued during idle periods.
- The empty-to-active transition grants at most one initial quantum.
- Positive spendable credit, defined as
  `deficit_bytes - outstanding_reserved_bytes`, is capped at two quanta.
- A global WDRR round grants at most one quantum to each backlogged direction.
  A round advances only after every backlogged direction has consumed a
  scheduling opportunity, reached a physical limit, or cannot dispatch its
  head IO. A timer tick or dispatcher wakeup alone never advances a round.
- Active rounds may add credit while IO is outstanding, but the outstanding
  reservation remains subtracted. This permits work-conserving filling of
  shared tokens without treating unfinished IO as free service.
- Global and per-direction outstanding reservations, rather than submitted
  request count, bound overshoot while service is awaiting completion.
- The default base quantum is 2 MiB. The 4:1 weights therefore grant an 8 MiB
  READ quantum and a 2 MiB WRITE quantum.

To prevent head-of-line starvation when one physical IO is larger than the
two-quantum credit cap, a direction may reserve exactly one oversized head IO
only when it has no existing outstanding reservation and positive deficit.
The reservation makes its spendable deficit negative. No additional contended
IO for that direction may dispatch until active WDRR rounds repay the debt.
The negative debt is bounded by the configured maximum physical IO size.
Single-direction traffic still bypasses WDRR but remains bounded by shared,
direction, and operation reservations.

When both directions have backlog and WRITE is physically capped at one slot,
the 4:1 ratio controls successive scheduling opportunities rather than an
instantaneous 4:1 slot split.

## WRITE pressure hysteresis

Contended WRITE capacity uses three states:

- `NORMAL`: at most one physical WRITE token.
- `BOOSTED`: at most two physical WRITE tokens.
- `COOLDOWN`: one token and promotion prohibited until the cooldown expires.

Promotion requires configured queue-age or queued-byte pressure for a
configured number of consecutive evaluation windows. Demotion requires low
queue age and bytes for a longer configured number of consecutive windows.
READ P99 above its protection threshold immediately leaves `BOOSTED`. A
cooldown interval prevents rapid promotion after demotion.

The initial defaults use one-second evaluation windows. Promotion requires
either oldest WRITE age at or above 500 ms or queued WRITE bytes at or above
64 MiB for three consecutive windows. Demotion requires oldest WRITE age below
100 ms and queued WRITE bytes below 16 MiB for five consecutive windows.
READ physical-IO P99 above 75 ms forces demotion. A ten-second cooldown follows
every demotion. All thresholds are configuration values and are emitted in the
startup fingerprint.

The first implementation uses local WRITE backlog age and bytes. GPU KV release
pressure is not inferred. It becomes a separate controller input only when
vLLM supplies an explicit signal.

## Active-operation focus and operation dispatch window

The runtime prefers one primary READ operation. A segment is a bounded
transport submission unit, not the operation's concurrency limit. Runtime
continues constructing segments for the same primary operation until the first
of these conditions holds:

- shared or READ physical-token reservations are full;
- the primary operation's 16-token reservation window is full;
- the primary operation's 48 MiB byte reservation window is full;
- the operation has no immediately dispatchable physical IO;
- cancellation, failure, or lease safety blocks further dispatch.

This allows the usual 2.25 MiB KV operation to fill 16 tokens through multiple
segments even though each segment stops at 8 requests or 16 MiB. It does not
allow one 192-key operation to publish all work: only the bounded reservations
are visible to GDS, and undispatched requests remain inside the operation.

A second READ operation may be activated only when shared tokens remain idle
after the scheduler has attempted to fill the primary operation window and the
primary:

- has no immediately dispatchable physical IO; or
- is blocked by its operation byte/token window while shared capacity is still
  usable by another operation.

The secondary operation receives one bounded inflight segment initially. It
cannot displace reservations already granted to the primary. An instantaneous
READ inflight count below 16, or merely having one primary segment inflight, is
not sufficient to activate it. The hard active READ operation maximum is two.
The hard active WRITE operation maximum is one.

When an operation becomes terminal, runtime activates the next waiting
operation. This policy is work-conserving only under the explicit conditions
above; it intentionally favors operation completion over owner-wide fairness.

## Operation state machine

An operation follows these states:

```text
WAITING -> ACTIVE -> DRAINING -> COMPLETED
                    |          -> PARTIAL_FAILED
WAITING -----------------------> CANCELED
ACTIVE/DRAINING -> CANCELING --> CANCELED
```

Rules:

- Cancellation prevents every new segment from being dispatched.
- Canceling an operation with no inflight segment transitions directly to
  `CANCELED`; otherwise it transitions to `CANCELING` until all physical IOs
  drain.
- Inflight synchronous cuFile calls are not force-canceled. They drain and
  publish their exact result.
- Queued, undispatched physical work is removed safely.
- One request/key failure updates only its mapped subranges.
- Other keys continue unless the operation has an explicit cancel or fatal
  consistency error.
- A terminal latch permits exactly one terminal transition and one completion
  callback.
- A request error does not mark the GDS transport unavailable.
- Driver shutdown or an explicit uninstall may stop the transport; ordinary
  IO errors may not.

## Reservation reconciliation

Each dispatched physical IO stores:

- direction and operation owner;
- planned and reserved bytes;
- reserved token count;
- original request/key subranges;
- device, registered buffer, segment, and physical offsets.

For every terminal IO:

1. Validate that actual transferred bytes do not exceed reserved bytes.
2. Release the full byte and token reservation exactly once.
3. Debit WDRR by actual completed bytes.
4. Mark fully covered subranges successful.
5. Mark the subrange intersecting a partial boundary failed.
6. Mark all remaining uncovered subranges failed.
7. Recompute segment and operation terminal state.

`actual > reserved`, duplicate completion, reservation underflow, or missing
mapping is a fatal error for the affected operation, not for the transport.

## Lease safety and indirect-renewal validation gate

Current Store leases are granted by `BatchGetReplicaList`. Code inspection
shows that this RPC is not a pure renewal operation:

- it increments Get and cache-hit counters;
- it evaluates promotion-on-hit;
- `GrantLeaseForGroup` may extend every member of a grouped key;
- Master stores one hard `lease_timeout=max(old, now+TTL)` per object, with no
  per-client reference count and no explicit read-lease release RPC;
- checked-in Master configurations use a 5-second default TTL.

Consequently, periodic `BatchGetReplicaList` renewal is disabled in baseline
and weighted-fair modes. TENT does not currently receive the absolute lease
deadline in each transfer request. Before enabling a deadline-aware production
mode, the upstream contract must provide that deadline and implement this
lease-budget gate:

1. Track the absolute lease deadline returned by the initial query for every
   unfinished READ key.
2. Before admitting or dispatching a segment, estimate queue time plus
   reserved-byte service time from conservative completed-throughput EWMA.
3. Require the estimate plus a one-second safety margin to fit inside the
   shortest lease remaining in the segment.
4. If it does not fit, do not dispatch the unsafe segment. Mark it for a
   one-shot refresh decision rather than repeatedly querying Master.
5. After dispatch, if the deadline becomes unsafe, stop new segments and drain
   inflight IO. Do not create an unbounded renewal loop.

A one-shot operation refresh through `BatchGetReplicaList` remains behind the
disabled-by-default `gds_indirect_lease_refresh` experiment. It queries all
unfinished keys in one batch, validates unchanged replica identities, and may
run at most once per operation. Missing keys, changed identities, RPC failure,
or insufficient renewed budget fail only the affected keys. Terminal or
canceling operations never launch it; an already-returning RPC is reconciled
without scheduling new work.

The experiment may be enabled by default only after all validation gates pass:

- ungrouped and grouped objects become removable no later than one TTL plus
  test jitter after the last refresh, proving there is no persistent lease
  leak;
- group-wide pin amplification is measured and bounded, including the number
  of indirectly extended members and bytes;
- no refresh begins after the operation terminal latch, including cancel/RPC
  races;
- an 80-operation benchmark with 126/192-key batches adds less than 5% Master
  CPU, less than 10% BatchGetReplicaList P99, and bounded shard-lock hold time;
- refresh QPS, keys, bytes, group fanout, Master latency, and promotion side
  effects are emitted separately;
- ordinary Get/cache-hit metrics used for workload evaluation are not silently
  treated as user Get traffic. If they cannot be separated, the experiment
  fails and a dedicated renewal RPC is required.

Until this gate passes, the 1024-owner / 2-GiB waiting high-watermark and
operation focus limit exposure, but they are not proof of lease safety. The
target workload must complete within the five-second lease budget; otherwise
the upstream integration must propagate lease deadlines so affected keys can
be refused before expired metadata is used.

## Merge planner and completion mapping

`merge_mode` has three values:

- `off`: do not plan or merge.
- `shadow`: plan and report candidates, execute originals.
- `enabled`: execute proven-safe merged physical IOs.

Requests may merge only when all conditions hold:

- identical opcode, segment, CUDA device, and compatible priority;
- contiguous target offsets;
- contiguous GPU addresses;
- addresses belong to the same registered GPU buffer, unless the registration
  layer explicitly proves a cross-buffer range safe;
- the merged range stays inside the block segment;
- the merged range does not cross the device maximum direct-IO size or the
  configured physical boundary;
- the merged size stays within the configured 8-16 MiB cap;
- executing it cannot violate operation segment or shared-token limits.

Every physical IO carries an ordered list of original request/key byte ranges.
A complete, partial, or failed physical result fans out through this mapping.
The default non-preemptible physical IO cap is 8 MiB so a low-priority merged
IO cannot block a high-priority operation indefinitely.

Shadow metrics include candidate IO count, eligible bytes, predicted physical
sizes, rejection reasons, cross-key candidates, and estimated IO-count
reduction.

## Startup capability probes

cuFile Batch is disabled and not probed unless an explicit experimental
startup configuration is added in a future change.

Async requires all of:

- experimental Async enabled in configuration;
- an explicit independent read-only probe path;
- configured safe offset, GPU device, and probe size;
- successful startup allocation, registration, submission, completion, and
  cleanup.

Missing configuration or any probe failure permanently disables Async for that
process. Production requests never probe capability and never retry through
Batch or Async as a discovery mechanism.

## Observability

### Startup fingerprint

- executable path and build ID/git SHA when available;
- canonical configuration path and SHA256;
- allowlisted environment values, including config path, CUDA visibility, and
  cuFile configuration;
- cuFile driver properties and effective scheduler settings;
- feature states: fixed/WDRR, adaptive, merge, Async, Batch.

### Operation timeline

Each operation emits structured start, periodic, and terminal records with:

- `operation_owner_id`, direction, key count, request count, and logical bytes;
- distinct upstream request and conversation counts only when explicit
  correlation identifiers were supplied; otherwise emit
  `distinct_request_count=unknown` and
  `distinct_conversation_count=unknown`;
- queue-enter, first-dispatch, first-physical-completion, last-completion, and
  terminal timestamps;
- active and waiting operation counts;
- admitted, dispatched, reserved, completed, failed, and canceled bytes;
- completed and failed key/request counts;
- segment count and physical IO count;
- lease renewal count and failures;
- terminal state and cause.

These records must say operation, not conversation.

### Synchronized baseline collector

A benchmark-side collector writes timestamped JSONL/CSV samples for:

- NVIDIA GPU utilization, memory, power, and PCIe where available;
- NVMe block counters, queue depth, bandwidth, latency, and utilization;
- runtime/GDS operation and physical IO summaries;
- vLLM Running, Waiting, Deferred, KV usage, restore, request latency, and
  generation throughput metrics.

The collector records missing tools or unavailable metrics explicitly instead
of silently producing zeros.

## Adaptive controller boundary

The existing adaptive concurrency controller is disabled in baseline and
weighted-fair configurations. A replacement controller is a later phase and
must consume:

- completed READ/WRITE throughput;
- cuFile and operation P99;
- runtime and GDS backlog;
- WRITE persistence pressure;
- an explicit GPU starvation/KV release signal when available.

It adjusts shared tokens or direction limits slowly, uses hysteresis, respects
hard minima/maxima, and cannot enable Async or Batch.

## Tests

Testing follows red-green-refactor and includes:

- idle credit reset and two-quantum cap;
- completed-byte WDRR with unequal 2.25 MiB and 15 MiB IOs;
- bounded reservation before completion;
- outstanding reservations prevent repeated dispatch before completion;
- oversized-head reservation creates bounded debt and blocks further
  contended dispatch until repayment;
- full, partial, zero-byte error, duplicate completion, and reconciliation
  underflow;
- WRITE promotion, demotion, READ-P99 protection, and cooldown;
- one primary READ operation fills all 16 tokens through multiple bounded
  segments;
- primary READ operation focus and exact second-operation activation rules;
- segment limits where request count wins and where bytes win;
- proof that a 192-key operation cannot enter one dispatch segment;
- cancel in WAITING, ACTIVE, and DRAINING;
- exactly-once terminal callback;
- key-isolated failures;
- lease-budget admission and unsafe-segment refusal;
- indirect-refresh disabled-by-default behavior;
- one-shot refresh identity change, missing key, terminal/cancel race, and RPC
  failure;
- ungrouped expiry, grouped pin amplification, post-terminal expiry, Master
  load, and metric-side-effect validation;
- shadow merge decisions and every rejection reason;
- enabled merge complete/partial mapping across original requests;
- registered buffer, device, segment, and physical-boundary enforcement;
- startup Async probe success/failure/no-configuration process latch;
- proof that production requests never probe Async or Batch;
- mixed READ/WRITE runtime tests with one scheduling authority;
- target-device GDS block integration and failure injection.

## Success gates

At the same workload and data set:

- BatchGet operation P50 and P99 completion time decrease;
- vLLM Running requests increase and Waiting/Deferred requests decrease;
- GPU utilization and generation throughput increase;
- actual cuFile READ throughput does not regress;
- no `CU_FILE_INTERNAL_ERROR(5030)` occurs;
- no normal 126/192-key operation is rejected wholesale;
- runtime and GDS backlogs stay bounded and drain after load;
- per-key status matches data validation under success, partial, failure, and
  cancellation.

No single metric is sufficient. The optimized mode is accepted only when the
operation, scheduler, GPU, inference, correctness, and bounded-queue gates hold
together.

## Implementation status (2026-07-23)

The fixed baseline and opt-in weighted-fair scheduler are implemented. Runtime
admission records total planned physical IOs and reserves a bounded concurrent
token charge plus bytes. This lets a multi-physical-IO owner progress through a
one-token WRITE baseline without over-admitting the device. Actual completed
bytes settle WDRR service debt, idle credit is bounded, and the primary READ
operation can fill its window through multiple bounded segments before one
secondary operation is considered. WRITE pressure promotion uses explicit
promotion, demotion, READ-P99 protection, and cooldown windows.

GDS consumes the runtime-selected order through separate READ and WRITE worker
pools under one shared device budget. cuFile Batch and Async calls are absent
from the production path. Physical merge execution is not enabled; shadow
statistics report safe contiguous candidates. Operation timeline logs and the
synchronized baseline collector expose queue wait, active workers, cuFile
latency, physical completion bytes, operation latency, queue lengths, and
config/executable identity.

Indirect lease refresh is intentionally not implemented in the production
path. Master expiry/group behavior tests and the external load-probe evaluator
define the acceptance gate. Absolute lease deadlines are not present in the
TENT request contract, so deadline-aware refusal remains an upstream
integration requirement. Scheduler cancellation is tested internally, but the
public TransferEngine API does not expose operation cancellation. Async
likewise remains disabled until a startup-only independent probe is
implemented and validated.

Host-independent scheduler, admission, FIFO, operation-timeline, collector,
lease-probe, and warmup tests are available. CUDA/cuFile, NVMe-oF, Master load,
data-integrity failure injection, and the 80-session GPU/inference matrix remain
target-host gates; no claim of end-to-end GDS success is made until those gates
pass.
