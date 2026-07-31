# GDS Fixed WRITE Floor Implementation Plan

> **Execution mode:** inline, test-first. The user-supplied design is the
> approved specification for this change.

**Goal:** Prevent Fixed-mode GDS WRITE starvation when READ and WRITE both
have demand, including the production single-slot refill pattern (14 READ
reservations, 0 WRITE reservations, one available shared token).

**Root cause:** `GdsOperationScheduler::select()` currently treats contention
as a WeightedFair-only state, while `canReserve()` independently treats queued
or reserved demand in both directions as contention. In Fixed mode, selection
therefore defaults to READ even though capacity accounting reserves a WRITE
floor. Repeated one-entry refills can keep choosing READ indefinitely.

**Required semantics:** With demand in both directions, Fixed mode must select
WRITE while its reserved token count is below
`min(contended_write_tokens, budget.max_write_tokens)`. READ-only and
WRITE-only behavior retains the standalone direction limits. WeightedFair
direction selection and WDRR charging remain unchanged.

---

## Task 1: Lock the regression with tests

**Files:**

- Modify: `mooncake-transfer-engine/tent/tests/admission_queue_test.cpp`
- Modify: `mooncake-transfer-engine/tent/tests/gds_operation_scheduler_test.cpp`
- Modify: `mooncake-transfer-engine/tent/tests/CMakeLists.txt`

Add the exact single-slot refill scenario: reserve 14 READs, enqueue remaining
READ and WRITE work, then call selection with `max_entries=1`. Assert the
selected reservation is WRITE. Run it against the current code and retain the
failure output. Remove existing assertions that require the WRITE to be last;
assert direction counts and membership instead. Link
`gds_operation_scheduler.cpp` into `admission_queue_test`.

## Task 2: Unify contention and implement the Fixed floor

**Files:**

- Modify: `mooncake-transfer-engine/tent/include/tent/runtime/gds_operation_scheduler.h`
- Modify: `mooncake-transfer-engine/tent/src/runtime/gds_operation_scheduler.cpp`

Add `hasDemand()`, `capacityContended()`, and
`fixedWriteFloorNeeded(budget)`. Use `capacityContended()` in both selection
and reservation accounting. Keep a separate `weighted_contended` predicate
for WDRR-only behavior. In Fixed mode, choose WRITE first whenever the
effective WRITE floor is not met; fall back to the alternate direction only
when the floor candidate cannot be admitted.

Run scheduler and admission-queue tests and confirm the regression becomes
green without changing the WeightedFair tests.

## Task 3: Add starvation diagnostics

**Files:**

- Modify: `mooncake-transfer-engine/tent/include/tent/runtime/gds_operation_scheduler.h`
- Modify: `mooncake-transfer-engine/tent/src/runtime/gds_operation_scheduler.cpp`
- Modify: `mooncake-transfer-engine/tent/include/tent/runtime/admission_queue.h`
- Modify: `mooncake-transfer-engine/tent/src/runtime/admission_queue.cpp`
- Modify: `mooncake-transfer-engine/tent/include/tent/runtime/transfer_engine_impl.h`
- Modify: `mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp`
- Modify tests above as needed

Expose cumulative scheduler counters for Fixed-floor opportunities,
dispatches, and blocks, plus the last selection budget. Count admission-level
floor misses when WRITE demand and budget exist but a selection returns no
WRITE reservation. Extend the one-second runtime summary with these fields and
with a cumulative `write_starvation_windows` count. A starvation window means
WRITE was queued, READ dispatched, WRITE did not dispatch, and a positive
WRITE budget was observed. Emit a rate-limited anomaly after three consecutive
such windows, including queue depth, reservations, budgets, and sequence
barrier.

## Task 4: Verify

Run the standalone scheduler and admission-queue operation tests on the local
host, compile all changed translation units that local dependencies permit,
run `git diff --check`, and inspect the complete diff. Provide the Linux CMake
commands and the 80-client/10-turn acceptance signals for the target GDS host.

