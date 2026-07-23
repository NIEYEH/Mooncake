// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tent/runtime/gds_operation_scheduler.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace mooncake::tent {
namespace {

constexpr size_t kMiB = 1UL << 20;

[[noreturn]] void fail(const char* expression, int line) {
    std::cerr << "FAILED line " << line << ": " << expression << std::endl;
    std::exit(1);
}

#define EXPECT_TRUE(expression)        \
    do {                               \
        if (!(expression)) {           \
            fail(#expression, __LINE__); \
        }                              \
    } while (0)

#define EXPECT_EQ(lhs, rhs)                    \
    do {                                       \
        const auto lhs_value = (lhs);          \
        const auto rhs_value = (rhs);          \
        if (lhs_value != rhs_value) {          \
            fail(#lhs " == " #rhs, __LINE__); \
        }                                      \
    } while (0)

GdsOperationSchedulerConfig weightedConfig() {
    GdsOperationSchedulerConfig config;
    config.mode = GdsSchedulerMode::WeightedFair;
    config.shared_tokens = 16;
    config.read_standalone_tokens = 16;
    config.write_standalone_tokens = 2;
    config.contended_write_tokens = 1;
    config.read_quantum_bytes = 8 * kMiB;
    config.write_quantum_bytes = 2 * kMiB;
    config.credit_cap_quanta = 2;
    config.primary_read_tokens = 16;
    config.primary_read_bytes = 48 * kMiB;
    config.secondary_segment_requests = 8;
    config.secondary_segment_bytes = 16 * kMiB;
    return config;
}

GdsDispatchEntry entry(uint64_t owner, uint64_t operation,
                       GdsDirection direction, size_t bytes,
                       size_t tokens = 1) {
    return GdsDispatchEntry{owner, operation, direction, bytes, tokens};
}

void testOutstandingReservationsBoundRepeatedSelection() {
    auto config = weightedConfig();
    GdsOperationScheduler scheduler(config);
    for (uint64_t index = 0; index < 32; ++index) {
        EXPECT_TRUE(scheduler
                        .enqueue(entry(100 + index, 7, GdsDirection::Read,
                                       2359296))
                        .ok());
    }
    for (uint64_t index = 0; index < 8; ++index) {
        EXPECT_TRUE(scheduler
                        .enqueue(entry(200 + index, 8, GdsDirection::Write,
                                       2359296))
                        .ok());
    }

    std::vector<GdsDispatchReservation> reservations;
    for (size_t wakeup = 0; wakeup < 32; ++wakeup) {
        auto selected = scheduler.select({16, 64 * kMiB, 32});
        reservations.insert(reservations.end(), selected.begin(),
                            selected.end());
    }

    const auto snapshot = scheduler.snapshot();
    EXPECT_TRUE(snapshot.reserved_tokens[0] +
                    snapshot.reserved_tokens[1] <=
                config.shared_tokens);
    EXPECT_TRUE(snapshot.reserved_tokens[1] <=
                config.contended_write_tokens);
    EXPECT_TRUE(snapshot.operation_reserved_tokens.at(7) <=
                config.primary_read_tokens);
    EXPECT_EQ(snapshot.reserved_bytes[0] + snapshot.reserved_bytes[1],
              snapshot.global_reserved_bytes);
}

void testPartialCompletionRefundsReservation() {
    GdsOperationScheduler scheduler(weightedConfig());
    EXPECT_TRUE(
        scheduler
            .enqueue(entry(1, 11, GdsDirection::Read, 8 * kMiB))
            .ok());
    auto selected = scheduler.select({1, 8 * kMiB, 1});
    EXPECT_EQ(selected.size(), 1u);
    EXPECT_TRUE(
        scheduler.complete(selected.front().id, 3 * kMiB, FAILED).ok());

    const auto snapshot = scheduler.snapshot();
    EXPECT_EQ(snapshot.reserved_bytes[0], 0u);
    EXPECT_EQ(snapshot.reserved_tokens[0], 0u);
    EXPECT_EQ(snapshot.completed_bytes[0], 3 * kMiB);
    EXPECT_EQ(snapshot.operation_reserved_tokens.at(11), 0u);
}

void testDuplicateAndOversizedCompletionFail() {
    GdsOperationScheduler scheduler(weightedConfig());
    EXPECT_TRUE(
        scheduler
            .enqueue(entry(1, 11, GdsDirection::Read, 2 * kMiB))
            .ok());
    auto selected = scheduler.select({1, 2 * kMiB, 1});
    EXPECT_EQ(selected.size(), 1u);
    EXPECT_TRUE(scheduler
                    .complete(selected.front().id, 2 * kMiB + 1, COMPLETED)
                    .IsInvalidArgument());
    EXPECT_TRUE(
        scheduler.complete(selected.front().id, 2 * kMiB, COMPLETED).ok());
    EXPECT_TRUE(
        scheduler.complete(selected.front().id, 2 * kMiB, COMPLETED)
            .IsInvalidEntry());
}

void testPrimaryReadOperationFillsSharedTokens() {
    auto config = weightedConfig();
    GdsOperationScheduler scheduler(config);
    for (uint64_t index = 0; index < 32; ++index) {
        EXPECT_TRUE(scheduler
                        .enqueue(entry(100 + index, 77, GdsDirection::Read,
                                       2359296))
                        .ok());
        EXPECT_TRUE(scheduler
                        .enqueue(entry(200 + index, 88, GdsDirection::Read,
                                       2359296))
                        .ok());
    }

    auto selected = scheduler.select({16, 48 * kMiB, 16});
    EXPECT_EQ(selected.size(), 16u);
    for (const auto& reservation : selected) {
        EXPECT_EQ(reservation.operation_owner_id, 77u);
    }
}

void testIdleDirectionCannotBankCredit() {
    auto config = weightedConfig();
    GdsOperationScheduler scheduler(config);
    for (size_t wakeup = 0; wakeup < 64; ++wakeup) {
        EXPECT_TRUE(scheduler.select({16, 64 * kMiB, 16}).empty());
    }
    EXPECT_TRUE(
        scheduler
            .enqueue(entry(1, 9, GdsDirection::Write, 8 * kMiB))
            .ok());
    EXPECT_TRUE(
        scheduler
            .enqueue(entry(2, 10, GdsDirection::Read, 2 * kMiB))
            .ok());
    auto selected = scheduler.select({16, 64 * kMiB, 16});
    EXPECT_TRUE(!selected.empty());
    const auto snapshot = scheduler.snapshot();
    EXPECT_TRUE(snapshot.spendable_deficit_bytes[1] <=
                static_cast<int64_t>(2 * config.write_quantum_bytes));
}

void testDispatcherWakeupDoesNotGrantAnotherQuantum() {
    auto config = weightedConfig();
    GdsOperationScheduler scheduler(config);
    for (uint64_t index = 0; index < 8; ++index) {
        EXPECT_TRUE(scheduler
                        .enqueue(entry(100 + index, 31, GdsDirection::Read,
                                       2 * kMiB))
                        .ok());
        EXPECT_TRUE(scheduler
                        .enqueue(entry(200 + index, 32, GdsDirection::Write,
                                       2 * kMiB))
                        .ok());
    }

    EXPECT_EQ(scheduler.select({16, 64 * kMiB, 1}).size(), 1u);
    EXPECT_EQ(scheduler.select({16, 64 * kMiB, 1}).size(), 1u);
    const auto snapshot = scheduler.snapshot();
    EXPECT_EQ(snapshot.reserved_bytes[0], 4 * kMiB);
    EXPECT_EQ(snapshot.spendable_deficit_bytes[0],
              static_cast<int64_t>(4 * kMiB));
}

void testDrainedOperationCanReceiveNextAdmissionSegment() {
    GdsOperationScheduler scheduler(weightedConfig());
    EXPECT_TRUE(
        scheduler
            .enqueue(entry(1, 71, GdsDirection::Read, 2 * kMiB))
            .ok());
    auto first = scheduler.select({16, 64 * kMiB, 16});
    EXPECT_EQ(first.size(), 1u);
    EXPECT_TRUE(
        scheduler.complete(first.front().id, 2 * kMiB, COMPLETED).ok());

    EXPECT_TRUE(
        scheduler
            .enqueue(entry(2, 71, GdsDirection::Read, 2 * kMiB))
            .ok());
    auto second = scheduler.select({16, 64 * kMiB, 16});
    EXPECT_EQ(second.size(), 1u);
    EXPECT_EQ(second.front().operation_owner_id, 71u);
}

void testFixedModeUsesWriteSlotWhenReadWindowIsFull() {
    auto config = weightedConfig();
    config.mode = GdsSchedulerMode::Fixed;
    GdsOperationScheduler scheduler(config);
    EXPECT_TRUE(
        scheduler
            .enqueue(entry(1, 81, GdsDirection::Read, 2 * kMiB))
            .ok());
    EXPECT_TRUE(
        scheduler
            .enqueue(entry(2, 81, GdsDirection::Read, 2 * kMiB))
            .ok());
    EXPECT_TRUE(
        scheduler
            .enqueue(entry(3, 82, GdsDirection::Write, 2 * kMiB))
            .ok());

    auto selected = scheduler.select(
        {16, 64 * kMiB, 2, 1, 1});
    EXPECT_EQ(selected.size(), 2u);
    EXPECT_EQ(selected[0].direction, GdsDirection::Read);
    EXPECT_EQ(selected[1].direction, GdsDirection::Write);
}

void testFixedModeReservesOneContendedWriteToken() {
    auto config = weightedConfig();
    config.mode = GdsSchedulerMode::Fixed;
    GdsOperationScheduler scheduler(config);
    for (uint64_t index = 0; index < 32; ++index) {
        EXPECT_TRUE(scheduler
                        .enqueue(entry(100 + index, 83, GdsDirection::Read,
                                       2 * kMiB))
                        .ok());
    }
    EXPECT_TRUE(
        scheduler
            .enqueue(entry(200, 84, GdsDirection::Write, 2 * kMiB))
            .ok());

    auto selected = scheduler.select(
        {16, 64 * kMiB, 16, 16, 1});
    EXPECT_EQ(selected.size(), 16u);
    size_t reads = 0;
    size_t writes = 0;
    for (const auto& reservation : selected) {
        if (reservation.direction == GdsDirection::Read) {
            ++reads;
        } else {
            ++writes;
        }
    }
    EXPECT_EQ(reads, 15u);
    EXPECT_EQ(writes, 1u);
}

}  // namespace
}  // namespace mooncake::tent

int main() {
    using namespace mooncake::tent;
    testOutstandingReservationsBoundRepeatedSelection();
    testPartialCompletionRefundsReservation();
    testDuplicateAndOversizedCompletionFail();
    testPrimaryReadOperationFillsSharedTokens();
    testIdleDirectionCannotBankCredit();
    testDispatcherWakeupDoesNotGrantAnotherQuantum();
    testDrainedOperationCanReceiveNextAdmissionSegment();
    testFixedModeUsesWriteSlotWhenReadWindowIsFull();
    testFixedModeReservesOneContendedWriteToken();
    std::cout << "gds_operation_scheduler_test: PASS" << std::endl;
    return 0;
}
