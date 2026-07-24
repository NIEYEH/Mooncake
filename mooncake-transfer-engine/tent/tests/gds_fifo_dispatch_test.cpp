// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "tent/transport/gds/gds_fifo_dispatch.h"

#include <cstdlib>
#include <iostream>

namespace mooncake::tent {
namespace {

[[noreturn]] void fail(const char* expression, int line) {
    std::cerr << "FAILED line " << line << ": " << expression << std::endl;
    std::exit(1);
}

#define EXPECT_TRUE(expression)          \
    do {                                 \
        if (!(expression)) {             \
            fail(#expression, __LINE__); \
        }                                \
    } while (0)

#define EXPECT_EQ(lhs, rhs)                    \
    do {                                       \
        const auto lhs_value = (lhs);          \
        const auto rhs_value = (rhs);          \
        if (lhs_value != rhs_value) {          \
            fail(#lhs " == " #rhs, __LINE__); \
        }                                      \
    } while (0)

void testSharedBudgetBoundsBothPools() {
    GdsFifoDispatchState state{16, 16, 4, 15, 1};
    EXPECT_TRUE(!gdsFifoFrontCanDispatch(state, false));
    EXPECT_TRUE(!gdsFifoFrontCanDispatch(state, true));

    --state.inflight_reads;
    EXPECT_TRUE(gdsFifoFrontCanDispatch(state, false));
    EXPECT_TRUE(gdsFifoFrontCanDispatch(state, true));
}

void testDirectionPoolLetsOppositeDirectionBypassBlockedFront() {
    GdsFifoDispatchState state{16, 16, 1, 8, 1};
    EXPECT_TRUE(!gdsFifoFrontCanDispatch(state, true));
    EXPECT_TRUE(gdsFifoFrontBlocksQueue(state, true));
    EXPECT_TRUE(gdsFifoFrontCanDispatch(state, false));
    EXPECT_TRUE(gdsFifoCanBypassBlockedFront(
        state, true, false));
    EXPECT_TRUE(!gdsFifoCanBypassBlockedFront(
        state, true, true));

    state.inflight_reads = 15;
    EXPECT_TRUE(!gdsFifoCanBypassBlockedFront(
        state, true, false));
}

void testReservationUpdatesPhysicalCounts() {
    GdsFifoDispatchState state{16, 16, 4, 0, 0};
    EXPECT_TRUE(gdsFifoReserve(state, false));
    EXPECT_TRUE(gdsFifoReserve(state, true));
    EXPECT_EQ(state.inflight_reads, 1u);
    EXPECT_EQ(state.inflight_writes, 1u);
    EXPECT_EQ(gdsFifoSharedInflight(state), 2u);
}

void testPartialDirectIoReportsActualBytesWithoutCompleting() {
    const auto partial = gdsDirectIoOutcome(1024, 4096);
    EXPECT_EQ(partial.transferred_bytes, 1024u);
    EXPECT_TRUE(!partial.completed);

    const auto complete = gdsDirectIoOutcome(4096, 4096);
    EXPECT_EQ(complete.transferred_bytes, 4096u);
    EXPECT_TRUE(complete.completed);

    const auto failed = gdsDirectIoOutcome(-1, 4096);
    EXPECT_EQ(failed.transferred_bytes, 0u);
    EXPECT_TRUE(!failed.completed);
}

void testWriteExecutionLimitTracksReadContention() {
    EXPECT_EQ(gdsFifoEffectiveWriteLimit(4, 1, false), 4u);
    EXPECT_EQ(gdsFifoEffectiveWriteLimit(4, 1, true), 1u);
    EXPECT_EQ(gdsFifoEffectiveWriteLimit(4, 2, true), 2u);

    // A zero limit is a valid execution policy for read-drain mode. The
    // opposite direction may still bypass the blocked WRITE head.
    GdsFifoDispatchState read_drain{16, 16, 0, 15, 0};
    EXPECT_TRUE(!gdsFifoFrontCanDispatch(read_drain, true));
    EXPECT_TRUE(gdsFifoFrontCanDispatch(read_drain, false));
    EXPECT_TRUE(gdsFifoCanBypassBlockedFront(read_drain, true, false));
}

}  // namespace
}  // namespace mooncake::tent

int main() {
    using namespace mooncake::tent;
    testSharedBudgetBoundsBothPools();
    testDirectionPoolLetsOppositeDirectionBypassBlockedFront();
    testReservationUpdatesPhysicalCounts();
    testPartialDirectIoReportsActualBytesWithoutCompleting();
    testWriteExecutionLimitTracksReadContention();
    std::cout << "gds_fifo_dispatch_test: PASS" << std::endl;
    return 0;
}
