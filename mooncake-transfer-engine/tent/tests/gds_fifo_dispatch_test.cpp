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

void testDirectionPoolBoundsFifoFront() {
    GdsFifoDispatchState state{16, 16, 1, 8, 1};
    EXPECT_TRUE(!gdsFifoFrontCanDispatch(state, true));
    // FIFO semantics do not skip this blocked WRITE to schedule a later READ.
    EXPECT_TRUE(gdsFifoFrontBlocksQueue(state, true));
    EXPECT_TRUE(gdsFifoFrontCanDispatch(state, false));
}

void testReservationUpdatesPhysicalCounts() {
    GdsFifoDispatchState state{16, 16, 4, 0, 0};
    EXPECT_TRUE(gdsFifoReserve(state, false));
    EXPECT_TRUE(gdsFifoReserve(state, true));
    EXPECT_EQ(state.inflight_reads, 1u);
    EXPECT_EQ(state.inflight_writes, 1u);
    EXPECT_EQ(gdsFifoSharedInflight(state), 2u);
}

}  // namespace
}  // namespace mooncake::tent

int main() {
    using namespace mooncake::tent;
    testSharedBudgetBoundsBothPools();
    testDirectionPoolBoundsFifoFront();
    testReservationUpdatesPhysicalCounts();
    std::cout << "gds_fifo_dispatch_test: PASS" << std::endl;
    return 0;
}
