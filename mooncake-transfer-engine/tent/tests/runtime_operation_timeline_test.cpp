// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "tent/runtime/runtime_operation_timeline.h"

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

void testTerminalSnapshotIsEmittedExactlyOnce() {
    RuntimeOperationTimeline timeline(77, GdsDirection::Read);
    for (size_t index = 0; index < 16; ++index) {
        EXPECT_TRUE(timeline.addPlanned(2359296, 1));
    }
    EXPECT_TRUE(timeline.recordDispatchSegment(7, 7 * 2359296));
    EXPECT_TRUE(timeline.recordDispatchSegment(7, 7 * 2359296));
    EXPECT_TRUE(timeline.recordDispatchSegment(2, 2 * 2359296));

    RuntimeOperationTerminal terminal;
    for (size_t index = 0; index < 15; ++index) {
        EXPECT_TRUE(timeline.recordCompletion(
            2359296, 1, COMPLETED));
        EXPECT_TRUE(!timeline.takeTerminal(terminal));
    }
    EXPECT_TRUE(timeline.recordCompletion(2359296, 1, COMPLETED));
    EXPECT_TRUE(timeline.takeTerminal(terminal));
    EXPECT_TRUE(!timeline.takeTerminal(terminal));
    EXPECT_EQ(terminal.logical_requests, 16u);
    EXPECT_EQ(terminal.dispatch_segments, 3u);
    EXPECT_EQ(terminal.completed_requests, 16u);
    EXPECT_EQ(terminal.failed_requests, 0u);
    EXPECT_EQ(terminal.planned_physical_ios, 16u);
    EXPECT_EQ(terminal.actual_completed_bytes, 16 * 2359296u);
}

void testPartialFailureSettlesOnlyMappedRequest() {
    RuntimeOperationTimeline timeline(88, GdsDirection::Read);
    EXPECT_TRUE(timeline.addPlanned(4096, 2));
    EXPECT_TRUE(timeline.addPlanned(4096, 2));
    EXPECT_TRUE(timeline.recordDispatchSegment(2, 8192));
    EXPECT_TRUE(timeline.recordCompletion(1024, 2, FAILED));

    RuntimeOperationTerminal terminal;
    EXPECT_TRUE(!timeline.takeTerminal(terminal));
    EXPECT_TRUE(timeline.recordCompletion(4096, 2, COMPLETED));
    EXPECT_TRUE(timeline.takeTerminal(terminal));
    EXPECT_EQ(terminal.completed_requests, 1u);
    EXPECT_EQ(terminal.failed_requests, 1u);
    EXPECT_EQ(terminal.actual_completed_bytes, 5120u);
}

}  // namespace
}  // namespace mooncake::tent

int main() {
    using namespace mooncake::tent;
    testTerminalSnapshotIsEmittedExactlyOnce();
    testPartialFailureSettlesOnlyMappedRequest();
    std::cout << "runtime_operation_timeline_test: PASS" << std::endl;
    return 0;
}
