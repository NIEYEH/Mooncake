// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tent/runtime/admission_queue.h"

#include <cstdlib>
#include <iostream>
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

GdsOperationSchedulerConfig schedulerConfig() {
    GdsOperationSchedulerConfig config;
    config.mode = GdsSchedulerMode::WeightedFair;
    config.shared_tokens = 16;
    config.read_standalone_tokens = 16;
    config.write_standalone_tokens = 2;
    config.contended_write_tokens = 1;
    config.primary_read_tokens = 16;
    config.primary_read_bytes = 48 * kMiB;
    return config;
}

QueueOwnerInput owner(size_t task_id, Request::OpCode opcode,
                      TransportType transport = GDS,
                      size_t bytes = 2359296, size_t tokens = 1) {
    QueueOwnerInput input;
    input.owner_task_id = task_id;
    input.request.opcode = opcode;
    input.request.target_id = 1;
    input.request.target_offset = task_id * 4096;
    input.request.length = bytes;
    input.transport = transport;
    input.physical_plan = {tokens, bytes};
    return input;
}

void admit(LocalTransferAdmissionQueue& queue, uint64_t operation,
           Request::OpCode opcode, size_t count,
           std::vector<QueueOwnerId>& admitted) {
    QueueSubmit submit;
    submit.batch_token = operation;
    submit.batch_slots_left = count;
    for (size_t index = 0; index < count; ++index) {
        submit.owners.push_back(owner(index, opcode));
    }
    EXPECT_TRUE(queue.tryAdmit(submit, admitted).ok());
}

void testPrimaryOperationFillsWindowBeforeSecondOperation() {
    LocalTransferAdmissionQueue queue(
        {64, 256 * kMiB, 0, 0}, schedulerConfig());
    std::vector<QueueOwnerId> first;
    std::vector<QueueOwnerId> second;
    admit(queue, 77, Request::READ, 32, first);
    admit(queue, 88, Request::READ, 32, second);

    auto picked =
        queue.pickForDispatch(16, 48 * kMiB, 16, 1);
    EXPECT_EQ(picked.size(), 16u);
    for (size_t index = 0; index < picked.size(); ++index) {
        EXPECT_EQ(picked[index], first[index]);
    }
}

void testPhysicalTokensBoundSelection() {
    LocalTransferAdmissionQueue queue(
        {16, 256 * kMiB, 0, 0}, schedulerConfig());
    QueueSubmit submit;
    submit.batch_token = 91;
    submit.batch_slots_left = 8;
    for (size_t index = 0; index < 8; ++index) {
        submit.owners.push_back(
            owner(index, Request::READ, GDS, 2 * kMiB, 3));
    }
    std::vector<QueueOwnerId> admitted;
    EXPECT_TRUE(queue.tryAdmit(submit, admitted).ok());
    auto picked =
        queue.pickForDispatch(16, 48 * kMiB, 16, 1);
    EXPECT_EQ(picked.size(), 5u);
}

void testMultiPhysicalWriteReservesAvailableConcurrency() {
    auto config = schedulerConfig();
    config.mode = GdsSchedulerMode::Fixed;
    config.write_standalone_tokens = 1;
    config.contended_write_tokens = 1;
    LocalTransferAdmissionQueue queue(
        {4, 64 * kMiB, 0, 0}, config);
    QueueSubmit submit;
    submit.batch_token = 95;
    submit.batch_slots_left = 1;
    submit.owners.push_back(
        owner(0, Request::WRITE, GDS, 2359296, 3));
    std::vector<QueueOwnerId> admitted;
    EXPECT_TRUE(queue.tryAdmit(submit, admitted).ok());

    const auto picked =
        queue.pickForDispatch(1, 48 * kMiB, 16, 1);
    EXPECT_EQ(picked.size(), 1u);
    size_t tokens = 0;
    EXPECT_TRUE(
        queue.getGdsReservationTokens(picked.front(), tokens).ok());
    EXPECT_EQ(tokens, 1u);
}

void testActualBytesReconcileReservation() {
    LocalTransferAdmissionQueue queue(
        {4, 64 * kMiB, 0, 0}, schedulerConfig());
    std::vector<QueueOwnerId> admitted;
    admit(queue, 92, Request::READ, 1, admitted);
    auto picked =
        queue.pickForDispatch(16, 48 * kMiB, 16, 1);
    EXPECT_EQ(picked.size(), 1u);
    EXPECT_TRUE(queue.complete(picked.front(), 1 * kMiB, FAILED).ok());
    const auto snapshot = queue.gdsSchedulerSnapshot();
    EXPECT_EQ(snapshot.completed_bytes[0], 1 * kMiB);
    EXPECT_EQ(snapshot.reserved_tokens[0], 0u);
}

void testTerminalOwnerPreservesPartialBytes() {
    LocalTransferAdmissionQueue queue(
        {4, 64 * kMiB, 0, 0}, schedulerConfig());
    QueueSubmit submit;
    submit.batch_token = 96;
    submit.batch_slots_left = 1;
    submit.owners.push_back(
        owner(0, Request::READ, GDS, 3 * kMiB));
    std::vector<QueueOwnerId> admitted;
    EXPECT_TRUE(queue.tryAdmit(submit, admitted).ok());
    const auto picked =
        queue.pickForDispatch(1, 48 * kMiB, 16, 1);
    EXPECT_EQ(picked.size(), 1u);
    EXPECT_TRUE(queue.complete(picked.front(), kMiB, FAILED).ok());

    TransferStatusEnum status = PENDING;
    size_t actual_transferred_bytes = 0;
    EXPECT_TRUE(
        queue.getPublicStatus(96, 0, status,
                              &actual_transferred_bytes)
            .ok());
    EXPECT_EQ(status, FAILED);
    EXPECT_EQ(actual_transferred_bytes, kMiB);
}

void testMixedDirectionPublicBatchRemainsValid() {
    LocalTransferAdmissionQueue queue(
        {4, 64 * kMiB, 0, 0}, schedulerConfig());
    QueueSubmit submit;
    submit.batch_token = 93;
    submit.batch_slots_left = 2;
    submit.owners.push_back(owner(0, Request::WRITE));
    submit.owners.push_back(owner(1, Request::READ));
    std::vector<QueueOwnerId> admitted;
    EXPECT_TRUE(queue.tryAdmit(submit, admitted).ok());
}

void testNonGdsOwnerIsDispatchBarrier() {
    LocalTransferAdmissionQueue queue(
        {4, 64 * kMiB, 0, 0}, schedulerConfig());
    QueueSubmit submit;
    submit.batch_token = 94;
    submit.batch_slots_left = 3;
    submit.owners.push_back(owner(0, Request::WRITE));
    submit.owners.push_back(owner(1, Request::WRITE, TCP));
    submit.owners.push_back(owner(2, Request::READ));
    std::vector<QueueOwnerId> admitted;
    EXPECT_TRUE(queue.tryAdmit(submit, admitted).ok());

    auto picked =
        queue.pickForDispatch(3, 48 * kMiB, 16, 1);
    EXPECT_EQ(picked.size(), 3u);
    EXPECT_EQ(picked[0], admitted[0]);
    EXPECT_EQ(picked[1], admitted[1]);
    EXPECT_EQ(picked[2], admitted[2]);
}

}  // namespace
}  // namespace mooncake::tent

int main() {
    using namespace mooncake::tent;
    testPrimaryOperationFillsWindowBeforeSecondOperation();
    testPhysicalTokensBoundSelection();
    testMultiPhysicalWriteReservesAvailableConcurrency();
    testActualBytesReconcileReservation();
    testTerminalOwnerPreservesPartialBytes();
    testMixedDirectionPublicBatchRemainsValid();
    testNonGdsOwnerIsDispatchBarrier();
    std::cout << "admission_queue_operation_test: PASS" << std::endl;
    return 0;
}
