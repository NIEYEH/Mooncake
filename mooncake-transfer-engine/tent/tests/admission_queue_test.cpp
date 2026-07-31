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

#include "tent/runtime/admission_queue.h"

#include <algorithm>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace mooncake {
namespace tent {
namespace {

QueueOwnerInput makeOwner(
    size_t public_task_id, size_t length,
    QueueOwnerKind kind = QueueOwnerKind::User,
    std::vector<size_t> derived_task_ids = std::vector<size_t>()) {
    QueueOwnerInput owner;
    owner.owner_task_id = public_task_id;
    owner.derived_task_ids = std::move(derived_task_ids);
    owner.request.opcode = Request::WRITE;
    owner.request.source = nullptr;
    owner.request.target_id = 1;
    owner.request.target_offset = public_task_id * 4096;
    owner.request.length = length;
    owner.kind = kind;
    return owner;
}

QueueOwnerInput makeGdsOwner(size_t public_task_id, size_t length,
                             Request::OpCode opcode) {
    auto owner = makeOwner(public_task_id, length);
    owner.request.opcode = opcode;
    owner.transport = GDS;
    return owner;
}

QueueSubmit makeSubmit(uint64_t batch_token, size_t batch_slots_left,
                       std::vector<QueueOwnerInput> owners) {
    QueueSubmit submit;
    submit.batch_token = batch_token;
    submit.batch_slots_left = batch_slots_left;
    submit.owners = std::move(owners);
    return submit;
}

TEST(AdmissionQueueTest, AllowsEmptySubmitAsNoOp) {
    LocalTransferAdmissionQueue queue({2, 128, 0, 0});
    std::vector<QueueOwnerId> admitted_ids{99};

    auto status = queue.tryAdmit(makeSubmit(1, 0, {}), admitted_ids);

    EXPECT_EQ(status.code(), Status::Code::kOk);
    EXPECT_TRUE(admitted_ids.empty());
    EXPECT_EQ(queue.outstandingOwners(), 0u);
    EXPECT_EQ(queue.outstandingBytes(), 0u);
}

TEST(AdmissionQueueTest, RejectsSubmitWhenQueueLimitsAreInvalid) {
    LocalTransferAdmissionQueue queue({1, 128, 2, 0});
    std::vector<QueueOwnerId> admitted_ids{99};

    auto status =
        queue.tryAdmit(makeSubmit(1, 1, {makeOwner(0, 16)}), admitted_ids);

    EXPECT_EQ(status.code(), Status::Code::kInvalidArgument);
    EXPECT_TRUE(admitted_ids.empty());
    EXPECT_EQ(queue.outstandingOwners(), 0u);
    EXPECT_EQ(queue.outstandingBytes(), 0u);
}

TEST(AdmissionQueueTest, RejectsInvalidInputsWithoutPartialAdmission) {
    LocalTransferAdmissionQueue queue({4, 128, 0, 0});
    std::vector<QueueOwnerId> admitted_ids{99};

    auto status = queue.tryAdmit(
        makeSubmit(
            1, 2,
            {makeOwner(0, 16, QueueOwnerKind::User, {1}), makeOwner(1, 16)}),
        admitted_ids);

    EXPECT_EQ(status.code(), Status::Code::kInvalidArgument);
    EXPECT_TRUE(admitted_ids.empty());
    EXPECT_EQ(queue.outstandingOwners(), 0u);
    EXPECT_EQ(queue.outstandingBytes(), 0u);

    status = queue.tryAdmit(makeSubmit(1, 1, {makeOwner(2, 16)}), admitted_ids);

    ASSERT_EQ(status.code(), Status::Code::kOk);
    ASSERT_EQ(admitted_ids.size(), 1u);
    EXPECT_EQ(admitted_ids[0], 1u);
}

TEST(AdmissionQueueTest, RejectsUnsupportedOwnerKindWithoutPartialAdmission) {
    LocalTransferAdmissionQueue queue({4, 128, 0, 0});
    std::vector<QueueOwnerId> admitted_ids{99};

    auto invalid_owner = makeOwner(0, 16, static_cast<QueueOwnerKind>(99), {1});
    auto status = queue.tryAdmit(makeSubmit(1, 2, {std::move(invalid_owner)}),
                                 admitted_ids);

    EXPECT_EQ(status.code(), Status::Code::kInvalidArgument);
    EXPECT_TRUE(admitted_ids.empty());
    EXPECT_EQ(queue.outstandingOwners(), 0u);
    EXPECT_EQ(queue.outstandingBytes(), 0u);

    status = queue.tryAdmit(makeSubmit(1, 1, {makeOwner(2, 16)}), admitted_ids);

    ASSERT_EQ(status.code(), Status::Code::kOk);
    ASSERT_EQ(admitted_ids.size(), 1u);
    EXPECT_EQ(admitted_ids[0], 1u);
}

TEST(AdmissionQueueTest, RejectsUnsupportedOpcodeWithoutPartialAdmission) {
    LocalTransferAdmissionQueue queue({4, 128, 0, 0});
    std::vector<QueueOwnerId> admitted_ids{99};
    auto invalid_owner = makeOwner(0, 16);
    invalid_owner.request.opcode = static_cast<Request::OpCode>(99);

    auto status = queue.tryAdmit(
        makeSubmit(1, 1, {std::move(invalid_owner)}), admitted_ids);

    EXPECT_EQ(status.code(), Status::Code::kInvalidArgument);
    EXPECT_TRUE(admitted_ids.empty());
    EXPECT_EQ(queue.outstandingOwners(), 0u);
    EXPECT_EQ(queue.outstandingBytes(), 0u);
}

TEST(AdmissionQueueTest, RejectsCapacityExceededWithoutPartialAdmission) {
    LocalTransferAdmissionQueue queue({1, 64, 0, 0});
    std::vector<QueueOwnerId> admitted_ids;

    auto status = queue.tryAdmit(
        makeSubmit(1, 2, {makeOwner(0, 16), makeOwner(1, 16)}), admitted_ids);

    EXPECT_EQ(status.code(), Status::Code::kTooManyRequests);
    EXPECT_TRUE(admitted_ids.empty());
    EXPECT_EQ(queue.outstandingOwners(), 0u);
    EXPECT_EQ(queue.outstandingBytes(), 0u);

    status = queue.tryAdmit(makeSubmit(1, 1, {makeOwner(0, 16)}), admitted_ids);

    ASSERT_EQ(status.code(), Status::Code::kOk);
    ASSERT_EQ(admitted_ids.size(), 1u);
    EXPECT_EQ(admitted_ids[0], 1u);
}

TEST(AdmissionQueueTest, RejectsExistingPublicTaskConflictWithoutMutation) {
    LocalTransferAdmissionQueue queue({4, 128, 0, 0});
    std::vector<QueueOwnerId> admitted_ids;

    auto status =
        queue.tryAdmit(makeSubmit(1, 1, {makeOwner(0, 16)}), admitted_ids);
    ASSERT_EQ(status.code(), Status::Code::kOk);
    ASSERT_EQ(admitted_ids.size(), 1u);
    EXPECT_EQ(admitted_ids[0], 1u);

    status = queue.tryAdmit(
        makeSubmit(1, 2, {makeOwner(1, 16), makeOwner(0, 16)}), admitted_ids);

    EXPECT_EQ(status.code(), Status::Code::kInvalidEntry);
    EXPECT_TRUE(admitted_ids.empty());
    EXPECT_EQ(queue.outstandingOwners(), 1u);
    EXPECT_EQ(queue.outstandingBytes(), 16u);

    QueueOwnerId owner_id = 0;
    status = queue.resolveOwner(1, 1, owner_id);
    EXPECT_EQ(status.code(), Status::Code::kInvalidEntry);

    auto picked = queue.pickForDispatch(1, 16);
    ASSERT_EQ(picked.size(), 1u);
    status = queue.complete(picked[0], TransferStatusEnum::COMPLETED);
    ASSERT_EQ(status.code(), Status::Code::kOk);
    status = queue.retireBatch(1);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    status = queue.tryAdmit(makeSubmit(2, 1, {makeOwner(0, 16)}), admitted_ids);

    ASSERT_EQ(status.code(), Status::Code::kOk);
    ASSERT_EQ(admitted_ids.size(), 1u);
    EXPECT_EQ(admitted_ids[0], 2u);
}

TEST(AdmissionQueueTest, AccountsPublicSlotsSeparatelyFromQueueOwners) {
    LocalTransferAdmissionQueue queue({2, 128, 0, 0});
    std::vector<QueueOwnerId> admitted_ids;

    auto status = queue.tryAdmit(
        makeSubmit(1, 2, {makeOwner(7, 32, QueueOwnerKind::User, {8, 9})}),
        admitted_ids);

    EXPECT_EQ(status.code(), Status::Code::kTooManyRequests);
    EXPECT_TRUE(admitted_ids.empty());
    EXPECT_EQ(queue.outstandingOwners(), 0u);

    status = queue.tryAdmit(
        makeSubmit(1, 3, {makeOwner(7, 32, QueueOwnerKind::User, {8, 9})}),
        admitted_ids);

    ASSERT_EQ(status.code(), Status::Code::kOk);
    ASSERT_EQ(admitted_ids.size(), 1u);
    EXPECT_EQ(queue.outstandingOwners(), 1u);
    EXPECT_EQ(queue.outstandingBytes(), 32u);

    QueueOwnerId resolved_owner = 0;
    status = queue.resolveOwner(1, 7, resolved_owner);
    EXPECT_EQ(status.code(), Status::Code::kOk);
    EXPECT_EQ(resolved_owner, admitted_ids[0]);
    status = queue.resolveOwner(1, 8, resolved_owner);
    EXPECT_EQ(status.code(), Status::Code::kOk);
    EXPECT_EQ(resolved_owner, admitted_ids[0]);
    status = queue.resolveOwner(1, 9, resolved_owner);
    EXPECT_EQ(status.code(), Status::Code::kOk);
    EXPECT_EQ(resolved_owner, admitted_ids[0]);
    status = queue.resolveOwner(1, 0, resolved_owner);
    EXPECT_EQ(status.code(), Status::Code::kInvalidEntry);
}

TEST(AdmissionQueueTest, PreservesStagingReserveForStagingInternalOwners) {
    LocalTransferAdmissionQueue queue({2, 100, 1, 40});
    std::vector<QueueOwnerId> admitted_ids;

    auto status =
        queue.tryAdmit(makeSubmit(1, 1, {makeOwner(0, 60)}), admitted_ids);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    status = queue.tryAdmit(makeSubmit(2, 1, {makeOwner(0, 1)}), admitted_ids);
    EXPECT_EQ(status.code(), Status::Code::kTooManyRequests);
    EXPECT_TRUE(admitted_ids.empty());
    EXPECT_EQ(queue.outstandingOwners(), 1u);
    EXPECT_EQ(queue.outstandingBytes(), 60u);

    status = queue.tryAdmit(
        makeSubmit(3, 1, {makeOwner(0, 40, QueueOwnerKind::StagingInternal)}),
        admitted_ids);
    ASSERT_EQ(status.code(), Status::Code::kOk);
    ASSERT_EQ(admitted_ids.size(), 1u);
    EXPECT_EQ(admitted_ids[0], 2u);
    EXPECT_EQ(queue.outstandingOwners(), 2u);
    EXPECT_EQ(queue.outstandingBytes(), 100u);
}

TEST(AdmissionQueueTest, KeepsAdmissionOrderForDispatch) {
    LocalTransferAdmissionQueue queue({4, 128, 0, 0});
    std::vector<QueueOwnerId> admitted_ids;

    auto status = queue.tryAdmit(
        makeSubmit(1, 2,
                   {makeOwner(0, 60),
                    makeOwner(1, 10, QueueOwnerKind::StagingInternal)}),
        admitted_ids);

    ASSERT_EQ(status.code(), Status::Code::kOk);
    ASSERT_EQ(admitted_ids.size(), 2u);
    const std::vector<QueueOwnerId> expected_ids{1, 2};
    EXPECT_EQ(admitted_ids, expected_ids);

    EXPECT_TRUE(queue.pickForDispatch(2, 50).empty());

    auto picked = queue.pickForDispatch(2, 70);

    EXPECT_EQ(picked, expected_ids);
}

TEST(AdmissionQueueTest, ReportsCapacityByOwnerClass) {
    LocalTransferAdmissionQueue queue({4, 100, 1, 25});
    std::vector<QueueOwnerId> admitted_ids;

    auto user_capacity = queue.availableCapacity(QueueOwnerKind::User);
    EXPECT_EQ(user_capacity.owners, 3u);
    EXPECT_EQ(user_capacity.bytes, 75u);
    auto staging_capacity =
        queue.availableCapacity(QueueOwnerKind::StagingInternal);
    EXPECT_EQ(staging_capacity.owners, 4u);
    EXPECT_EQ(staging_capacity.bytes, 100u);

    ASSERT_TRUE(
        queue.tryAdmit(makeSubmit(1, 1, {makeOwner(0, 40)}), admitted_ids)
            .ok());
    user_capacity = queue.availableCapacity(QueueOwnerKind::User);
    EXPECT_EQ(user_capacity.owners, 2u);
    EXPECT_EQ(user_capacity.bytes, 35u);
    staging_capacity =
        queue.availableCapacity(QueueOwnerKind::StagingInternal);
    EXPECT_EQ(staging_capacity.owners, 3u);
    EXPECT_EQ(staging_capacity.bytes, 60u);
}

TEST(AdmissionQueueTest, UsesReadSlotsAndOneContendedWriteSlot) {
    LocalTransferAdmissionQueue queue({4, 128, 0, 0});
    std::vector<QueueOwnerId> admitted_ids;

    auto status = queue.tryAdmit(
        makeSubmit(1, 4,
                   {makeGdsOwner(0, 16, Request::WRITE),
                    makeGdsOwner(1, 16, Request::READ),
                    makeGdsOwner(2, 16, Request::WRITE),
                    makeGdsOwner(3, 16, Request::READ)}),
        admitted_ids);

    ASSERT_EQ(status.code(), Status::Code::kOk);
    auto picked = queue.pickForDispatch(4, 64);
    ASSERT_EQ(picked.size(), 3u);
    EXPECT_EQ(std::count(picked.begin(), picked.end(), 2u), 1);
    EXPECT_EQ(std::count(picked.begin(), picked.end(), 4u), 1);
    const bool dispatched_first_write =
        std::count(picked.begin(), picked.end(), 1u) == 1;
    const bool dispatched_second_write =
        std::count(picked.begin(), picked.end(), 3u) == 1;
    EXPECT_NE(dispatched_first_write, dispatched_second_write);
    for (const auto owner_id : picked) {
        ASSERT_TRUE(queue.complete(owner_id, COMPLETED).ok());
    }
    const auto second = queue.pickForDispatch(4, 64);
    ASSERT_EQ(second.size(), 1u);
    EXPECT_TRUE(second.front() == 1u || second.front() == 3u);
    EXPECT_NE(second.front() == 1u, dispatched_first_write);
}

TEST(AdmissionQueueTest, FixedModeReservesOneWriteTokenUnderContention) {
    LocalTransferAdmissionQueue queue({32, 1024, 0, 0});
    std::vector<QueueOwnerInput> owners;
    owners.push_back(makeGdsOwner(0, 16, Request::WRITE));
    for (size_t task_id = 1; task_id <= 17; ++task_id) {
        owners.push_back(makeGdsOwner(task_id, 16, Request::READ));
    }

    std::vector<QueueOwnerId> admitted_ids;
    const size_t owner_count = owners.size();
    auto status = queue.tryAdmit(
        makeSubmit(1, owner_count, std::move(owners)), admitted_ids);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    const auto picked = queue.pickForDispatch(owner_count, 1024);
    ASSERT_EQ(picked.size(), 16u);
    EXPECT_EQ(std::count(picked.begin(), picked.end(), 1u), 1);
}

TEST(AdmissionQueueTest, RestoresContendedWriteFloorBeforeRefillingRead) {
    LocalTransferAdmissionQueue queue({64, 4096, 0, 0});
    std::vector<QueueOwnerInput> owners;
    owners.push_back(makeGdsOwner(0, 16, Request::WRITE));
    owners.push_back(makeGdsOwner(1, 16, Request::WRITE));
    for (size_t task_id = 2; task_id < 20; ++task_id) {
        owners.push_back(makeGdsOwner(task_id, 16, Request::READ));
    }

    std::vector<QueueOwnerId> admitted_ids;
    const size_t owner_count = owners.size();
    ASSERT_TRUE(
        queue.tryAdmit(
                 makeSubmit(1, owner_count, std::move(owners)),
                 admitted_ids)
            .ok());
    const QueueOwnerId write0 = admitted_ids[0];
    const QueueOwnerId write1 = admitted_ids[1];
    const auto is_write = [&](QueueOwnerId owner_id) {
        return owner_id == write0 || owner_id == write1;
    };

    const auto initial = queue.pickForDispatch(16, 4096, 16, 1);
    ASSERT_EQ(initial.size(), 16u);
    ASSERT_EQ(std::count_if(initial.begin(), initial.end(), is_write), 1);
    const auto selected_write =
        std::find_if(initial.begin(), initial.end(), is_write);
    const auto selected_read = std::find_if(
        initial.begin(), initial.end(),
        [&](QueueOwnerId owner_id) { return !is_write(owner_id); });
    ASSERT_NE(selected_write, initial.end());
    ASSERT_NE(selected_read, initial.end());
    ASSERT_TRUE(queue.complete(*selected_read, COMPLETED).ok());
    ASSERT_TRUE(queue.complete(*selected_write, COMPLETED).ok());

    const auto before_refill = queue.gdsSchedulerSnapshot();
    ASSERT_EQ(before_refill.reserved_tokens[0], 14u);
    ASSERT_EQ(before_refill.reserved_tokens[1], 0u);

    const auto write_refill = queue.pickForDispatch(1, 4096, 1, 1);
    ASSERT_EQ(write_refill.size(), 1u);
    EXPECT_TRUE(is_write(write_refill.front()));

    const auto read_refill = queue.pickForDispatch(1, 4096, 1, 1);
    ASSERT_EQ(read_refill.size(), 1u);
    EXPECT_FALSE(is_write(read_refill.front()));
}

TEST(AdmissionQueueTest, OutstandingReservationsBoundSharedWindow) {
    LocalTransferAdmissionQueue queue({32, 1024, 0, 0});
    std::vector<QueueOwnerInput> owners;
    owners.push_back(makeGdsOwner(0, 32, Request::WRITE));
    for (size_t task_id = 1; task_id <= 17; ++task_id) {
        owners.push_back(makeGdsOwner(task_id, 16, Request::READ));
    }

    std::vector<QueueOwnerId> admitted_ids;
    const size_t owner_count = owners.size();
    auto status = queue.tryAdmit(
        makeSubmit(1, owner_count, std::move(owners)), admitted_ids);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    const auto initial = queue.pickForDispatch(owner_count, 1024);
    ASSERT_EQ(initial.size(), 16u);
    EXPECT_TRUE(queue.pickForDispatch(owner_count, 1024).empty());

    ASSERT_TRUE(queue.complete(initial.front(), COMPLETED).ok());
    const auto refill = queue.pickForDispatch(owner_count, 1024);
    ASSERT_EQ(refill.size(), 1u);
}

TEST(AdmissionQueueTest, GdsReadPriorityDoesNotBypassEarlierNonGdsOwner) {
    LocalTransferAdmissionQueue queue({3, 128, 0, 0});
    std::vector<QueueOwnerId> admitted_ids;
    auto status = queue.tryAdmit(
        makeSubmit(1, 3,
                   {makeGdsOwner(0, 16, Request::WRITE), makeOwner(1, 16),
                    makeGdsOwner(2, 16, Request::READ)}),
        admitted_ids);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    const std::vector<QueueOwnerId> expected_ids{1, 2, 3};
    EXPECT_EQ(queue.pickForDispatch(3, 48), expected_ids);
}

TEST(AdmissionQueueTest, ReportsWriteFloorMissBehindSequenceBarrier) {
    LocalTransferAdmissionQueue queue({3, 128, 0, 0});
    std::vector<QueueOwnerId> admitted_ids;
    auto status = queue.tryAdmit(
        makeSubmit(1, 3,
                   {makeGdsOwner(0, 16, Request::READ),
                    makeOwner(1, 16),
                    makeGdsOwner(2, 16, Request::WRITE)}),
        admitted_ids);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    const auto picked = queue.pickForDispatch(1, 48, 1, 1);
    ASSERT_EQ(picked.size(), 1u);
    EXPECT_EQ(picked.front(), 1u);
    const auto snapshot = queue.gdsSchedulerSnapshot();
    EXPECT_EQ(snapshot.fixed_write_floor_needed, 1u);
    EXPECT_EQ(snapshot.fixed_write_floor_dispatched, 0u);
    EXPECT_EQ(snapshot.fixed_write_floor_blocked, 1u);
    EXPECT_EQ(snapshot.gds_write_floor_missed, 1u);
    EXPECT_EQ(snapshot.last_select_max_enqueue_sequence, 1u);
}

TEST(AdmissionQueueTest, EnforcesIndependentGdsDirectionBudgets) {
    LocalTransferAdmissionQueue queue({8, 256, 0, 0});
    std::vector<QueueOwnerId> admitted_ids;
    auto status = queue.tryAdmit(
        makeSubmit(1, 6,
                   {makeGdsOwner(0, 16, Request::READ),
                    makeGdsOwner(1, 16, Request::READ),
                    makeGdsOwner(2, 16, Request::READ),
                    makeGdsOwner(3, 16, Request::WRITE),
                    makeGdsOwner(4, 16, Request::WRITE),
                    makeGdsOwner(5, 16, Request::WRITE)}),
        admitted_ids);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    const auto picked = queue.pickForDispatch(6, 256, 2, 1);
    ASSERT_EQ(picked.size(), 3u);
    EXPECT_EQ(std::count_if(picked.begin(), picked.end(),
                            [](QueueOwnerId owner_id) {
                                return owner_id >= 1 && owner_id <= 3;
                            }),
              2);
    EXPECT_EQ(std::count_if(picked.begin(), picked.end(),
                            [](QueueOwnerId owner_id) {
                                return owner_id >= 4 && owner_id <= 6;
                            }),
              1);
}

TEST(AdmissionQueueTest, RequiresDispatchBeforeTerminalCompletion) {
    LocalTransferAdmissionQueue queue({2, 128, 0, 0});
    std::vector<QueueOwnerId> admitted_ids;

    auto status =
        queue.tryAdmit(makeSubmit(1, 1, {makeOwner(0, 16)}), admitted_ids);
    ASSERT_EQ(status.code(), Status::Code::kOk);
    ASSERT_EQ(admitted_ids.size(), 1u);

    status = queue.complete(admitted_ids[0], TransferStatusEnum::COMPLETED);
    EXPECT_EQ(status.code(), Status::Code::kInvalidEntry);
    status = queue.complete(admitted_ids[0], TransferStatusEnum::PENDING);
    EXPECT_EQ(status.code(), Status::Code::kInvalidArgument);
    EXPECT_EQ(queue.outstandingOwners(), 1u);
    EXPECT_EQ(queue.outstandingBytes(), 16u);

    auto picked = queue.pickForDispatch(1, 16);
    ASSERT_EQ(picked.size(), 1u);

    status = queue.complete(picked[0], TransferStatusEnum::COMPLETED);
    EXPECT_EQ(status.code(), Status::Code::kOk);
    EXPECT_EQ(queue.outstandingOwners(), 0u);
    EXPECT_EQ(queue.outstandingBytes(), 0u);

    status = queue.complete(picked[0], TransferStatusEnum::COMPLETED);
    EXPECT_EQ(status.code(), Status::Code::kInvalidEntry);
}

TEST(AdmissionQueueTest, RetainsTerminalStatusUntilBatchRetire) {
    LocalTransferAdmissionQueue queue({2, 128, 0, 0});
    std::vector<QueueOwnerId> admitted_ids;

    auto status = queue.tryAdmit(
        makeSubmit(1, 2, {makeOwner(0, 16, QueueOwnerKind::User, {1})}),
        admitted_ids);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    TransferStatusEnum public_status = TransferStatusEnum::INVALID;
    status = queue.getPublicStatus(1, 1, public_status);
    EXPECT_EQ(status.code(), Status::Code::kOk);
    EXPECT_EQ(public_status, TransferStatusEnum::PENDING);

    auto picked = queue.pickForDispatch(1, 16);
    ASSERT_EQ(picked.size(), 1u);
    status = queue.complete(picked[0], TransferStatusEnum::FAILED);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    status = queue.getPublicStatus(1, 0, public_status);
    EXPECT_EQ(status.code(), Status::Code::kOk);
    EXPECT_EQ(public_status, TransferStatusEnum::FAILED);
    status = queue.getPublicStatus(1, 1, public_status);
    EXPECT_EQ(status.code(), Status::Code::kOk);
    EXPECT_EQ(public_status, TransferStatusEnum::FAILED);

    status = queue.retireBatch(1);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    QueueOwnerId resolved_owner = 0;
    status = queue.resolveOwner(1, 0, resolved_owner);
    EXPECT_EQ(status.code(), Status::Code::kInvalidEntry);
    status = queue.getPublicStatus(1, 1, public_status);
    EXPECT_EQ(status.code(), Status::Code::kInvalidEntry);
}

TEST(AdmissionQueueTest, RetainsSpecificTerminalStatus) {
    LocalTransferAdmissionQueue queue({1, 128, 0, 0});
    std::vector<QueueOwnerId> admitted_ids;

    auto status =
        queue.tryAdmit(makeSubmit(1, 1, {makeOwner(0, 16)}), admitted_ids);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    auto picked = queue.pickForDispatch(1, 16);
    ASSERT_EQ(picked.size(), 1u);
    status = queue.complete(picked[0], TransferStatusEnum::TIMEOUT);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    TransferStatusEnum public_status = TransferStatusEnum::PENDING;
    status = queue.getPublicStatus(1, 0, public_status);
    ASSERT_EQ(status.code(), Status::Code::kOk);
    EXPECT_EQ(public_status, TransferStatusEnum::TIMEOUT);
}

TEST(AdmissionQueueTest, RejectsRetireWithNonTerminalOwners) {
    LocalTransferAdmissionQueue queue({2, 128, 0, 0});
    std::vector<QueueOwnerId> admitted_ids;

    auto status = queue.tryAdmit(
        makeSubmit(1, 2, {makeOwner(0, 16), makeOwner(1, 16)}), admitted_ids);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    auto picked = queue.pickForDispatch(1, 16);
    ASSERT_EQ(picked.size(), 1u);
    status = queue.complete(picked[0], TransferStatusEnum::COMPLETED);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    status = queue.retireBatch(1);
    EXPECT_EQ(status.code(), Status::Code::kInvalidEntry);

    picked = queue.pickForDispatch(1, 16);
    ASSERT_EQ(picked.size(), 1u);
    status = queue.complete(picked[0], TransferStatusEnum::COMPLETED);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    status = queue.retireBatch(1);
    EXPECT_EQ(status.code(), Status::Code::kOk);
}

TEST(AdmissionQueueTest, AllowsBatchTokenReuseAfterRetire) {
    LocalTransferAdmissionQueue queue({1, 128, 0, 0});
    std::vector<QueueOwnerId> admitted_ids;

    auto status =
        queue.tryAdmit(makeSubmit(1, 1, {makeOwner(0, 16)}), admitted_ids);
    ASSERT_EQ(status.code(), Status::Code::kOk);
    ASSERT_EQ(admitted_ids.size(), 1u);
    EXPECT_EQ(admitted_ids[0], 1u);

    auto picked = queue.pickForDispatch(1, 16);
    ASSERT_EQ(picked.size(), 1u);
    status = queue.complete(picked[0], TransferStatusEnum::COMPLETED);
    ASSERT_EQ(status.code(), Status::Code::kOk);
    status = queue.retireBatch(1);
    ASSERT_EQ(status.code(), Status::Code::kOk);

    status = queue.tryAdmit(makeSubmit(1, 1, {makeOwner(0, 16)}), admitted_ids);

    ASSERT_EQ(status.code(), Status::Code::kOk);
    ASSERT_EQ(admitted_ids.size(), 1u);
    EXPECT_EQ(admitted_ids[0], 2u);

    QueueOwnerId resolved_owner = 0;
    status = queue.resolveOwner(1, 0, resolved_owner);
    EXPECT_EQ(status.code(), Status::Code::kOk);
    EXPECT_EQ(resolved_owner, 2u);
}

}  // namespace
}  // namespace tent
}  // namespace mooncake
