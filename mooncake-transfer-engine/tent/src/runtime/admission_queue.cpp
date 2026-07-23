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
#include <iterator>
#include <limits>
#include <set>

namespace mooncake {
namespace tent {
namespace {

using PublicTaskKey = std::pair<uint64_t, size_t>;

bool isSupportedTerminalStatus(TransferStatusEnum status) {
    return status == TransferStatusEnum::COMPLETED ||
           status == TransferStatusEnum::INVALID ||
           status == TransferStatusEnum::CANCELED ||
           status == TransferStatusEnum::TIMEOUT ||
           status == TransferStatusEnum::FAILED;
}

bool isSupportedOwnerKind(QueueOwnerKind kind) {
    switch (kind) {
        case QueueOwnerKind::User:
        case QueueOwnerKind::StagingInternal:
            return true;
    }
    return false;
}

bool isSupportedOpcode(Request::OpCode opcode) {
    return opcode == Request::READ || opcode == Request::WRITE;
}

GdsDirection toGdsDirection(Request::OpCode opcode) {
    return opcode == Request::READ ? GdsDirection::Read
                                   : GdsDirection::Write;
}

Status schedulerOperationId(uint64_t batch_token, Request::OpCode opcode,
                            uint64_t& operation_id) {
    if (batch_token > (std::numeric_limits<uint64_t>::max() >> 1)) {
        return Status::InvalidArgument(
            "batch token is too large for GDS direction key" LOC_MARK);
    }
    operation_id = (batch_token << 1) |
                   (opcode == Request::WRITE ? uint64_t{1} : uint64_t{0});
    return Status::OK();
}

Status checkedAdd(size_t lhs, size_t rhs, size_t& out) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return Status::InvalidArgument(
            "admission queue charge overflow" LOC_MARK);
    }
    out = lhs + rhs;
    return Status::OK();
}

Status validateLimits(const QueueLimits& limits) {
    if (limits.staging_owner_reserve > limits.max_outstanding_owners) {
        return Status::InvalidArgument(
            "staging owner reserve exceeds owner limit" LOC_MARK);
    }
    if (limits.staging_byte_reserve > limits.max_outstanding_bytes) {
        return Status::InvalidArgument(
            "staging byte reserve exceeds byte limit" LOC_MARK);
    }
    return Status::OK();
}

}  // namespace

LocalTransferAdmissionQueue::LocalTransferAdmissionQueue(
    QueueLimits limits, GdsOperationSchedulerConfig gds_scheduler_config)
    : limits_(limits),
      limits_status_(validateLimits(limits)),
      gds_scheduler_(gds_scheduler_config) {}

Status LocalTransferAdmissionQueue::tryAdmit(
    const QueueSubmit& submit, std::vector<QueueOwnerId>& admitted_owner_ids) {
    admitted_owner_ids.clear();
    CHECK_STATUS(limits_status_);
    CHECK_STATUS(gds_scheduler_.status());
    if (submit.batch_token == 0) {
        return Status::InvalidArgument("invalid batch token" LOC_MARK);
    }
    if (submit.owners.empty()) return Status::OK();

    std::set<PublicTaskKey> public_keys;
    size_t byte_charge = 0;
    size_t user_owner_charge = 0;
    size_t user_byte_charge = 0;

    for (const auto& owner : submit.owners) {
        if (!isSupportedOwnerKind(owner.kind)) {
            return Status::InvalidArgument(
                "unsupported queue owner kind" LOC_MARK);
        }
        if (!isSupportedOpcode(owner.request.opcode)) {
            return Status::InvalidArgument(
                "unsupported queue owner opcode" LOC_MARK);
        }
        if (owner.transport < UNSPEC || owner.transport >= kNumTransportTypes) {
            return Status::InvalidArgument(
                "unsupported queue owner transport" LOC_MARK);
        }
        if (owner.request.length == 0) {
            return Status::InvalidArgument("empty transfer request" LOC_MARK);
        }
        if (owner.transport == GDS &&
            (owner.physical_plan.physical_ios == 0 ||
             (owner.physical_plan.physical_bytes != 0 &&
              owner.physical_plan.physical_bytes != owner.request.length))) {
            return Status::InvalidArgument(
                "invalid GDS physical plan" LOC_MARK);
        }
        if (owner.transport == GDS) {
            uint64_t operation_id = 0;
            CHECK_STATUS(schedulerOperationId(
                submit.batch_token, owner.request.opcode, operation_id));
        }

        const PublicTaskKey owner_key{submit.batch_token, owner.owner_task_id};
        if (!public_keys.insert(owner_key).second) {
            return Status::InvalidArgument("duplicate public task id" LOC_MARK);
        }
        for (const auto derived_task_id : owner.derived_task_ids) {
            if (derived_task_id == owner.owner_task_id) {
                return Status::InvalidArgument(
                    "owner task id appears in derived task ids" LOC_MARK);
            }
            const PublicTaskKey derived_key{submit.batch_token,
                                            derived_task_id};
            if (!public_keys.insert(derived_key).second) {
                return Status::InvalidArgument(
                    "duplicate public task id" LOC_MARK);
            }
        }

        CHECK_STATUS(
            checkedAdd(byte_charge, owner.request.length, byte_charge));
        if (owner.kind == QueueOwnerKind::User) {
            CHECK_STATUS(checkedAdd(user_owner_charge, 1, user_owner_charge));
            CHECK_STATUS(checkedAdd(user_byte_charge, owner.request.length,
                                    user_byte_charge));
        }
    }

    if (public_keys.size() > submit.batch_slots_left) {
        return Status::TooManyRequests(
            "batch public task capacity exceeded" LOC_MARK);
    }

    for (const auto& key : public_keys) {
        if (public_to_owner_.count(key)) {
            return Status::InvalidEntry(
                "public task id already admitted" LOC_MARK);
        }
    }

    const size_t owner_charge = submit.owners.size();
    size_t next_outstanding_owners = 0;
    size_t next_outstanding_bytes = 0;
    size_t next_user_owners = 0;
    size_t next_user_bytes = 0;
    CHECK_STATUS(
        checkedAdd(outstanding_owners_, owner_charge, next_outstanding_owners));
    CHECK_STATUS(
        checkedAdd(outstanding_bytes_, byte_charge, next_outstanding_bytes));
    CHECK_STATUS(checkedAdd(outstanding_user_owners_, user_owner_charge,
                            next_user_owners));
    CHECK_STATUS(
        checkedAdd(outstanding_user_bytes_, user_byte_charge, next_user_bytes));

    const size_t user_owner_limit =
        limits_.max_outstanding_owners - limits_.staging_owner_reserve;
    const size_t user_byte_limit =
        limits_.max_outstanding_bytes - limits_.staging_byte_reserve;

    if (next_outstanding_owners > limits_.max_outstanding_owners) {
        return Status::TooManyRequests(
            "queue owner capacity exceeded" LOC_MARK);
    }
    if (next_outstanding_bytes > limits_.max_outstanding_bytes) {
        return Status::TooManyRequests("queue byte capacity exceeded" LOC_MARK);
    }
    if (next_user_owners > user_owner_limit) {
        return Status::TooManyRequests("user owner capacity exceeded" LOC_MARK);
    }
    if (next_user_bytes > user_byte_limit) {
        return Status::TooManyRequests("user byte capacity exceeded" LOC_MARK);
    }

    admitted_owner_ids.reserve(submit.owners.size());
    for (const auto& owner_input : submit.owners) {
        const QueueOwnerId owner_id = next_owner_id_++;
        QueueOwner owner;
        owner.batch_token = submit.batch_token;
        owner.request = owner_input.request;
        owner.transport = owner_input.transport;
        owner.kind = owner_input.kind;
        owner.physical_plan = owner_input.physical_plan;
        if (owner.physical_plan.physical_bytes == 0) {
            owner.physical_plan.physical_bytes = owner.request.length;
        }
        owners_.emplace(owner_id, owner);

        if (owner.transport == GDS) {
            uint64_t operation_id = 0;
            CHECK_STATUS(schedulerOperationId(
                submit.batch_token, owner.request.opcode, operation_id));
            CHECK_STATUS(gds_scheduler_.enqueue(GdsDispatchEntry{
                owner_id, operation_id,
                toGdsDirection(owner.request.opcode),
                owner.physical_plan.physical_bytes,
                owner.physical_plan.physical_ios, owner_id}));
        }

        public_to_owner_[{submit.batch_token, owner_input.owner_task_id}] =
            owner_id;
        for (const auto derived_task_id : owner_input.derived_task_ids) {
            public_to_owner_[{submit.batch_token, derived_task_id}] = owner_id;
        }
        fifo_.push_back(owner_id);
        admitted_owner_ids.push_back(owner_id);
    }

    outstanding_owners_ = next_outstanding_owners;
    outstanding_bytes_ = next_outstanding_bytes;
    outstanding_user_owners_ = next_user_owners;
    outstanding_user_bytes_ = next_user_bytes;
    return Status::OK();
}

std::vector<QueueOwnerId> LocalTransferAdmissionQueue::pickForDispatch(
    size_t max_owners, size_t max_bytes) {
    const size_t unlimited = std::numeric_limits<size_t>::max();
    return pickForDispatch(max_owners, max_bytes, unlimited, unlimited);
}

std::vector<QueueOwnerId> LocalTransferAdmissionQueue::pickForDispatch(
    size_t max_owners, size_t max_bytes, size_t max_gds_reads,
    size_t max_gds_writes) {
    std::vector<QueueOwnerId> picked;
    if (max_owners == 0 || max_bytes == 0) return picked;

    size_t used_owners = 0;
    size_t used_bytes = 0;
    size_t used_gds_reads = 0;
    size_t used_gds_writes = 0;
    while (used_owners < max_owners) {
        // Non-queued entries should not normally remain in fifo_, but stale
        // entries are removed defensively.
        for (auto fifo_it = fifo_.begin(); fifo_it != fifo_.end();) {
            auto owner_it = owners_.find(*fifo_it);
            if (owner_it == owners_.end() ||
                owner_it->second.state != QueueState::Queued) {
                fifo_it = fifo_.erase(fifo_it);
            } else {
                ++fifo_it;
            }
        }
        if (fifo_.empty()) {
            break;
        }

        const auto non_gds_it =
            std::find_if(fifo_.begin(), fifo_.end(), [&](QueueOwnerId owner_id) {
                return owners_.at(owner_id).transport != GDS;
            });
        const auto scheduler_snapshot = gds_scheduler_.snapshot();
        const size_t remaining_read_tokens =
            max_gds_reads > used_gds_reads
                ? max_gds_reads - used_gds_reads
                : 0;
        const size_t remaining_write_tokens =
            max_gds_writes > used_gds_writes
                ? max_gds_writes - used_gds_writes
                : 0;
        const size_t remaining_bytes = max_bytes - used_bytes;
        const uint64_t sequence_barrier =
            non_gds_it == fifo_.end()
                ? std::numeric_limits<uint64_t>::max()
                : (*non_gds_it == 0 ? 0 : *non_gds_it - 1);
        const size_t absolute_read_tokens =
            scheduler_snapshot.reserved_tokens[0] +
            remaining_read_tokens;
        const size_t absolute_write_tokens =
            scheduler_snapshot.reserved_tokens[1] +
            remaining_write_tokens;
        const size_t absolute_tokens =
            scheduler_snapshot.global_reserved_tokens +
            remaining_read_tokens + remaining_write_tokens;
        const size_t absolute_bytes =
            scheduler_snapshot.global_reserved_bytes + remaining_bytes;
        auto reservations = gds_scheduler_.select(
            {absolute_tokens, absolute_bytes, max_owners - used_owners,
             absolute_read_tokens, absolute_write_tokens,
             sequence_barrier});
        if (!reservations.empty()) {
            for (const auto& reservation : reservations) {
                auto owner_it = owners_.find(reservation.owner_id);
                if (owner_it == owners_.end() ||
                    owner_it->second.state != QueueState::Queued) {
                    return {};
                }
                auto fifo_it =
                    std::find(fifo_.begin(), fifo_.end(),
                              reservation.owner_id);
                if (fifo_it == fifo_.end()) return {};
                fifo_.erase(fifo_it);
                auto& owner = owner_it->second;
                owner.state = QueueState::Dispatching;
                owner.gds_reservation_id = reservation.id;
                picked.push_back(reservation.owner_id);
                ++used_owners;
                used_bytes += owner.request.length;
                if (owner.request.opcode == Request::READ) {
                    used_gds_reads += reservation.tokens;
                } else {
                    used_gds_writes += reservation.tokens;
                }
            }
            continue;
        }

        if (non_gds_it == fifo_.end()) break;
        auto owner_it = owners_.find(*non_gds_it);
        const auto& owner = owner_it->second;
        if (owner.request.length > remaining_bytes) break;
        const QueueOwnerId owner_id = *non_gds_it;
        fifo_.erase(non_gds_it);
        owner_it->second.state = QueueState::Dispatching;
        picked.push_back(owner_id);
        ++used_owners;
        used_bytes += owner.request.length;
    }
    return picked;
}

Status LocalTransferAdmissionQueue::complete(
    QueueOwnerId owner_id, TransferStatusEnum terminal_status) {
    auto owner_it = owners_.find(owner_id);
    if (owner_it == owners_.end()) {
        return Status::InvalidEntry("queue owner not found" LOC_MARK);
    }
    const size_t actual_transferred_bytes =
        terminal_status == TransferStatusEnum::COMPLETED
            ? owner_it->second.request.length
            : 0;
    return complete(owner_id, actual_transferred_bytes, terminal_status);
}

Status LocalTransferAdmissionQueue::complete(
    QueueOwnerId owner_id, size_t actual_transferred_bytes,
    TransferStatusEnum terminal_status) {
    if (owner_id == 0) {
        return Status::InvalidArgument("invalid queue owner id" LOC_MARK);
    }
    if (!isSupportedTerminalStatus(terminal_status)) {
        return Status::InvalidArgument("unsupported terminal status" LOC_MARK);
    }

    auto owner_it = owners_.find(owner_id);
    if (owner_it == owners_.end()) {
        return Status::InvalidEntry("queue owner not found" LOC_MARK);
    }
    auto& owner = owner_it->second;
    if (owner.state != QueueState::Dispatching) {
        return Status::InvalidEntry("queue owner is not dispatching" LOC_MARK);
    }
    if (owner.transport == GDS) {
        if (owner.gds_reservation_id == 0) {
            return Status::InternalError(
                "dispatching GDS owner has no reservation" LOC_MARK);
        }
        CHECK_STATUS(gds_scheduler_.complete(
            owner.gds_reservation_id, actual_transferred_bytes,
            terminal_status));
        owner.gds_reservation_id = 0;
    } else if (actual_transferred_bytes > owner.request.length) {
        return Status::InvalidArgument(
            "completion exceeds queue owner bytes" LOC_MARK);
    }

    owner.state = QueueState::Terminal;
    owner.terminal_status = terminal_status;
    --outstanding_owners_;
    outstanding_bytes_ -= owner.request.length;
    if (owner.kind == QueueOwnerKind::User) {
        --outstanding_user_owners_;
        outstanding_user_bytes_ -= owner.request.length;
    }
    return Status::OK();
}

Status LocalTransferAdmissionQueue::retireBatch(uint64_t batch_token) {
    if (batch_token == 0) {
        return Status::InvalidArgument("invalid batch token" LOC_MARK);
    }

    const PublicTaskKey batch_begin{batch_token, 0};
    auto public_begin = public_to_owner_.lower_bound(batch_begin);
    auto public_end = public_to_owner_.upper_bound(
        {batch_token, std::numeric_limits<size_t>::max()});

    std::set<QueueOwnerId> owner_ids;
    for (auto it = public_begin; it != public_end; ++it) {
        owner_ids.insert(it->second);
    }

    for (const auto owner_id : owner_ids) {
        auto owner_it = owners_.find(owner_id);
        if (owner_it == owners_.end()) {
            return Status::InternalError(
                "queue owner mapping is stale" LOC_MARK);
        }

        const auto& owner = owner_it->second;
        if (owner.batch_token != batch_token) {
            return Status::InternalError(
                "queue owner batch token mismatch" LOC_MARK);
        }
        if (owner.state != QueueState::Terminal) {
            return Status::InvalidEntry(
                "batch has non-terminal queue owners" LOC_MARK);
        }
    }

    for (const auto owner_id : owner_ids) {
        owners_.erase(owner_id);
    }
    public_to_owner_.erase(public_begin, public_end);
    return Status::OK();
}

Status LocalTransferAdmissionQueue::resolveOwner(uint64_t batch_token,
                                                 size_t public_task_id,
                                                 QueueOwnerId& owner_id) const {
    if (batch_token == 0) {
        return Status::InvalidArgument("invalid batch token" LOC_MARK);
    }
    auto it = public_to_owner_.find({batch_token, public_task_id});
    if (it == public_to_owner_.end()) {
        return Status::InvalidEntry("public task id not found" LOC_MARK);
    }
    owner_id = it->second;
    return Status::OK();
}

Status LocalTransferAdmissionQueue::getPublicStatus(
    uint64_t batch_token, size_t public_task_id,
    TransferStatusEnum& status) const {
    QueueOwnerId owner_id = 0;
    CHECK_STATUS(resolveOwner(batch_token, public_task_id, owner_id));
    auto owner_it = owners_.find(owner_id);
    if (owner_it == owners_.end()) {
        return Status::InternalError("queue owner mapping is stale" LOC_MARK);
    }
    switch (owner_it->second.state) {
        case QueueState::Queued:
        case QueueState::Dispatching:
            status = TransferStatusEnum::PENDING;
            break;
        case QueueState::Terminal:
            status = owner_it->second.terminal_status;
            break;
    }
    return Status::OK();
}

size_t LocalTransferAdmissionQueue::outstandingOwners() const {
    return outstanding_owners_;
}

size_t LocalTransferAdmissionQueue::outstandingBytes() const {
    return outstanding_bytes_;
}

QueueCapacity LocalTransferAdmissionQueue::availableCapacity(
    QueueOwnerKind kind) const {
    if (!limits_status_.ok()) return {};
    QueueCapacity capacity{
        limits_.max_outstanding_owners > outstanding_owners_
            ? limits_.max_outstanding_owners - outstanding_owners_
            : 0,
        limits_.max_outstanding_bytes > outstanding_bytes_
            ? limits_.max_outstanding_bytes - outstanding_bytes_
            : 0};
    if (kind != QueueOwnerKind::User) return capacity;

    const size_t user_owner_limit =
        limits_.max_outstanding_owners - limits_.staging_owner_reserve;
    const size_t user_byte_limit =
        limits_.max_outstanding_bytes - limits_.staging_byte_reserve;
    const size_t available_user_owners =
        user_owner_limit > outstanding_user_owners_
            ? user_owner_limit - outstanding_user_owners_
            : 0;
    const size_t available_user_bytes =
        user_byte_limit > outstanding_user_bytes_
            ? user_byte_limit - outstanding_user_bytes_
            : 0;
    capacity.owners = std::min(capacity.owners, available_user_owners);
    capacity.bytes = std::min(capacity.bytes, available_user_bytes);
    return capacity;
}

GdsOperationSchedulerSnapshot
LocalTransferAdmissionQueue::gdsSchedulerSnapshot() const {
    return gds_scheduler_.snapshot();
}

}  // namespace tent
}  // namespace mooncake
