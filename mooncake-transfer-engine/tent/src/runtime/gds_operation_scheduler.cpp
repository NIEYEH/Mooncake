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

#include <algorithm>
#include <limits>

namespace mooncake::tent {
namespace {

bool isTerminal(TransferStatusEnum status) {
    return status == COMPLETED || status == INVALID || status == CANCELED ||
           status == TIMEOUT || status == FAILED;
}

bool checkedAdd(size_t lhs, size_t rhs, size_t& result) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) return false;
    result = lhs + rhs;
    return true;
}

}  // namespace

bool gdsDispatchSegmentCanAppend(const GdsDispatchSegment& segment,
                                 size_t request_bytes,
                                 size_t max_requests,
                                 size_t max_bytes) {
    if (max_requests == 0 || max_bytes == 0 ||
        segment.requests >= max_requests ||
        request_bytes > max_bytes ||
        segment.bytes > max_bytes - request_bytes) {
        return false;
    }
    return true;
}

void gdsDispatchSegmentAppend(GdsDispatchSegment& segment,
                              size_t request_bytes) {
    ++segment.requests;
    segment.bytes += request_bytes;
}

GdsOperationScheduler::GdsOperationScheduler(
    GdsOperationSchedulerConfig config)
    : config_(config), config_status_(validateConfig()) {}

size_t GdsOperationScheduler::index(GdsDirection direction) {
    return direction == GdsDirection::Read ? 0 : 1;
}

Status GdsOperationScheduler::validateConfig() const {
    if (config_.shared_tokens == 0 ||
        config_.read_standalone_tokens == 0 ||
        config_.write_standalone_tokens == 0 ||
        config_.contended_write_tokens == 0 ||
        config_.read_quantum_bytes == 0 ||
        config_.write_quantum_bytes == 0 ||
        config_.credit_cap_quanta == 0 ||
        config_.primary_read_tokens == 0 ||
        config_.primary_read_bytes == 0 ||
        config_.secondary_segment_requests == 0 ||
        config_.secondary_segment_bytes == 0) {
        return Status::InvalidArgument(
            "GDS operation scheduler limits must be non-zero" LOC_MARK);
    }
    if (config_.read_standalone_tokens > config_.shared_tokens ||
        config_.write_standalone_tokens > config_.shared_tokens ||
        config_.contended_write_tokens >
            config_.write_standalone_tokens ||
        config_.primary_read_tokens > config_.shared_tokens) {
        return Status::InvalidArgument(
            "GDS direction/operation tokens exceed shared tokens" LOC_MARK);
    }
    return Status::OK();
}

Status GdsOperationScheduler::enqueue(const GdsDispatchEntry& entry) {
    CHECK_STATUS(config_status_);
    if (entry.owner_id == 0 || entry.operation_owner_id == 0 ||
        entry.physical_bytes == 0 || entry.physical_tokens == 0) {
        return Status::InvalidArgument(
            "invalid GDS scheduler entry" LOC_MARK);
    }
    if (entry.physical_tokens > config_.shared_tokens) {
        return Status::InvalidArgument(
            "GDS entry exceeds shared physical tokens" LOC_MARK);
    }
    if (!known_owner_ids_.insert(entry.owner_id).second) {
        return Status::InvalidEntry(
            "duplicate GDS scheduler owner" LOC_MARK);
    }

    auto [operation_it, inserted] =
        operations_.try_emplace(entry.operation_owner_id);
    auto& operation = operation_it->second;
    if (inserted) {
        operation.direction = entry.direction;
    } else if (operation.direction != entry.direction) {
        known_owner_ids_.erase(entry.owner_id);
        return Status::InvalidArgument(
            "one GDS operation mixes READ and WRITE" LOC_MARK);
    } else if (operation.canceled) {
        known_owner_ids_.erase(entry.owner_id);
        return Status::InvalidEntry(
            "cannot enqueue a canceled GDS operation" LOC_MARK);
    }

    auto& order = entry.direction == GdsDirection::Read
                      ? read_operation_order_
                      : write_operation_order_;
    if (std::find(order.begin(), order.end(), entry.operation_owner_id) ==
        order.end()) {
        order.push_back(entry.operation_owner_id);
    }
    ++operation.queued_entries;
    directions_[index(entry.direction)].round_exhausted = false;
    queued_.push_back(entry);
    return Status::OK();
}

bool GdsOperationScheduler::hasQueued(GdsDirection direction) const {
    return std::any_of(queued_.begin(), queued_.end(),
                       [&](const GdsDispatchEntry& entry) {
                           auto operation_it =
                               operations_.find(entry.operation_owner_id);
                           return entry.direction == direction &&
                                  operation_it != operations_.end() &&
                                  !operation_it->second.canceled;
                       });
}

size_t GdsOperationScheduler::directionTokenLimit(
    GdsDirection direction, bool contended) const {
    if (direction == GdsDirection::Read) {
        if (contended &&
            config_.shared_tokens > config_.contended_write_tokens) {
            return std::min(
                config_.read_standalone_tokens,
                config_.shared_tokens - config_.contended_write_tokens);
        }
        return config_.read_standalone_tokens;
    }
    return contended ? config_.contended_write_tokens
                     : config_.write_standalone_tokens;
}

bool GdsOperationScheduler::directionHasCapacity(
    GdsDirection direction, bool contended) const {
    return directions_[index(direction)].outstanding_reserved_tokens <
           directionTokenLimit(direction, contended);
}

void GdsOperationScheduler::cleanupOperationOrder() {
    const auto cleanup = [&](std::deque<uint64_t>& order) {
        for (auto it = order.begin(); it != order.end();) {
            auto operation_it = operations_.find(*it);
            if (operation_it == operations_.end() ||
                (operation_it->second.queued_entries == 0 &&
                 operation_it->second.reserved_tokens == 0)) {
                it = order.erase(it);
            } else {
                ++it;
            }
        }
    };
    cleanup(read_operation_order_);
    cleanup(write_operation_order_);
}

uint64_t GdsOperationScheduler::primaryReadOperation() {
    cleanupOperationOrder();
    for (uint64_t operation_id : read_operation_order_) {
        auto operation_it = operations_.find(operation_id);
        if (operation_it != operations_.end() &&
            !operation_it->second.canceled) {
            return operation_id;
        }
    }
    return 0;
}

bool GdsOperationScheduler::canReserve(
    const GdsDispatchEntry& entry,
    const GdsDispatchBudget& budget) {
    size_t next_global_tokens = 0;
    size_t next_global_bytes = 0;
    const size_t global_tokens =
        directions_[0].outstanding_reserved_tokens +
        directions_[1].outstanding_reserved_tokens;
    const size_t global_bytes =
        directions_[0].outstanding_reserved_bytes +
        directions_[1].outstanding_reserved_bytes;
    if (!checkedAdd(global_tokens, entry.physical_tokens,
                    next_global_tokens) ||
        !checkedAdd(global_bytes, entry.physical_bytes, next_global_bytes)) {
        return false;
    }
    if (next_global_tokens > config_.shared_tokens ||
        next_global_tokens > budget.max_tokens ||
        next_global_bytes > budget.max_bytes) {
        return false;
    }

    const bool contended =
        hasQueued(GdsDirection::Read) && hasQueued(GdsDirection::Write);
    const auto& direction = directions_[index(entry.direction)];
    size_t next_direction_tokens = 0;
    const size_t budget_direction_tokens =
        entry.direction == GdsDirection::Read ? budget.max_read_tokens
                                              : budget.max_write_tokens;
    if (!checkedAdd(direction.outstanding_reserved_tokens,
                    entry.physical_tokens, next_direction_tokens) ||
        next_direction_tokens >
            directionTokenLimit(entry.direction, contended) ||
        next_direction_tokens > budget_direction_tokens ||
        entry.enqueue_sequence > budget.max_enqueue_sequence) {
        return false;
    }

    const auto operation_it = operations_.find(entry.operation_owner_id);
    if (operation_it == operations_.end() ||
        operation_it->second.canceled) {
        return false;
    }
    if (entry.direction == GdsDirection::Read &&
        entry.operation_owner_id == primaryReadOperation()) {
        size_t next_operation_tokens = 0;
        size_t next_operation_bytes = 0;
        if (!checkedAdd(operation_it->second.reserved_tokens,
                        entry.physical_tokens, next_operation_tokens) ||
            !checkedAdd(operation_it->second.reserved_bytes,
                        entry.physical_bytes, next_operation_bytes) ||
            next_operation_tokens > config_.primary_read_tokens ||
            next_operation_bytes > config_.primary_read_bytes) {
            return false;
        }
    }
    return true;
}

std::deque<GdsDispatchEntry>::iterator
GdsOperationScheduler::findCandidate(GdsDirection direction,
                                     const GdsDispatchBudget& budget,
                                     size_t secondary_requests,
                                     size_t secondary_bytes) {
    if (direction == GdsDirection::Write) {
        cleanupOperationOrder();
        for (uint64_t operation_id : write_operation_order_) {
            auto it = std::find_if(
                queued_.begin(), queued_.end(),
                [&](const GdsDispatchEntry& entry) {
                    return entry.operation_owner_id == operation_id &&
                           canReserve(entry, budget);
                });
            if (it != queued_.end()) return it;
        }
        return queued_.end();
    }

    const uint64_t primary = primaryReadOperation();
    if (primary != 0) {
        auto primary_it = std::find_if(
            queued_.begin(), queued_.end(),
            [&](const GdsDispatchEntry& entry) {
                return entry.operation_owner_id == primary &&
                       canReserve(entry, budget);
            });
        if (primary_it != queued_.end()) return primary_it;
    }

    // A secondary READ operation may consume one bounded segment only after
    // the primary cannot produce an immediately reservable physical IO.
    for (uint64_t operation_id : read_operation_order_) {
        if (operation_id == primary) continue;
        auto operation_it = operations_.find(operation_id);
        if (operation_it == operations_.end() ||
            operation_it->second.canceled ||
            operation_it->second.reserved_tokens >=
                config_.secondary_segment_requests ||
            operation_it->second.reserved_bytes >=
                config_.secondary_segment_bytes) {
            continue;
        }
        auto candidate = std::find_if(
            queued_.begin(), queued_.end(),
            [&](const GdsDispatchEntry& entry) {
                if (entry.operation_owner_id != operation_id ||
                    !canReserve(entry, budget)) {
                    return false;
                }
                return secondary_requests <
                           config_.secondary_segment_requests &&
                       entry.physical_bytes <=
                           config_.secondary_segment_bytes - secondary_bytes;
            });
        if (candidate != queued_.end()) return candidate;
    }
    return queued_.end();
}

void GdsOperationScheduler::enterOrLeaveContention(bool contended) {
    if (contended == contention_active_) return;
    contention_active_ = contended;
    for (auto& direction : directions_) {
        direction.deficit_bytes = 0;
        direction.round_credit_granted = false;
        direction.round_exhausted = false;
    }
    round_cursor_ = GdsDirection::Read;
}

void GdsOperationScheduler::grantRoundCredit(bool read_backlog,
                                             bool write_backlog) {
    const std::array<bool, 2> backlog{read_backlog, write_backlog};
    const std::array<size_t, 2> quantum{config_.read_quantum_bytes,
                                        config_.write_quantum_bytes};
    for (size_t direction_index = 0; direction_index < 2;
        ++direction_index) {
        if (!backlog[direction_index] ||
            directions_[direction_index].round_credit_granted) {
            continue;
        }
        auto& direction = directions_[direction_index];
        const size_t credit_cap =
            quantum[direction_index] * config_.credit_cap_quanta;
        const int64_t maximum_deficit =
            static_cast<int64_t>(direction.outstanding_reserved_bytes) +
            static_cast<int64_t>(credit_cap);
        if (direction.deficit_bytes >
            std::numeric_limits<int64_t>::max() -
                static_cast<int64_t>(quantum[direction_index])) {
            direction.deficit_bytes = maximum_deficit;
        } else {
            direction.deficit_bytes =
                std::min(maximum_deficit,
                         direction.deficit_bytes +
                             static_cast<int64_t>(
                                 quantum[direction_index]));
        }
        direction.round_credit_granted = true;
    }
}

void GdsOperationScheduler::advanceRoundIfDone(bool read_backlog,
                                               bool write_backlog) {
    const bool read_done =
        !read_backlog || directions_[0].round_exhausted;
    const bool write_done =
        !write_backlog || directions_[1].round_exhausted;
    if (!read_done || !write_done) return;
    directions_[0].round_credit_granted = false;
    directions_[1].round_credit_granted = false;
    directions_[0].round_exhausted = false;
    directions_[1].round_exhausted = false;
}

bool GdsOperationScheduler::canSpendWdrr(
    const GdsDispatchEntry& entry) const {
    const auto& direction = directions_[index(entry.direction)];
    const int64_t spendable =
        direction.deficit_bytes -
        static_cast<int64_t>(direction.outstanding_reserved_bytes);
    if (entry.physical_bytes <=
        static_cast<size_t>(std::max<int64_t>(0, spendable))) {
        return true;
    }
    // Jumbo exception: exactly one oversized physical IO may create bounded
    // negative debt, and only with positive credit and no older reservation.
    return direction.outstanding_reserved_tokens == 0 &&
           direction.deficit_bytes > 0;
}

GdsDirection GdsOperationScheduler::chooseWeightedDirection(
    bool read_backlog, bool write_backlog) {
    if (!read_backlog) return GdsDirection::Write;
    if (!write_backlog) return GdsDirection::Read;
    return round_cursor_;
}

GdsDispatchReservation GdsOperationScheduler::reserve(
    std::deque<GdsDispatchEntry>::iterator entry_it, bool wdrr_charged) {
    const auto entry = *entry_it;
    queued_.erase(entry_it);

    auto& direction = directions_[index(entry.direction)];
    direction.outstanding_reserved_bytes += entry.physical_bytes;
    direction.outstanding_reserved_tokens += entry.physical_tokens;
    auto& operation = operations_.at(entry.operation_owner_id);
    --operation.queued_entries;
    operation.reserved_bytes += entry.physical_bytes;
    operation.reserved_tokens += entry.physical_tokens;

    GdsDispatchReservation reservation{
        next_reservation_id_++, entry.owner_id, entry.operation_owner_id,
        entry.direction, entry.physical_bytes, entry.physical_tokens};
    reservations_.emplace(
        reservation.id,
        ReservationState{reservation, false, wdrr_charged});
    return reservation;
}

std::vector<GdsDispatchReservation> GdsOperationScheduler::select(
    const GdsDispatchBudget& budget) {
    std::vector<GdsDispatchReservation> selected;
    if (!config_status_.ok() || budget.max_tokens == 0 ||
        budget.max_bytes == 0 || budget.max_entries == 0) {
        return selected;
    }

    size_t secondary_requests = 0;
    size_t secondary_bytes = 0;
    size_t no_progress_rounds = 0;
    while (selected.size() < budget.max_entries) {
        const bool read_backlog = hasQueued(GdsDirection::Read);
        const bool write_backlog = hasQueued(GdsDirection::Write);
        if (!read_backlog && !write_backlog) break;
        const bool contended =
            config_.mode == GdsSchedulerMode::WeightedFair &&
            read_backlog && write_backlog;
        enterOrLeaveContention(contended);

        GdsDirection direction = GdsDirection::Read;
        if (contended) {
            advanceRoundIfDone(read_backlog, write_backlog);
            grantRoundCredit(read_backlog, write_backlog);
            direction =
                chooseWeightedDirection(read_backlog, write_backlog);
        } else if (!read_backlog) {
            direction = GdsDirection::Write;
        }

        auto candidate =
            findCandidate(direction, budget, secondary_requests,
                          secondary_bytes);
        bool direction_capacity =
            directionHasCapacity(direction, contended);
        bool spendable =
            candidate != queued_.end() &&
            (!contended || canSpendWdrr(*candidate));
        if (!contended &&
            (candidate == queued_.end() || !direction_capacity) &&
            read_backlog && write_backlog) {
            const auto alternate =
                direction == GdsDirection::Read ? GdsDirection::Write
                                                : GdsDirection::Read;
            auto alternate_candidate =
                findCandidate(alternate, budget, secondary_requests,
                              secondary_bytes);
            const bool alternate_capacity =
                directionHasCapacity(alternate, false);
            if (alternate_candidate != queued_.end() &&
                alternate_capacity) {
                direction = alternate;
                candidate = alternate_candidate;
                direction_capacity = true;
                spendable = true;
            }
        }
        if (candidate == queued_.end() || !direction_capacity ||
            !spendable) {
            if (!contended) break;
            directions_[index(direction)].round_exhausted = true;
            round_cursor_ = direction == GdsDirection::Read
                                ? GdsDirection::Write
                                : GdsDirection::Read;
            ++no_progress_rounds;
            if (no_progress_rounds > 4) break;
            continue;
        }

        no_progress_rounds = 0;
        const uint64_t primary = primaryReadOperation();
        const bool is_secondary =
            direction == GdsDirection::Read &&
            candidate->operation_owner_id != primary;
        const size_t candidate_bytes = candidate->physical_bytes;
        auto reservation = reserve(candidate, contended);
        if (is_secondary) {
            ++secondary_requests;
            secondary_bytes += candidate_bytes;
        }
        selected.push_back(reservation);

        if (contended) {
            const auto next_candidate =
                findCandidate(direction, budget, secondary_requests,
                              secondary_bytes);
            if (next_candidate == queued_.end() ||
                !directionHasCapacity(direction, true) ||
                !canSpendWdrr(*next_candidate)) {
                directions_[index(direction)].round_exhausted = true;
                round_cursor_ = direction == GdsDirection::Read
                                    ? GdsDirection::Write
                                    : GdsDirection::Read;
            } else {
                directions_[index(direction)].round_exhausted = false;
            }
        }
    }
    return selected;
}

Status GdsOperationScheduler::complete(
    uint64_t reservation_id, size_t actual_transferred_bytes,
    TransferStatusEnum terminal_status) {
    if (!isTerminal(terminal_status)) {
        return Status::InvalidArgument(
            "GDS reservation completion must be terminal" LOC_MARK);
    }
    auto reservation_it = reservations_.find(reservation_id);
    if (reservation_it == reservations_.end() ||
        reservation_it->second.reconciled) {
        return Status::InvalidEntry(
            "duplicate or missing GDS reservation" LOC_MARK);
    }
    auto& reservation = reservation_it->second;
    if (actual_transferred_bytes > reservation.value.bytes) {
        return Status::InvalidArgument(
            "GDS completion exceeds reservation" LOC_MARK);
    }

    auto& direction = directions_[index(reservation.value.direction)];
    auto operation_it =
        operations_.find(reservation.value.operation_owner_id);
    if (direction.outstanding_reserved_bytes < reservation.value.bytes ||
        direction.outstanding_reserved_tokens < reservation.value.tokens ||
        operation_it == operations_.end() ||
        operation_it->second.reserved_bytes < reservation.value.bytes ||
        operation_it->second.reserved_tokens < reservation.value.tokens) {
        return Status::InternalError(
            "GDS reservation accounting underflow" LOC_MARK);
    }

    direction.outstanding_reserved_bytes -= reservation.value.bytes;
    direction.outstanding_reserved_tokens -= reservation.value.tokens;
    direction.completed_bytes += actual_transferred_bytes;
    if (reservation.wdrr_charged) {
        direction.deficit_bytes -=
            static_cast<int64_t>(actual_transferred_bytes);
    }
    auto& operation = operation_it->second;
    operation.reserved_bytes -= reservation.value.bytes;
    operation.reserved_tokens -= reservation.value.tokens;
    reservation.reconciled = true;

    resetIdleDirection(reservation.value.direction);
    cleanupOperationOrder();
    return Status::OK();
}

Status GdsOperationScheduler::cancelOperation(
    uint64_t operation_owner_id) {
    auto operation_it = operations_.find(operation_owner_id);
    if (operation_it == operations_.end()) {
        return Status::InvalidEntry(
            "GDS operation not found" LOC_MARK);
    }
    auto& operation = operation_it->second;
    if (operation.canceled) return Status::OK();
    operation.canceled = true;

    for (auto it = queued_.begin(); it != queued_.end();) {
        if (it->operation_owner_id == operation_owner_id) {
            known_owner_ids_.erase(it->owner_id);
            --operation.queued_entries;
            it = queued_.erase(it);
        } else {
            ++it;
        }
    }
    resetIdleDirection(operation.direction);
    cleanupOperationOrder();
    return Status::OK();
}

Status GdsOperationScheduler::retireOperation(
    uint64_t operation_owner_id) {
    if (operation_owner_id == 0) {
        return Status::InvalidArgument(
            "invalid GDS operation owner" LOC_MARK);
    }
    auto operation_it = operations_.find(operation_owner_id);
    if (operation_it == operations_.end()) return Status::OK();
    if (operation_it->second.queued_entries != 0 ||
        operation_it->second.reserved_tokens != 0 ||
        operation_it->second.reserved_bytes != 0) {
        return Status::InvalidEntry(
            "cannot retire active GDS operation" LOC_MARK);
    }
    for (auto reservation_it = reservations_.begin();
         reservation_it != reservations_.end();) {
        if (reservation_it->second.value.operation_owner_id !=
            operation_owner_id) {
            ++reservation_it;
            continue;
        }
        if (!reservation_it->second.reconciled) {
            return Status::InvalidEntry(
                "cannot retire operation with unreconciled reservation"
                LOC_MARK);
        }
        known_owner_ids_.erase(
            reservation_it->second.value.owner_id);
        reservation_it = reservations_.erase(reservation_it);
    }
    operations_.erase(operation_it);
    cleanupOperationOrder();
    return Status::OK();
}

void GdsOperationScheduler::resetIdleDirection(
    GdsDirection direction_value) {
    auto& direction = directions_[index(direction_value)];
    if (!hasQueued(direction_value) &&
        direction.outstanding_reserved_tokens == 0) {
        direction.deficit_bytes = 0;
        direction.round_credit_granted = false;
        direction.round_exhausted = false;
    }
}

GdsOperationSchedulerSnapshot GdsOperationScheduler::snapshot() const {
    GdsOperationSchedulerSnapshot result;
    for (const auto& entry : queued_) {
        ++result.queued_entries[index(entry.direction)];
    }
    for (size_t direction_index = 0; direction_index < 2;
         ++direction_index) {
        const auto& direction = directions_[direction_index];
        result.reserved_bytes[direction_index] =
            direction.outstanding_reserved_bytes;
        result.reserved_tokens[direction_index] =
            direction.outstanding_reserved_tokens;
        result.completed_bytes[direction_index] =
            direction.completed_bytes;
        result.spendable_deficit_bytes[direction_index] =
            direction.deficit_bytes -
            static_cast<int64_t>(direction.outstanding_reserved_bytes);
        result.global_reserved_bytes +=
            direction.outstanding_reserved_bytes;
        result.global_reserved_tokens +=
            direction.outstanding_reserved_tokens;
    }
    for (const auto& [operation_id, operation] : operations_) {
        result.operation_reserved_bytes.emplace(operation_id,
                                                operation.reserved_bytes);
        result.operation_reserved_tokens.emplace(operation_id,
                                                 operation.reserved_tokens);
    }
    return result;
}

}  // namespace mooncake::tent
