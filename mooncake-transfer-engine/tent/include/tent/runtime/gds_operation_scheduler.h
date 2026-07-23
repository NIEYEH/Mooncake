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

#ifndef TENT_RUNTIME_GDS_OPERATION_SCHEDULER_H_
#define TENT_RUNTIME_GDS_OPERATION_SCHEDULER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tent/common/status.h"
#include "tent/common/types.h"

namespace mooncake::tent {

enum class GdsDirection : uint8_t {
    Read = 0,
    Write = 1,
};

enum class GdsSchedulerMode : uint8_t {
    Fixed = 0,
    WeightedFair = 1,
};

struct GdsOperationSchedulerConfig {
    GdsSchedulerMode mode{GdsSchedulerMode::Fixed};
    size_t shared_tokens{16};
    size_t read_standalone_tokens{16};
    size_t write_standalone_tokens{1};
    size_t contended_write_tokens{1};
    size_t read_quantum_bytes{8UL << 20};
    size_t write_quantum_bytes{2UL << 20};
    size_t credit_cap_quanta{2};
    size_t primary_read_tokens{16};
    size_t primary_read_bytes{48UL << 20};
    size_t secondary_segment_requests{8};
    size_t secondary_segment_bytes{16UL << 20};
};

struct GdsDispatchEntry {
    uint64_t owner_id{0};
    uint64_t operation_owner_id{0};
    GdsDirection direction{GdsDirection::Read};
    size_t physical_bytes{0};
    size_t physical_tokens{0};
    uint64_t enqueue_sequence{0};
};

struct GdsDispatchBudget {
    size_t max_tokens{0};
    size_t max_bytes{0};
    size_t max_entries{0};
    size_t max_read_tokens{std::numeric_limits<size_t>::max()};
    size_t max_write_tokens{std::numeric_limits<size_t>::max()};
    uint64_t max_enqueue_sequence{std::numeric_limits<uint64_t>::max()};
};

struct GdsDispatchReservation {
    uint64_t id{0};
    uint64_t owner_id{0};
    uint64_t operation_owner_id{0};
    GdsDirection direction{GdsDirection::Read};
    size_t bytes{0};
    size_t tokens{0};
};

struct GdsOperationSchedulerSnapshot {
    std::array<size_t, 2> queued_entries{};
    std::array<size_t, 2> reserved_bytes{};
    std::array<size_t, 2> reserved_tokens{};
    std::array<size_t, 2> completed_bytes{};
    std::array<int64_t, 2> spendable_deficit_bytes{};
    size_t global_reserved_bytes{0};
    size_t global_reserved_tokens{0};
    std::unordered_map<uint64_t, size_t> operation_reserved_bytes;
    std::unordered_map<uint64_t, size_t> operation_reserved_tokens;
};

// Single-threaded scheduling policy. TransferEngineImpl owns synchronization.
// The class contains no CUDA/cuFile state so reservation semantics can be
// tested on ordinary build hosts.
class GdsOperationScheduler {
   public:
    explicit GdsOperationScheduler(GdsOperationSchedulerConfig config);

    Status status() const { return config_status_; }

    Status enqueue(const GdsDispatchEntry& entry);

    std::vector<GdsDispatchReservation> select(
        const GdsDispatchBudget& budget);

    Status complete(uint64_t reservation_id, size_t actual_transferred_bytes,
                    TransferStatusEnum terminal_status);

    Status cancelOperation(uint64_t operation_owner_id);

    GdsOperationSchedulerSnapshot snapshot() const;

   private:
    struct DirectionState {
        int64_t deficit_bytes{0};
        size_t outstanding_reserved_bytes{0};
        size_t outstanding_reserved_tokens{0};
        size_t completed_bytes{0};
        bool round_credit_granted{false};
        bool round_exhausted{false};
    };

    struct OperationState {
        GdsDirection direction{GdsDirection::Read};
        size_t queued_entries{0};
        size_t reserved_bytes{0};
        size_t reserved_tokens{0};
        bool canceled{false};
    };

    struct ReservationState {
        GdsDispatchReservation value;
        bool reconciled{false};
        bool wdrr_charged{false};
    };

    static size_t index(GdsDirection direction);

    Status validateConfig() const;

    bool hasQueued(GdsDirection direction) const;

    bool directionHasCapacity(GdsDirection direction, bool contended) const;

    size_t directionTokenLimit(GdsDirection direction,
                               bool contended) const;

    uint64_t primaryReadOperation();

    std::deque<GdsDispatchEntry>::iterator findCandidate(
        GdsDirection direction, const GdsDispatchBudget& budget,
        size_t secondary_requests, size_t secondary_bytes);

    bool canReserve(const GdsDispatchEntry& entry,
                    const GdsDispatchBudget& budget);

    bool canSpendWdrr(const GdsDispatchEntry& entry) const;

    void enterOrLeaveContention(bool contended);

    void grantRoundCredit(bool read_backlog, bool write_backlog);

    void advanceRoundIfDone(bool read_backlog, bool write_backlog);

    GdsDirection chooseWeightedDirection(bool read_backlog,
                                         bool write_backlog);

    GdsDispatchReservation reserve(
        std::deque<GdsDispatchEntry>::iterator entry, bool wdrr_charged);

    void cleanupOperationOrder();

    void resetIdleDirection(GdsDirection direction);

    GdsOperationSchedulerConfig config_;
    Status config_status_;
    std::deque<GdsDispatchEntry> queued_;
    std::unordered_set<uint64_t> known_owner_ids_;
    std::unordered_map<uint64_t, OperationState> operations_;
    std::deque<uint64_t> read_operation_order_;
    std::deque<uint64_t> write_operation_order_;
    std::array<DirectionState, 2> directions_{};
    std::unordered_map<uint64_t, ReservationState> reservations_;
    uint64_t next_reservation_id_{1};
    GdsDirection round_cursor_{GdsDirection::Read};
    bool contention_active_{false};
};

}  // namespace mooncake::tent

#endif  // TENT_RUNTIME_GDS_OPERATION_SCHEDULER_H_
