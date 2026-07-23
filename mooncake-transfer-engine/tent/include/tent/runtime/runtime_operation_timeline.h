// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#ifndef TENT_RUNTIME_OPERATION_TIMELINE_H_
#define TENT_RUNTIME_OPERATION_TIMELINE_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "tent/common/types.h"
#include "tent/runtime/gds_operation_scheduler.h"

namespace mooncake::tent {

struct RuntimeOperationTerminal {
    uint64_t batch_token{0};
    GdsDirection direction{GdsDirection::Read};
    size_t logical_requests{0};
    size_t logical_bytes{0};
    size_t planned_physical_ios{0};
    size_t dispatch_segments{0};
    size_t dispatched_requests{0};
    size_t completed_requests{0};
    size_t failed_requests{0};
    size_t settled_physical_ios{0};
    size_t actual_completed_bytes{0};
    double queue_wait_us{0.0};
    double execution_us{0.0};
    double total_us{0.0};
};

class RuntimeOperationTimeline {
   public:
    using Clock = std::chrono::steady_clock;

    RuntimeOperationTimeline(
        uint64_t batch_token, GdsDirection direction,
        Clock::time_point queue_enter = Clock::now())
        : batch_token_(batch_token),
          direction_(direction),
          queue_enter_(queue_enter) {}

    bool addPlanned(size_t logical_bytes, size_t physical_ios) {
        return addPlannedBatch(1, logical_bytes, physical_ios);
    }

    bool addPlannedBatch(size_t logical_requests, size_t logical_bytes,
                         size_t physical_ios) {
        if (terminal_emitted_ || logical_requests == 0 ||
            logical_bytes == 0 || physical_ios == 0 ||
            addOverflows(logical_requests_, logical_requests) ||
            addOverflows(logical_bytes_, logical_bytes) ||
            addOverflows(planned_physical_ios_, physical_ios)) {
            return false;
        }
        logical_requests_ += logical_requests;
        logical_bytes_ += logical_bytes;
        planned_physical_ios_ += physical_ios;
        return true;
    }

    bool recordDispatchSegment(
        size_t requests, size_t bytes,
        Clock::time_point dispatched_at = Clock::now()) {
        if (terminal_emitted_ || requests == 0 || bytes == 0 ||
            requests > logical_requests_ - dispatched_requests_ ||
            addOverflows(dispatch_segments_, 1) ||
            addOverflows(dispatched_requests_, requests)) {
            return false;
        }
        if (!has_first_dispatch_) {
            first_dispatch_ = dispatched_at;
            has_first_dispatch_ = true;
        }
        ++dispatch_segments_;
        dispatched_requests_ += requests;
        return true;
    }

    bool recordCompletion(
        size_t actual_bytes, size_t settled_physical_ios,
        TransferStatusEnum status,
        Clock::time_point completed_at = Clock::now()) {
        if (terminal_emitted_ ||
            settled_requests_ >= logical_requests_ ||
            settled_physical_ios == 0 ||
            addOverflows(settled_physical_ios_,
                         settled_physical_ios) ||
            settled_physical_ios_ + settled_physical_ios >
                planned_physical_ios_ ||
            addOverflows(actual_completed_bytes_, actual_bytes)) {
            return false;
        }
        if (!has_first_completion_) {
            first_completion_ = completed_at;
            has_first_completion_ = true;
        }
        ++settled_requests_;
        settled_physical_ios_ += settled_physical_ios;
        actual_completed_bytes_ += actual_bytes;
        if (status == TransferStatusEnum::COMPLETED) {
            ++completed_requests_;
        } else {
            ++failed_requests_;
        }
        last_completion_ = completed_at;
        return true;
    }

    bool takeTerminal(RuntimeOperationTerminal& terminal) {
        if (terminal_emitted_ || logical_requests_ == 0 ||
            settled_requests_ != logical_requests_) {
            return false;
        }
        terminal_emitted_ = true;
        const auto terminal_time =
            has_first_completion_ ? last_completion_ : Clock::now();
        terminal.batch_token = batch_token_;
        terminal.direction = direction_;
        terminal.logical_requests = logical_requests_;
        terminal.logical_bytes = logical_bytes_;
        terminal.planned_physical_ios = planned_physical_ios_;
        terminal.dispatch_segments = dispatch_segments_;
        terminal.dispatched_requests = dispatched_requests_;
        terminal.completed_requests = completed_requests_;
        terminal.failed_requests = failed_requests_;
        terminal.settled_physical_ios = settled_physical_ios_;
        terminal.actual_completed_bytes = actual_completed_bytes_;
        terminal.queue_wait_us =
            has_first_dispatch_
                ? micros(first_dispatch_ - queue_enter_)
                : 0.0;
        terminal.execution_us =
            has_first_dispatch_
                ? micros(terminal_time - first_dispatch_)
                : 0.0;
        terminal.total_us = micros(terminal_time - queue_enter_);
        return true;
    }

   private:
    static bool addOverflows(size_t lhs, size_t rhs) {
        return rhs > std::numeric_limits<size_t>::max() - lhs;
    }

    template <typename Duration>
    static double micros(Duration duration) {
        return std::chrono::duration<double, std::micro>(
                   duration)
            .count();
    }

    uint64_t batch_token_{0};
    GdsDirection direction_{GdsDirection::Read};
    Clock::time_point queue_enter_{};
    Clock::time_point first_dispatch_{};
    Clock::time_point first_completion_{};
    Clock::time_point last_completion_{};
    bool has_first_dispatch_{false};
    bool has_first_completion_{false};
    bool terminal_emitted_{false};
    size_t logical_requests_{0};
    size_t logical_bytes_{0};
    size_t planned_physical_ios_{0};
    size_t dispatch_segments_{0};
    size_t dispatched_requests_{0};
    size_t settled_requests_{0};
    size_t completed_requests_{0};
    size_t failed_requests_{0};
    size_t settled_physical_ios_{0};
    size_t actual_completed_bytes_{0};
};

}  // namespace mooncake::tent

#endif  // TENT_RUNTIME_OPERATION_TIMELINE_H_
