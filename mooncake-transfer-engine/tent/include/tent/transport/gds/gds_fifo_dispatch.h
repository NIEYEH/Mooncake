// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#ifndef TENT_TRANSPORT_GDS_FIFO_DISPATCH_H_
#define TENT_TRANSPORT_GDS_FIFO_DISPATCH_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace mooncake::tent {

struct GdsFifoDispatchState {
    size_t shared_limit{0};
    size_t read_limit{0};
    size_t write_limit{0};
    size_t inflight_reads{0};
    size_t inflight_writes{0};
};

struct GdsDirectIoOutcome {
    size_t transferred_bytes{0};
    bool completed{false};
};

// cuFile returns the transferred byte count or a negative error. Preserve a
// positive short result for reservation/WDRR reconciliation while keeping the
// request terminal status failed. A defensive cap prevents a malformed driver
// result from exceeding the request's reservation.
inline GdsDirectIoOutcome gdsDirectIoOutcome(
    int64_t direct_result, size_t requested_bytes) {
    if (direct_result <= 0 || requested_bytes == 0) return {};
    const uint64_t result = static_cast<uint64_t>(direct_result);
    const size_t transferred =
        result > requested_bytes ? requested_bytes
                                 : static_cast<size_t>(result);
    return {transferred, result == requested_bytes};
}

inline size_t gdsFifoEffectiveWriteLimit(
    size_t configured_limit, size_t contended_limit,
    bool read_pressure) {
    return read_pressure ? std::min(configured_limit, contended_limit)
                         : configured_limit;
}

inline size_t gdsFifoSharedInflight(
    const GdsFifoDispatchState& state) {
    if (state.inflight_writes >
        std::numeric_limits<size_t>::max() - state.inflight_reads) {
        return std::numeric_limits<size_t>::max();
    }
    return state.inflight_reads + state.inflight_writes;
}

inline bool gdsFifoFrontCanDispatch(
    const GdsFifoDispatchState& state, bool write) {
    if (state.shared_limit == 0 ||
        gdsFifoSharedInflight(state) >= state.shared_limit) {
        return false;
    }
    return write ? state.inflight_writes < state.write_limit
                 : state.inflight_reads < state.read_limit;
}

inline bool gdsFifoFrontBlocksQueue(
    const GdsFifoDispatchState& state, bool write) {
    return !gdsFifoFrontCanDispatch(state, write);
}

inline bool gdsFifoReserve(GdsFifoDispatchState& state, bool write) {
    if (!gdsFifoFrontCanDispatch(state, write)) return false;
    if (write) {
        ++state.inflight_writes;
    } else {
        ++state.inflight_reads;
    }
    return true;
}

}  // namespace mooncake::tent

#endif  // TENT_TRANSPORT_GDS_FIFO_DISPATCH_H_
