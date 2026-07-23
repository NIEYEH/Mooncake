// Copyright 2025 KVCache.AI
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

#ifndef GDS_TRANSPORT_H_
#define GDS_TRANSPORT_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/types.h>

#include <cufile.h>

#include "tent/common/concurrent/thread_pool.h"
#include "tent/runtime/control_plane.h"
#include "tent/runtime/transport.h"
#include "tent/transport/gds/gds_fifo_dispatch.h"

namespace mooncake {
namespace tent {

class GdsFileContext;

struct IOParamRange {
    size_t base;
    size_t count;
};

struct GdsSubBatch : public Transport::SubBatch {
    size_t max_size;
    std::vector<IOParamRange> io_param_ranges;
    std::vector<TransferStatusEnum> io_statuses;
    std::vector<size_t> io_transferred_bytes;
    virtual size_t size() const { return io_param_ranges.size(); }
};

enum class GdsAdaptiveAction {
    NONE,
    REDUCE,
    RECOVER,
    HOLD_AT_MINIMUM
};

// Pure adaptive-control state and helpers are intentionally independent of a
// CUDA device so overload behavior remains covered by ordinary unit tests.
struct GdsAdaptiveState {
    size_t configured_limit{1};
    size_t current_limit{1};
    size_t minimum_limit{1};
    size_t completions_since_evaluation{0};
    size_t healthy_windows{0};
    size_t degraded_at_minimum_windows{0};
    double target_p99_us{0.0};
    double baseline_p99_us{0.0};
    bool saturation_since_evaluation{false};
    std::deque<double> recent_io_latency_us;
};

GdsAdaptiveAction adjustGdsAdaptiveConcurrency(
    GdsAdaptiveState& state, double p99_us, size_t queued_ios,
    double degradation_ratio, double recovery_ratio,
    size_t recovery_windows);

bool gdsAdaptiveEvaluationReady(const GdsAdaptiveState& state,
                                size_t sample_window,
                                size_t evaluation_interval);

size_t gdsAvailableWorkerSlots(size_t effective_limit,
                               size_t inflight_ios);

size_t gdsEffectiveWriteLimit(size_t adaptive_write_limit,
                              bool reads_active);

bool gdsReadWindowSaturated(size_t effective_read_limit,
                            size_t inflight_reads,
                            size_t inflight_writes,
                            bool write_is_at_minimum);

double gdsNearestRankP99(std::vector<double> samples);

class GdsTransport : public Transport {
   public:
    GdsTransport();

    ~GdsTransport();

    virtual Status install(std::string& local_segment_name,
                           std::shared_ptr<ControlService> metadata,
                           std::shared_ptr<Topology> local_topology,
                           std::shared_ptr<Config> conf = nullptr);

    virtual Status uninstall();

    virtual Status allocateSubBatch(SubBatchRef& batch, size_t max_size);

    virtual Status freeSubBatch(SubBatchRef& batch);

    virtual Status submitTransferTasks(
        SubBatchRef batch, const std::vector<Request>& request_list);

    virtual Status getTransferStatus(SubBatchRef batch, int task_id,
                                     TransferStatus& status);

    virtual Status addMemoryBuffer(BufferDesc& desc,
                                   const MemoryOptions& options);

    virtual Status removeMemoryBuffer(BufferDesc& desc);

    virtual const char* getName() const { return "gds"; }

    Status planRuntimeQueueRequest(const Request& request,
                                   RuntimeQueuePlan& plan) override;

    size_t runtimeQueueDispatchLimit(Request::OpCode opcode) const override;

    Status setRuntimeQueueContendedWriteLimit(size_t tokens) override;

    void updateRuntimeQueueDepth(size_t queued_reads,
                                 size_t queued_writes) override;

   private:
    std::string getGdsFilePath(SegmentID handle);

    GdsFileContext* findFileContext(SegmentID target_id);

    Status validateRequest(const Request& request);

    bool findRegisteredBuffer(const void* addr, size_t length,
                              void*& registered_base,
                              size_t& registered_offset, int& device_id);

    Status resolveIoBuffer(const Request& request, void*& io_base,
                           size_t& io_offset, int& device_id);

    size_t ioChunkSizeForDevice(int device_id) const;

    struct IoOperation {
        CUfileHandle_t file_handle;
        void* dev_ptr_base;
        size_t size;
        off_t file_offset;
        off_t dev_ptr_offset;
        bool write;
    };

    struct PendingIo {
        GdsSubBatch* owner;
        size_t param_index;
        int device_id;
        IoOperation operation;
        std::chrono::steady_clock::time_point enqueued_at;
    };

    struct DirectIo {
        uint64_t id;
        GdsSubBatch* owner;
        size_t param_index;
        int device_id;
        IoOperation operation;
        std::chrono::steady_clock::time_point enqueued_at;
        bool dispatched_at_capacity{false};
    };

    // Runtime may pass several logical requests at once, but this scheduler
    // always dequeues and submits them as independent synchronous cuFile IOs.
    // No cuFile Batch API handle is created by this transport.
    Status dispatchPendingIoLocked();

    void executeDirectIo(std::shared_ptr<DirectIo> direct_io);

    Status pollInflightIoLocked();

    bool subBatchHasWorkLocked(const GdsSubBatch* batch) const;

    void updateIoMetricsLocked() const;

    void recordAdaptiveSampleLocked(bool write, double io_latency_us,
                                    bool dispatched_at_capacity);

    void evaluateAdaptiveConcurrencyLocked(bool write);

    void markSaturatedIoLocked();

    void recordIoSummaryLocked(bool write, size_t transferred_bytes,
                               bool success, double queue_wait_us,
                               double io_latency_us,
                               double total_latency_us);

    void maybeLogIoSummaryLocked(
        std::chrono::steady_clock::time_point now);

   private:
    bool installed_;
    std::string local_segment_name_;
    std::shared_ptr<Topology> local_topology_;
    std::shared_ptr<ControlService> metadata_;
    std::shared_ptr<Config> conf_;

    RWSpinlock file_context_lock_;
    using FileContextMap =
        std::unordered_map<SegmentID, std::shared_ptr<GdsFileContext>>;
    FileContextMap file_context_map_;
    size_t max_io_size_;
    size_t read_worker_threads_;
    size_t write_worker_threads_;
    size_t max_inflight_ios_;
    size_t max_inflight_reads_;
    size_t max_inflight_writes_;
    size_t runtime_contended_write_limit_{1};
    size_t submit_retry_count_;
    bool merge_shadow_enabled_;
    bool adaptive_concurrency_enabled_;
    size_t adaptive_sample_window_;
    size_t adaptive_evaluation_interval_;
    double adaptive_degradation_ratio_;
    double adaptive_recovery_ratio_;
    size_t adaptive_recovery_windows_;

    // Runtime owns direction/operation scheduling. Transport preserves that
    // order in one FIFO and only enforces shared-device plus worker-pool
    // resource limits.
    std::deque<PendingIo> pending_ios_;
    size_t pending_read_ios_{0};
    size_t pending_write_ios_{0};
    size_t pending_read_bytes_{0};
    size_t pending_write_bytes_{0};
    std::unordered_map<uint64_t, std::shared_ptr<DirectIo>>
        inflight_direct_reads_;
    std::unordered_map<uint64_t, std::shared_ptr<DirectIo>>
        inflight_direct_writes_;
    size_t active_read_workers_{0};
    size_t active_write_workers_{0};
    GdsAdaptiveState read_adaptive_;
    GdsAdaptiveState write_adaptive_;
    uint64_t next_direct_io_id_{1};
    bool read_window_blocked_{false};
    bool write_window_blocked_{false};
    mutable std::mutex scheduler_lock_;
    std::unique_ptr<ThreadPool> read_thread_pool_;
    std::unique_ptr<ThreadPool> write_thread_pool_;
    std::atomic<bool> shutting_down_{false};
    std::atomic<size_t> runtime_queued_reads_{0};
    std::atomic<size_t> runtime_queued_writes_{0};

    struct IoSummaryDirection {
        size_t completions{0};
        size_t failures{0};
        size_t bytes{0};
        size_t peak_active_workers{0};
        size_t samples_seen{0};
        std::vector<double> queue_wait_us;
        std::vector<double> io_latency_us;
        std::vector<double> total_latency_us;
    };
    IoSummaryDirection read_io_summary_;
    IoSummaryDirection write_io_summary_;
    std::chrono::steady_clock::time_point io_summary_started_at_;
    std::chrono::steady_clock::time_point
        last_read_minimum_degraded_warning_at_;
    std::chrono::steady_clock::time_point
        last_write_minimum_degraded_warning_at_;

    // Track all allocated sub-batches to clean up on uninstall.
    std::vector<GdsSubBatch*> allocated_batches_;
    std::mutex allocated_batches_lock_;

    struct RegisteredBuffer {
        size_t length;
        int device_id;
    };
    std::map<std::uintptr_t, RegisteredBuffer> registered_buffers_;
    std::mutex registered_buffers_lock_;
};
}  // namespace tent
}  // namespace mooncake

#endif  // GDS_TRANSPORT_H_
