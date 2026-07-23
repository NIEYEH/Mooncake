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

#include <bits/stdint-uintn.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cufile.h>

#include "tent/runtime/control_plane.h"
#include "tent/runtime/transport.h"

namespace mooncake {
namespace tent {

class GdsFileContext;

struct IOParamRange {
    size_t base;
    size_t count;
};

// Wrapper for reusable CUfileBatchHandle_t
// cuFileBatchIOSetUp is expensive, so we reuse handles
struct BatchHandle {
    CUfileBatchHandle_t handle;
    int max_nr;  // max number of batch entries
    int device_id;  // CUDA device whose context created this handle
};

struct GdsSubBatch;

struct GdsIoBatch {
    struct IoRef {
        GdsSubBatch* owner;
        size_t param_index;
    };

    BatchHandle* batch_handle;
    int device_id;
    std::vector<CUfileIOParams_t> params;
    std::vector<IoRef> refs;
    std::vector<CUfileIOEvents_t> events;
    size_t consecutive_poll_errors{0};
    bool capacity_filled{false};
};

struct GdsSubBatch : public Transport::SubBatch {
    size_t max_size;
    std::vector<IOParamRange> io_param_ranges;
    std::vector<CUfileIOParams_t> io_params;
    std::vector<TransferStatusEnum> io_statuses;
    std::vector<size_t> io_transferred_bytes;
    virtual size_t size() const { return io_param_ranges.size(); }
};

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

   private:
    std::string getGdsFilePath(SegmentID handle);

    GdsFileContext* findFileContext(SegmentID target_id);

    Status validateRequest(const Request& request);

    bool findRegisteredBuffer(const void* addr, size_t length,
                              void*& registered_base,
                              size_t& registered_offset, int& device_id);

    Status resolveIoBuffer(const Request& request, void*& io_base,
                           size_t& io_offset, int& device_id);

    Status acquireBatchHandle(int device_id, BatchHandle*& handle);

    void releaseBatchHandle(BatchHandle* handle);

    struct PendingIo {
        GdsSubBatch* owner;
        size_t param_index;
        int device_id;
        CUfileIOParams_t params;
        std::chrono::steady_clock::time_point enqueued_at;
        // Zero uses the configured opcode limit. A failed multi-entry batch
        // is requeued with a smaller limit so one driver rejection does not
        // fail every logical request in the physical batch.
        size_t max_group_entries{0};
    };

    // All methods with the Locked suffix require scheduler_lock_. A global
    // scheduler is necessary because Store clients commonly submit one KV
    // object per SubBatch. Per-SubBatch cuFile queues turn that workload into
    // depth-1 physical I/O even when dozens of requests are waiting.
    Status dispatchPendingIoLocked();

    Status pollInflightIoLocked();

    void failPhysicalBatchLocked(GdsIoBatch& io_batch);

    bool subBatchHasWorkLocked(const GdsSubBatch* batch) const;

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
    size_t io_batch_depth_;
    size_t max_io_size_;
    size_t max_inflight_batches_;
    size_t read_batch_depth_;
    size_t write_batch_depth_;
    size_t max_read_batch_bytes_;
    size_t max_write_batch_bytes_;
    size_t submit_retry_count_;
    size_t max_status_poll_errors_;
    std::chrono::microseconds aggregation_delay_;
    std::chrono::microseconds status_poll_interval_;

    // The physical scheduler is transport-wide. It coalesces I/O from
    // independent logical SubBatches and enforces one bounded cuFile window.
    std::deque<PendingIo> pending_ios_;
    std::list<std::unique_ptr<GdsIoBatch>> inflight_io_batches_;
    std::chrono::steady_clock::time_point last_status_poll_{};
    bool dispatch_window_blocked_{false};
    std::mutex scheduler_lock_;

    // Object pool for BatchHandle to avoid frequent cuFileBatchIOSetUp/Destroy
    // CUfileBatchHandle_t is reusable per cuFile API documentation
    std::vector<BatchHandle*> handle_pool_;
    std::mutex handle_pool_lock_;

    // Track all allocated sub-batches to clean up on uninstall
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
