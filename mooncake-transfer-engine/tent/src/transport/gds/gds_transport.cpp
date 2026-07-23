// Copyright 2024 KVCache.AI
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

#include "tent/transport/gds/gds_transport.h"

#include <bits/stdint-uintn.h>
#include <glog/logging.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iomanip>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "tent/runtime/slab.h"
#include "tent/metrics/tent_metrics.h"

namespace mooncake {
namespace tent {
namespace {

constexpr size_t kMaxCuFileBatchDepth = 64;
constexpr size_t kDefaultInflightCuFileBatches = 1;
constexpr size_t kMaxInflightCuFileBatches = 16;
constexpr size_t kSafeUnregisteredBatchIoSize = 960 * 1024;
constexpr size_t kDefaultReadBatchDepth = 64;
constexpr size_t kDefaultWriteBatchDepth = 32;
constexpr size_t kDefaultMaxReadBatchBytes = 256ull << 20;
constexpr size_t kDefaultMaxWriteBatchBytes = 64ull << 20;
constexpr size_t kDefaultSubmitRetryCount = 1;
constexpr size_t kDefaultMaxStatusPollErrors = 20;
constexpr size_t kDefaultAggregationDelayUs = 50;
constexpr size_t kDefaultStatusPollIntervalUs = 50;
constexpr size_t kSubmitRetryBackoffUs = 50;

class CudaDeviceGuard {
   public:
    ~CudaDeviceGuard() {
        if (!restore_required_) return;
        auto result = cudaSetDevice(saved_device_);
        if (result != cudaSuccess) {
            LOG(ERROR) << "Failed to restore CUDA device " << saved_device_
                       << ": " << cudaGetErrorString(result);
        }
    }

    Status activate(int device_id) {
        if (device_id < 0) return Status::OK();

        auto result = cudaGetDevice(&saved_device_);
        if (result != cudaSuccess)
            return Status::InternalError(
                std::string("cudaGetDevice before GDS operation: ") +
                cudaGetErrorString(result) + LOC_MARK);

        // Even when the reported device is unchanged, this makes its primary
        // context current on a fresh vLLM worker thread.
        result = cudaSetDevice(device_id);
        if (result != cudaSuccess)
            return Status::InternalError(
                std::string("cudaSetDevice before GDS operation: ") +
                cudaGetErrorString(result) + LOC_MARK);

        restore_required_ = saved_device_ != device_id;
        return Status::OK();
    }

    Status restore() {
        if (!restore_required_) return Status::OK();
        auto result = cudaSetDevice(saved_device_);
        if (result != cudaSuccess)
            return Status::InternalError(
                std::string("cudaSetDevice after GDS operation: ") +
                cudaGetErrorString(result) + LOC_MARK);
        restore_required_ = false;
        return Status::OK();
    }

   private:
    int saved_device_ = -1;
    bool restore_required_ = false;
};

}  // namespace

class GdsFileContext {
   public:
    explicit GdsFileContext(const std::string& path) : ready_(false) {
        memset(&desc_, 0, sizeof(desc_));
        desc_.handle.fd = -1;
        int fd = open(path.c_str(), O_RDWR | O_DIRECT);
        if (fd < 0) {
            PLOG(ERROR) << "Failed to open GDS storage path " << path;
            return;
        }
        desc_.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        desc_.handle.fd = fd;
        auto result = cuFileHandleRegister(&handle_, &desc_);
        if (result.err != CU_FILE_SUCCESS) {
            TENT_RECORD_GDS_HANDLE_REGISTRATION_FAILED();
            LOG(ERROR) << "Failed to register GDS storage handle: Code "
                       << result.err;
            close(desc_.handle.fd);
            desc_.handle.fd = -1;
            return;
        }
        ready_ = true;
    }

    GdsFileContext(const GdsFileContext&) = delete;
    GdsFileContext& operator=(const GdsFileContext&) = delete;

    ~GdsFileContext() {
        if (handle_) cuFileHandleDeregister(handle_);
        if (desc_.handle.fd >= 0) close(desc_.handle.fd);
    }

    CUfileHandle_t getHandle() const { return handle_; }

    bool ready() const { return ready_; }

   private:
    CUfileHandle_t handle_ = NULL;
    CUfileDescr_t desc_;
    bool ready_;
};

TransferStatusEnum parseTransferStatus(CUfileStatus_t status) {
    switch (status) {
        case CUFILE_WAITING:
            return PENDING;
        case CUFILE_PENDING:
            return PENDING;
        case CUFILE_INVALID:
            return INVALID;
        case CUFILE_CANCELED:
            return CANCELED;
        case CUFILE_COMPLETE:
            return COMPLETED;
        case CUFILE_TIMEOUT:
            return TIMEOUT;
        case CUFILE_FAILED:
        default:
            return FAILED;
    }
}

GdsTransport::GdsTransport()
    : installed_(false),
      io_batch_depth_(0),
      max_io_size_(1ull << 20),
      max_inflight_batches_(kDefaultInflightCuFileBatches),
      read_batch_depth_(kDefaultReadBatchDepth),
      write_batch_depth_(kDefaultWriteBatchDepth),
      max_read_batch_bytes_(kDefaultMaxReadBatchBytes),
      max_write_batch_bytes_(kDefaultMaxWriteBatchBytes),
      submit_retry_count_(kDefaultSubmitRetryCount),
      max_status_poll_errors_(kDefaultMaxStatusPollErrors),
      aggregation_delay_(kDefaultAggregationDelayUs),
      status_poll_interval_(kDefaultStatusPollIntervalUs) {}

GdsTransport::~GdsTransport() { uninstall(); }

Status GdsTransport::install(std::string& local_segment_name,
                             std::shared_ptr<ControlService> metadata,
                             std::shared_ptr<Topology> local_topology,
                             std::shared_ptr<Config> conf) {
    if (installed_) {
        return Status::InvalidArgument(
            "GDS transport has been installed" LOC_MARK);
    }

    metadata_ = metadata;
    local_segment_name_ = local_segment_name;
    local_topology_ = local_topology;
    conf_ = conf;
    auto driver_result = cuFileDriverOpen();
    if (driver_result.err != CU_FILE_SUCCESS) {
        return Status::InternalError(
            std::string("Failed to open cuFile driver: Code ") +
            std::to_string(driver_result.err) + ", CUDA error " +
            std::to_string(driver_result.cu_err) + LOC_MARK);
    }
    const int configured_batch_depth =
        conf_->get("transports/gds/io_batch_depth", 32);
    if (configured_batch_depth <= 0)
        return Status::InvalidArgument(
            "GDS io_batch_depth must be greater than zero" LOC_MARK);
    io_batch_depth_ = std::min(static_cast<size_t>(configured_batch_depth),
                               kMaxCuFileBatchDepth);
    if (static_cast<size_t>(configured_batch_depth) > io_batch_depth_) {
        LOG(WARNING) << "GDS io_batch_depth=" << configured_batch_depth
                     << " exceeds the safe cuFile batch depth; using "
                     << io_batch_depth_;
    }
    const int requested_inflight_batches = conf_->get(
        "transports/gds/max_inflight_batches",
        static_cast<int>(kDefaultInflightCuFileBatches));
    if (requested_inflight_batches <= 0)
        return Status::InvalidArgument(
            "GDS max_inflight_batches must be greater than zero" LOC_MARK);
    const bool allow_concurrent_batches = conf_->get(
        "transports/gds/allow_concurrent_batches", false);
    const int configured_inflight_batches =
        allow_concurrent_batches ? requested_inflight_batches : 1;
    if (requested_inflight_batches > 1 && !allow_concurrent_batches) {
        LOG(WARNING)
            << "GDS max_inflight_batches=" << requested_inflight_batches
            << " requested, but concurrent cuFile batch handles can abort "
               "some driver versions; using 1. Set "
               "transports/gds/allow_concurrent_batches=true only after "
               "validating the installed cuFile stack.";
    }
    max_inflight_batches_ = std::min(
        static_cast<size_t>(configured_inflight_batches),
        kMaxInflightCuFileBatches);
    if (static_cast<size_t>(configured_inflight_batches) >
        max_inflight_batches_) {
        LOG(WARNING) << "GDS max_inflight_batches="
                     << configured_inflight_batches
                     << " exceeds the safety limit; using "
                     << max_inflight_batches_;
    }
    const int configured_read_batch_depth =
        conf_->get("transports/gds/read_batch_depth",
                   static_cast<int>(kDefaultReadBatchDepth));
    const int configured_write_batch_depth =
        conf_->get("transports/gds/write_batch_depth",
                   static_cast<int>(kDefaultWriteBatchDepth));
    max_read_batch_bytes_ = conf_->get(
        "transports/gds/max_read_batch_bytes", kDefaultMaxReadBatchBytes);
    max_write_batch_bytes_ = conf_->get(
        "transports/gds/max_write_batch_bytes", kDefaultMaxWriteBatchBytes);
    if (configured_read_batch_depth <= 0 ||
        configured_write_batch_depth <= 0 || max_read_batch_bytes_ == 0 ||
        max_write_batch_bytes_ == 0) {
        return Status::InvalidArgument(
            "GDS read/write batch limits must be greater than zero"
            LOC_MARK);
    }
    read_batch_depth_ =
        std::min(static_cast<size_t>(configured_read_batch_depth),
                 io_batch_depth_);
    write_batch_depth_ =
        std::min(static_cast<size_t>(configured_write_batch_depth),
                 io_batch_depth_);
    const int configured_submit_retries =
        conf_->get("transports/gds/submit_retry_count",
                   static_cast<int>(kDefaultSubmitRetryCount));
    const int configured_max_poll_errors =
        conf_->get("transports/gds/max_status_poll_errors",
                   static_cast<int>(kDefaultMaxStatusPollErrors));
    const int configured_aggregation_delay_us =
        conf_->get("transports/gds/aggregation_delay_us",
                   static_cast<int>(kDefaultAggregationDelayUs));
    const int configured_status_poll_interval_us =
        conf_->get("transports/gds/status_poll_interval_us",
                   static_cast<int>(kDefaultStatusPollIntervalUs));
    if (configured_submit_retries < 0 || configured_max_poll_errors <= 0 ||
        configured_aggregation_delay_us < 0 ||
        configured_status_poll_interval_us < 0) {
        return Status::InvalidArgument(
            "Invalid GDS retry or scheduling configuration" LOC_MARK);
    }
    submit_retry_count_ = static_cast<size_t>(configured_submit_retries);
    max_status_poll_errors_ =
        static_cast<size_t>(configured_max_poll_errors);
    aggregation_delay_ =
        std::chrono::microseconds(configured_aggregation_delay_us);
    status_poll_interval_ =
        std::chrono::microseconds(configured_status_poll_interval_us);

    CUfileDrvProps_t properties{};
    auto properties_result = cuFileDriverGetProperties(&properties);
    if (properties_result.err == CU_FILE_SUCCESS &&
        properties.nvfs.max_direct_io_size != 0 &&
        properties.nvfs.max_direct_io_size <=
            std::numeric_limits<size_t>::max() / 1024) {
        max_io_size_ = properties.nvfs.max_direct_io_size * 1024;
    } else {
        LOG(WARNING) << "Unable to query cuFile max_direct_io_size; using "
                     << max_io_size_ << " bytes";
    }
    LOG(INFO) << "GDS transport limits: configured_batch_depth="
              << configured_batch_depth
              << ", effective_batch_depth=" << io_batch_depth_
              << ", registered_max_io_size=" << max_io_size_
              << ", unregistered_max_io_size="
              << std::min(max_io_size_, kSafeUnregisteredBatchIoSize)
              << ", requested_inflight_batches="
              << requested_inflight_batches
              << ", max_inflight_batches=" << max_inflight_batches_
              << ", allow_concurrent_batches=" << allow_concurrent_batches
              << ", read_batch_depth=" << read_batch_depth_
              << ", write_batch_depth=" << write_batch_depth_
              << ", max_read_batch_bytes=" << max_read_batch_bytes_
              << ", max_write_batch_bytes=" << max_write_batch_bytes_
              << ", aggregation_delay_us=" << aggregation_delay_.count()
              << ", status_poll_interval_us="
              << status_poll_interval_.count()
              << ", submit_retry_count=" << submit_retry_count_
              << ", max_status_poll_errors=" << max_status_poll_errors_;
    installed_ = true;
    caps.dram_to_file = true;
    caps.gpu_to_file = true;
    return Status::OK();
}

Status GdsTransport::uninstall() {
    if (installed_) {
        {
            std::lock_guard<std::mutex> scheduler_guard(scheduler_lock_);
            pending_ios_.clear();
            for (auto& io_batch : inflight_io_batches_) {
                if (!io_batch->batch_handle) continue;
                cuFileBatchIODestroy(io_batch->batch_handle->handle);
                delete io_batch->batch_handle;
                io_batch->batch_handle = nullptr;
            }
            inflight_io_batches_.clear();
            dispatch_window_blocked_ = false;
            last_status_poll_ = {};
        }
        // Clean up all allocated sub-batches (if user forgot to free them)
        {
            std::lock_guard<std::mutex> lock(allocated_batches_lock_);
            for (auto* gds_batch : allocated_batches_) {
                // Deallocate the sub-batch
                Slab<GdsSubBatch>::Get().deallocate(gds_batch);
            }
            allocated_batches_.clear();
        }

        // Clean up all handles in the pool
        std::lock_guard<std::mutex> lock(handle_pool_lock_);
        for (auto* batch_handle : handle_pool_) {
            cuFileBatchIODestroy(batch_handle->handle);
            delete batch_handle;
        }
        handle_pool_.clear();

        metadata_.reset();
        installed_ = false;
    }
    return Status::OK();
}

Status GdsTransport::allocateSubBatch(SubBatchRef& batch, size_t max_size) {
    auto gds_batch = Slab<GdsSubBatch>::Get().allocate();
    if (!gds_batch)
        return Status::InternalError("Unable to allocate GDS sub-batch");
    gds_batch->max_size = max_size;
    gds_batch->io_params.clear();
    gds_batch->io_params.reserve(io_batch_depth_);
    gds_batch->io_param_ranges.clear();
    gds_batch->io_statuses.clear();
    gds_batch->io_statuses.reserve(io_batch_depth_);
    gds_batch->io_transferred_bytes.clear();
    gds_batch->io_transferred_bytes.reserve(io_batch_depth_);

    // Track this batch for cleanup on uninstall
    {
        std::lock_guard<std::mutex> lock(allocated_batches_lock_);
        allocated_batches_.push_back(gds_batch);
    }

    batch = gds_batch;
    return Status::OK();
}

Status GdsTransport::freeSubBatch(SubBatchRef& batch) {
    auto gds_batch = dynamic_cast<GdsSubBatch*>(batch);
    if (!gds_batch)
        return Status::InvalidArgument("Invalid GDS sub-batch" LOC_MARK);

    {
        std::lock_guard<std::mutex> scheduler_guard(scheduler_lock_);
        if (subBatchHasWorkLocked(gds_batch)) {
            return Status::InvalidEntry(
                "Cannot free GDS sub-batch with pending physical I/O"
                LOC_MARK);
        }
    }

    // Remove from tracking list.
    {
        std::lock_guard<std::mutex> lock(allocated_batches_lock_);
        auto it = std::find(allocated_batches_.begin(),
                            allocated_batches_.end(), gds_batch);
        if (it != allocated_batches_.end()) {
            allocated_batches_.erase(it);
        }
    }

    // Deallocate the GdsSubBatch (each allocation gets a fresh one)
    Slab<GdsSubBatch>::Get().deallocate(gds_batch);
    batch = nullptr;
    return Status::OK();
}

std::string GdsTransport::getGdsFilePath(SegmentID target_id) {
    std::string ret;
    auto status = metadata_->segmentManager().withCachedSegment(
        target_id, [&](SegmentDesc* segment) {
            if (segment->type == SegmentType::File) {
                auto& detail = std::get<FileSegmentDesc>(segment->detail);
                if (detail.buffers.empty())
                    return Status::NeedsRefreshCache(
                        "No buffers found" LOC_MARK);
                ret = detail.buffers[0].path;
                return Status::OK();
            }
            if (segment->type == SegmentType::Block) {
                auto& detail = std::get<BlockSegmentDesc>(segment->detail);
                if (detail.path.empty())
                    return Status::NeedsRefreshCache(
                        "Empty block path" LOC_MARK);
                ret = detail.path;
                return Status::OK();
            }
            return Status::NeedsRefreshCache(
                "Segment type is not File or Block" LOC_MARK);
        });
    if (!status.ok()) return "";
    return ret;
}

GdsFileContext* GdsTransport::findFileContext(SegmentID target_id) {
    thread_local FileContextMap tl_file_context_map;
    if (tl_file_context_map.count(target_id))
        return tl_file_context_map[target_id].get();

    RWSpinlock::WriteGuard guard(file_context_lock_);
    if (!file_context_map_.count(target_id)) {
        std::string path = getGdsFilePath(target_id);
        if (path.empty()) return nullptr;
        file_context_map_[target_id] = std::make_shared<GdsFileContext>(path);
    }

    tl_file_context_map = file_context_map_;
    return tl_file_context_map[target_id].get();
}

bool GdsTransport::findRegisteredBuffer(const void* addr, size_t length,
                                        void*& registered_base,
                                        size_t& registered_offset,
                                        int& device_id) {
    if (!addr || length == 0) return false;

    const auto begin = reinterpret_cast<std::uintptr_t>(addr);
    if (length > std::numeric_limits<std::uintptr_t>::max() - begin)
        return false;
    const auto end = begin + length;

    std::lock_guard<std::mutex> lock(registered_buffers_lock_);
    auto it = registered_buffers_.upper_bound(begin);
    if (it == registered_buffers_.begin()) return false;
    --it;
    if (it->second.length >
        std::numeric_limits<std::uintptr_t>::max() - it->first)
        return false;
    if (begin < it->first || end > it->first + it->second.length) return false;
    registered_base = reinterpret_cast<void*>(it->first);
    registered_offset = begin - it->first;
    device_id = it->second.device_id;
    return true;
}

Status GdsTransport::resolveIoBuffer(const Request& request, void*& io_base,
                                     size_t& io_offset, int& device_id) {
    if (findRegisteredBuffer(request.source, request.length, io_base,
                             io_offset, device_id)) {
        return Status::OK();
    }

    if (Platform::getLoader().getMemoryType(request.source) == MTYPE_CPU) {
        io_base = request.source;
        io_offset = 0;
        device_id = -1;
        return Status::OK();
    }

    return Status::AddressNotRegistered(
        "GDS CUDA request is outside a cuFile-registered buffer" LOC_MARK);
}

Status GdsTransport::acquireBatchHandle(int device_id, BatchHandle*& handle) {
    handle = nullptr;
    {
        std::lock_guard<std::mutex> lock(handle_pool_lock_);
        auto it = std::find_if(
            handle_pool_.begin(), handle_pool_.end(),
            [this, device_id](const BatchHandle* handle) {
                return handle->max_nr == static_cast<int>(io_batch_depth_) &&
                       handle->device_id == device_id;
            });
        if (it != handle_pool_.end()) {
            handle = *it;
            handle_pool_.erase(it);
            return Status::OK();
        }
    }

    handle = new BatchHandle{
        .handle = nullptr,
        .max_nr = static_cast<int>(io_batch_depth_),
        .device_id = device_id,
    };
    auto setup_result =
        cuFileBatchIOSetUp(&handle->handle, handle->max_nr);

    if (setup_result.err != CU_FILE_SUCCESS) {
        LOG(ERROR) << "Failed to setup GDS batch IO: code="
                   << setup_result.err << ", cuda_error="
                   << setup_result.cu_err << ", device_id=" << device_id
                   << ", batch_depth=" << io_batch_depth_;
        delete handle;
        handle = nullptr;
        return Status::InternalError(
            std::string("Failed to setup GDS batch IO: Code ") +
            std::to_string(setup_result.err) + ", CUDA error " +
            std::to_string(setup_result.cu_err) + LOC_MARK);
    }

    return Status::OK();
}

void GdsTransport::releaseBatchHandle(BatchHandle* handle) {
    if (!handle) return;
    std::lock_guard<std::mutex> lock(handle_pool_lock_);
    handle_pool_.push_back(handle);
}

void GdsTransport::failPhysicalBatchLocked(GdsIoBatch& io_batch) {
    std::unordered_set<GdsSubBatch*> owners;
    for (const auto& ref : io_batch.refs) {
        if (!ref.owner || ref.param_index >= ref.owner->io_statuses.size())
            continue;
        ref.owner->io_statuses[ref.param_index] = FAILED;
        owners.insert(ref.owner);
    }
    for (auto* owner : owners) owner->notifyProgress();
}

bool GdsTransport::subBatchHasWorkLocked(const GdsSubBatch* batch) const {
    if (!batch) return false;
    if (std::any_of(pending_ios_.begin(), pending_ios_.end(),
                    [batch](const PendingIo& pending) {
                        return pending.owner == batch;
                    })) {
        return true;
    }
    return std::any_of(
        inflight_io_batches_.begin(), inflight_io_batches_.end(),
        [batch](const std::unique_ptr<GdsIoBatch>& io_batch) {
            return std::any_of(
                io_batch->refs.begin(), io_batch->refs.end(),
                [batch](const GdsIoBatch::IoRef& ref) {
                    return ref.owner == batch;
                });
        });
}

Status GdsTransport::dispatchPendingIoLocked() {
    while (!pending_ios_.empty() &&
           inflight_io_batches_.size() < max_inflight_batches_) {
        const int device_id = pending_ios_.front().device_id;
        const auto opcode = pending_ios_.front().params.opcode;
        const size_t opcode_batch_depth =
            opcode == CUFILE_WRITE ? write_batch_depth_ : read_batch_depth_;
        const size_t max_batch_bytes =
            opcode == CUFILE_WRITE ? max_write_batch_bytes_
                                   : max_read_batch_bytes_;
        const size_t split_depth = pending_ios_.front().max_group_entries;
        const size_t max_entries =
            split_depth == 0 ? opcode_batch_depth
                             : std::min(split_depth, opcode_batch_depth);

        size_t io_count = 0;
        size_t io_bytes = 0;
        bool byte_capacity_filled = false;
        for (const auto& pending : pending_ios_) {
            if (pending.device_id != device_id ||
                pending.params.opcode != opcode) {
                continue;
            }
            const size_t io_size = pending.params.u.batch.size;
            if (io_count > 0 &&
                (io_bytes >= max_batch_bytes ||
                 io_size > max_batch_bytes - io_bytes)) {
                byte_capacity_filled = true;
                break;
            }
            ++io_count;
            io_bytes += io_size;
            if (io_count == max_entries) break;
        }
        // A single IO larger than the configured byte window must still make
        // forward progress; the window constrains aggregation, not validity.
        if (io_count == 0) {
            io_count = 1;
            io_bytes = pending_ios_.front().params.u.batch.size;
            byte_capacity_filled = true;
        }
        const bool capacity_filled =
            io_count == max_entries || byte_capacity_filled;
        const auto now = std::chrono::steady_clock::now();
        const bool aggregation_due =
            aggregation_delay_.count() == 0 ||
            now - pending_ios_.front().enqueued_at >= aggregation_delay_;
        if (!capacity_filled && !aggregation_due) break;

        auto io_batch = std::make_unique<GdsIoBatch>();
        io_batch->batch_handle = nullptr;
        io_batch->device_id = device_id;
        io_batch->capacity_filled = capacity_filled;
        io_batch->params.reserve(io_count);
        io_batch->refs.reserve(io_count);
        io_batch->events.resize(io_count);

        for (auto it = pending_ios_.begin();
             it != pending_ios_.end() && io_batch->params.size() < io_count;) {
            if (it->device_id != device_id ||
                it->params.opcode != opcode) {
                ++it;
                continue;
            }
            io_batch->params.push_back(it->params);
            io_batch->refs.push_back({it->owner, it->param_index});
            it = pending_ios_.erase(it);
        }
        for (size_t index = 0; index < io_batch->params.size(); ++index) {
            // Cookies are local to this physical batch. This avoids exposing
            // pointers into a logical SubBatch whose vectors may grow later.
            io_batch->params[index].cookie = reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(index + 1));
        }

        CudaDeviceGuard device_guard;
        auto status = device_guard.activate(device_id);
        if (!status.ok()) {
            LOG(ERROR) << "Failed to activate CUDA device for GDS batch: "
                       << status.ToString();
            failPhysicalBatchLocked(*io_batch);
            continue;
        }

        CUfileError_t last_submit_result{};
        bool submitted = false;
        // Large internal-error batches should be split, not submitted again
        // unchanged. Retrying an identical 100+ MiB request only repeats the
        // expensive driver failure. Keep configured retries for the singleton
        // leaf where no further split is possible.
        const size_t submit_attempts =
            io_batch->params.size() > 1 ? 1 : submit_retry_count_ + 1;
        for (size_t attempt = 0; attempt < submit_attempts; ++attempt) {
            status = acquireBatchHandle(device_id, io_batch->batch_handle);
            if (!status.ok()) {
                LOG(ERROR) << "Failed to acquire GDS batch handle: "
                           << status.ToString();
                if (attempt + 1 < submit_attempts) {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(kSubmitRetryBackoffUs));
                }
                continue;
            }

            const auto submit_start = std::chrono::steady_clock::now();
            last_submit_result = cuFileBatchIOSubmit(
                io_batch->batch_handle->handle,
                static_cast<unsigned>(io_batch->params.size()),
                io_batch->params.data(), 0);
            if (last_submit_result.err == CU_FILE_SUCCESS) {
                const double submit_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - submit_start)
                                                  .count();
                size_t physical_bytes = 0;
                for (const auto& param : io_batch->params)
                    physical_bytes += param.u.batch.size;
                TentMetrics::instance().recordGdsPhysicalBatch(
                    io_batch->params.size(), physical_bytes, submit_seconds,
                    !io_batch->capacity_filled);
                LOG_EVERY_N(INFO, 64)
                    << "GDS global physical batch dispatched: physical_ios="
                    << io_batch->params.size()
                    << ", physical_bytes=" << physical_bytes
                    << ", opcode="
                    << (opcode == CUFILE_WRITE ? "WRITE" : "READ")
                    << ", handle_capacity=" << io_batch_depth_
                    << ", queued_ios_after_dispatch=" << pending_ios_.size()
                    << ", inflight_batches_after_dispatch="
                    << (inflight_io_batches_.size() + 1);
                submitted = true;
                break;
            }

            TENT_RECORD_GDS_BATCH_SUBMIT_FAILED();
            cuFileBatchIODestroy(io_batch->batch_handle->handle);
            delete io_batch->batch_handle;
            io_batch->batch_handle = nullptr;
            LOG(WARNING) << "GDS physical batch submission attempt "
                         << (attempt + 1) << "/"
                         << submit_attempts
                         << " failed: io_count=" << io_batch->params.size()
                         << ", io_bytes=" << io_bytes
                         << ", opcode="
                         << (opcode == CUFILE_WRITE ? "WRITE" : "READ")
                         << ", handle_capacity=" << io_batch_depth_
                         << ", cuFile_error=" << last_submit_result.err
                         << ", cuda_error=" << last_submit_result.cu_err;
            if (attempt + 1 < submit_attempts) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(kSubmitRetryBackoffUs));
            }
        }

        if (!submitted) {
            if (io_batch->params.size() > 1) {
                const size_t split_limit =
                    (io_batch->params.size() + 1) / 2;
                const auto retry_time =
                    std::chrono::steady_clock::now() - aggregation_delay_;
                for (size_t index = io_batch->params.size(); index > 0;
                     --index) {
                    const size_t source_index = index - 1;
                    const auto& ref = io_batch->refs[source_index];
                    pending_ios_.push_front(PendingIo{
                        ref.owner, ref.param_index, io_batch->device_id,
                        io_batch->params[source_index], retry_time,
                        split_limit});
                }
                LOG(WARNING)
                    << "GDS physical batch rejected; requeued as smaller "
                       "batches: original_ios="
                    << io_batch->params.size()
                    << ", next_max_ios=" << split_limit
                    << ", original_bytes=" << io_bytes
                    << ", opcode="
                    << (opcode == CUFILE_WRITE ? "WRITE" : "READ")
                    << ", cuFile_error=" << last_submit_result.err;
                continue;
            }
            const auto& first = io_batch->params.front();
            // Some cuFile versions reject batch WRITE submissions while the
            // synchronous GDS API for the same registered file/buffer tuple
            // remains valid. At the binary-split leaf, use that API before
            // surfacing a logical task failure. This stays on cuFile/GDS and
            // does not route through a POSIX transport.
            errno = 0;
            const ssize_t direct_result =
                opcode == CUFILE_WRITE
                    ? cuFileWrite(first.fh, first.u.batch.devPtr_base,
                                  first.u.batch.size,
                                  first.u.batch.file_offset,
                                  first.u.batch.devPtr_offset)
                    : cuFileRead(first.fh, first.u.batch.devPtr_base,
                                 first.u.batch.size,
                                 first.u.batch.file_offset,
                                 first.u.batch.devPtr_offset);
            if (direct_result ==
                static_cast<ssize_t>(first.u.batch.size)) {
                const auto& ref = io_batch->refs.front();
                if (ref.owner &&
                    ref.param_index < ref.owner->io_statuses.size()) {
                    ref.owner->io_transferred_bytes[ref.param_index] =
                        first.u.batch.size;
                    ref.owner->io_statuses[ref.param_index] = COMPLETED;
                    ref.owner->notifyProgress();
                    LOG_EVERY_N(WARNING, 64)
                        << "GDS batch API rejected a singleton; synchronous "
                           "cuFile fallback completed: opcode="
                        << (opcode == CUFILE_WRITE ? "WRITE" : "READ")
                        << ", bytes=" << first.u.batch.size
                        << ", batch_error=" << last_submit_result.err;
                    continue;
                }
                LOG(ERROR) << "Synchronous GDS fallback completed but the "
                              "logical owner is invalid";
            }
            LOG(ERROR) << "Failed to submit physical GDS batch after retries: "
                       << "io_count=" << io_batch->params.size()
                       << ", cuFile_error=" << last_submit_result.err
                       << ", direct_result=" << direct_result
                       << ", errno=" << errno
                       << ", first_dev_ptr=" << first.u.batch.devPtr_base
                       << ", first_dev_offset="
                       << first.u.batch.devPtr_offset
                       << ", first_file_offset="
                       << first.u.batch.file_offset
                       << ", first_size=" << first.u.batch.size;
            failPhysicalBatchLocked(*io_batch);
            continue;
        }
        inflight_io_batches_.push_back(std::move(io_batch));
    }

    if (!pending_ios_.empty() &&
        inflight_io_batches_.size() >= max_inflight_batches_) {
        const size_t queued_batches =
            (pending_ios_.size() + io_batch_depth_ - 1) / io_batch_depth_;
        if (!dispatch_window_blocked_) {
            TentMetrics::instance().recordGdsDispatchWindowFull(
                queued_batches, inflight_io_batches_.size());
            dispatch_window_blocked_ = true;
            LOG_EVERY_N(INFO, 256)
                << "GDS dispatch window full: queued_batches="
                << queued_batches
                << ", queued_ios=" << pending_ios_.size()
                << ", inflight_batches=" << inflight_io_batches_.size()
                << ", max_inflight_batches=" << max_inflight_batches_;
        }
    } else {
        dispatch_window_blocked_ = false;
    }
    return Status::OK();
}

Status GdsTransport::pollInflightIoLocked() {
    const auto now = std::chrono::steady_clock::now();
    if (!inflight_io_batches_.empty() &&
        last_status_poll_ != std::chrono::steady_clock::time_point{} &&
        now - last_status_poll_ < status_poll_interval_) {
        return dispatchPendingIoLocked();
    }
    last_status_poll_ = now;

    std::unordered_set<GdsSubBatch*> owners_with_progress;
    for (auto it = inflight_io_batches_.begin();
         it != inflight_io_batches_.end();) {
        auto& io_batch = **it;
        CudaDeviceGuard device_guard;
        auto status = device_guard.activate(io_batch.device_id);
        if (!status.ok()) {
            LOG(ERROR) << "Failed to activate CUDA device while polling GDS: "
                       << status.ToString();
            failPhysicalBatchLocked(io_batch);
            releaseBatchHandle(io_batch.batch_handle);
            it = inflight_io_batches_.erase(it);
            continue;
        }

        unsigned num_events = static_cast<unsigned>(io_batch.events.size());
        auto result = cuFileBatchIOGetStatus(
            io_batch.batch_handle->handle, 0, &num_events,
            io_batch.events.data(), nullptr);
        if (result.err != CU_FILE_SUCCESS) {
            ++io_batch.consecutive_poll_errors;
            LOG_EVERY_N(WARNING, 64)
                << "Failed to query GDS batch status: code=" << result.err
                << ", consecutive_errors="
                << io_batch.consecutive_poll_errors << "/"
                << max_status_poll_errors_;
            if (io_batch.consecutive_poll_errors >= max_status_poll_errors_) {
                failPhysicalBatchLocked(io_batch);
                cuFileBatchIODestroy(io_batch.batch_handle->handle);
                delete io_batch.batch_handle;
                io_batch.batch_handle = nullptr;
                it = inflight_io_batches_.erase(it);
            } else {
                ++it;
            }
            auto restore_status = device_guard.restore();
            if (!restore_status.ok()) LOG(ERROR) << restore_status.ToString();
            continue;
        }
        io_batch.consecutive_poll_errors = 0;

        bool invalid_completion = false;
        for (unsigned event_index = 0; event_index < num_events;
             ++event_index) {
            const auto& event = io_batch.events[event_index];
            const std::uintptr_t cookie =
                reinterpret_cast<std::uintptr_t>(event.cookie);
            if (cookie == 0 || cookie > io_batch.refs.size()) {
                LOG(ERROR) << "GDS completion returned an invalid cookie: "
                           << cookie << ", batch_size="
                           << io_batch.refs.size();
                invalid_completion = true;
                break;
            }
            const size_t physical_index = cookie - 1;
            const auto& ref = io_batch.refs[physical_index];
            if (!ref.owner ||
                ref.param_index >= ref.owner->io_statuses.size()) {
                LOG(ERROR) << "GDS completion references an invalid owner";
                invalid_completion = true;
                break;
            }

            auto event_status = parseTransferStatus(event.status);
            if (event_status == COMPLETED) {
                const size_t expected =
                    io_batch.params[physical_index].u.batch.size;
                const size_t transferred = static_cast<size_t>(event.ret);
                if (transferred != expected) {
                    LOG(ERROR) << "Short GDS IO completion: expected="
                               << expected << ", actual=" << transferred;
                    event_status = FAILED;
                } else {
                    ref.owner->io_transferred_bytes[ref.param_index] =
                        transferred;
                }
            } else if (event_status != PENDING) {
                LOG(ERROR) << "GDS IO completion failed: cuFile_status="
                           << static_cast<int>(event.status)
                           << ", return_value=" << event.ret
                           << ", file_offset="
                           << io_batch.params[physical_index]
                                  .u.batch.file_offset
                           << ", size="
                           << io_batch.params[physical_index].u.batch.size;
            }
            ref.owner->io_statuses[ref.param_index] = event_status;
            owners_with_progress.insert(ref.owner);
        }

        if (invalid_completion) failPhysicalBatchLocked(io_batch);

        const bool terminal = std::all_of(
            io_batch.refs.begin(), io_batch.refs.end(),
            [](const GdsIoBatch::IoRef& ref) {
                return ref.owner &&
                       ref.param_index < ref.owner->io_statuses.size() &&
                       ref.owner->io_statuses[ref.param_index] != PENDING;
            });
        auto restore_status = device_guard.restore();
        if (!restore_status.ok()) LOG(ERROR) << restore_status.ToString();
        if (!terminal) {
            ++it;
            continue;
        }

        releaseBatchHandle(io_batch.batch_handle);
        io_batch.batch_handle = nullptr;
        it = inflight_io_batches_.erase(it);
    }

    for (auto* owner : owners_with_progress) owner->notifyProgress();
    return dispatchPendingIoLocked();
}

Status GdsTransport::validateRequest(const Request& request) {
    if (request.length == 0)
        return Status::InvalidArgument(
            "GDS request length must be greater than zero" LOC_MARK);
    void* registered_base = nullptr;
    size_t registered_offset = 0;
    int device_id = -1;
    CHECK_STATUS(resolveIoBuffer(request, registered_base, registered_offset,
                                 device_id));
    (void)registered_base;
    (void)device_id;
    if (registered_offset >
            static_cast<size_t>(std::numeric_limits<off_t>::max()) ||
        request.length > static_cast<size_t>(
                             std::numeric_limits<off_t>::max() -
                             static_cast<off_t>(registered_offset)))
        return Status::InvalidArgument(
            "GDS GPU buffer offset exceeds off_t range" LOC_MARK);
    const uint64_t max_file_offset =
        static_cast<uint64_t>(std::numeric_limits<off_t>::max());
    if (request.target_offset > max_file_offset)
        return Status::InvalidArgument(
            "GDS target offset exceeds off_t range" LOC_MARK);
    if (request.length > max_file_offset - request.target_offset)
        return Status::InvalidArgument(
            "GDS target range exceeds off_t range" LOC_MARK);

    const uint64_t request_end = request.target_offset + request.length;
    return metadata_->segmentManager().withCachedSegment(
        request.target_id, [&](SegmentDesc* segment) {
            if (segment->type == SegmentType::File) {
                const auto& detail =
                    std::get<FileSegmentDesc>(segment->detail);
                if (detail.buffers.empty())
                    return Status::InvalidArgument(
                        "GDS file segment has no buffers" LOC_MARK);

                const auto& buffer = detail.buffers.front();
                if (buffer.length >
                    std::numeric_limits<uint64_t>::max() - buffer.offset)
                    return Status::InvalidArgument(
                        "GDS file segment range overflows" LOC_MARK);
                const uint64_t buffer_end = buffer.offset + buffer.length;
                if (request.target_offset < buffer.offset ||
                    request_end > buffer_end)
                    return Status::InvalidArgument(
                        "GDS request is outside the file segment" LOC_MARK);
                return Status::OK();
            }

            if (segment->type == SegmentType::Block) {
                const auto& detail =
                    std::get<BlockSegmentDesc>(segment->detail);
                if (detail.length >
                    std::numeric_limits<uint64_t>::max() - detail.offset)
                    return Status::InvalidArgument(
                        "GDS block segment range overflows" LOC_MARK);
                const uint64_t segment_end = detail.offset + detail.length;
                if (request.target_offset < detail.offset ||
                    request_end > segment_end)
                    return Status::InvalidArgument(
                        "GDS request is outside the block segment" LOC_MARK);

                const uint64_t alignment = detail.allocation_alignment;
                if (alignment == 0 || request.target_offset % alignment != 0 ||
                    request.length % alignment != 0)
                    return Status::InvalidArgument(
                        "GDS block request is not allocation-aligned" LOC_MARK);
                return Status::OK();
            }

            return Status::InvalidArgument(
                "GDS target is not a file or block segment" LOC_MARK);
        });
}

Status GdsTransport::submitTransferTasks(
    SubBatchRef batch, const std::vector<Request>& request_list) {
    auto gds_batch = dynamic_cast<GdsSubBatch*>(batch);
    if (!gds_batch)
        return Status::InvalidArgument("Invalid GDS sub-batch" LOC_MARK);

    if (request_list.empty())
        return Status::InvalidArgument("Empty GDS request list" LOC_MARK);

    std::lock_guard<std::mutex> scheduler_guard(scheduler_lock_);
    if (gds_batch->io_param_ranges.size() > gds_batch->max_size ||
        request_list.size() >
            gds_batch->max_size - gds_batch->io_param_ranges.size())
        return Status::TooManyRequests(
            "Exceed GDS logical batch capacity" LOC_MARK);
    size_t num_params = 0;
    size_t first_param_index = gds_batch->io_params.size();
    std::vector<GdsFileContext*> contexts;
    std::vector<void*> io_bases;
    std::vector<size_t> io_offsets;
    std::vector<size_t> io_chunk_sizes;
    contexts.reserve(request_list.size());
    io_bases.reserve(request_list.size());
    io_offsets.reserve(request_list.size());
    io_chunk_sizes.reserve(request_list.size());
    int batch_device_id = -1;
    for (const auto& request : request_list) {
        auto status = validateRequest(request);
        if (!status.ok()) {
            LOG(ERROR) << "Rejected GDS request: target_id="
                       << request.target_id << ", opcode="
                       << (request.opcode == Request::READ ? "READ" : "WRITE")
                       << ", offset=" << request.target_offset
                       << ", length=" << request.length << ": "
                       << status.ToString();
            return status;
        }

        void* io_base = nullptr;
        size_t io_offset = 0;
        int request_device_id = -1;
        CHECK_STATUS(resolveIoBuffer(request, io_base, io_offset,
                                     request_device_id));
        if (request_device_id >= 0) {
            if (batch_device_id >= 0 &&
                request_device_id != batch_device_id) {
                return Status::InvalidArgument(
                    "GDS batch contains buffers from multiple CUDA devices"
                    LOC_MARK);
            }
            batch_device_id = request_device_id;
        }

        // Registered CUDA buffers can use the driver's full direct-I/O size.
        // The 960 KiB limit exists only for cuFile's unregistered/bounce-buffer
        // path; applying it to vLLM's registered KV allocation triples the
        // physical I/O count for every 2.25 MiB cache object.
        const size_t io_chunk_size =
            request_device_id >= 0
                ? max_io_size_
                : std::min(max_io_size_, kSafeUnregisteredBatchIoSize);
        const size_t request_params =
            1 + (request.length - 1) / io_chunk_size;
        if (request_params > std::numeric_limits<size_t>::max() - num_params)
            return Status::TooManyRequests(
                "GDS request count overflows" LOC_MARK);
        num_params += request_params;

        GdsFileContext* context = findFileContext(request.target_id);
        if (!context || !context->ready())
            return Status::InvalidArgument("Invalid remote segment" LOC_MARK);
        contexts.push_back(context);
        io_bases.push_back(io_base);
        io_offsets.push_back(io_offset);
        io_chunk_sizes.push_back(io_chunk_size);
    }
    for (size_t request_index = 0; request_index < request_list.size();
         ++request_index) {
        const auto& request = request_list[request_index];
        GdsFileContext* context = contexts[request_index];
        IOParamRange range{gds_batch->io_params.size(), 0};
        const size_t io_chunk_size = io_chunk_sizes[request_index];
        for (size_t offset = 0; offset < request.length;
             offset += io_chunk_size) {
            size_t length = std::min(io_chunk_size, request.length - offset);
            const size_t param_index = gds_batch->io_params.size();
            CUfileIOParams_t params{};
            params.mode = CUFILE_BATCH;
            params.opcode =
                (request.opcode == Request::READ) ? CUFILE_READ : CUFILE_WRITE;
            params.cookie = reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(param_index + 1));
            // Keep the base pointer identical to the pointer registered with
            // cuFileBufRegister. KV blocks are interior slices of one large
            // registered allocation and must be addressed via devPtr_offset.
            params.u.batch.devPtr_base = io_bases[request_index];
            params.u.batch.devPtr_offset = static_cast<off_t>(
                io_offsets[request_index] + offset);
            params.u.batch.file_offset = request.target_offset + offset;
            params.u.batch.size = length;
            params.fh = context->getHandle();
            gds_batch->io_params.push_back(params);
            gds_batch->io_statuses.push_back(PENDING);
            gds_batch->io_transferred_bytes.push_back(0);
            range.count++;
        }
        gds_batch->io_param_ranges.push_back(range);
    }

    const auto count_planned_batches =
        [&](const auto opcode, size_t max_entries,
            size_t max_bytes) -> size_t {
        size_t batch_count = 0;
        size_t batch_entries = 0;
        size_t batch_bytes = 0;
        for (size_t index = first_param_index;
             index < gds_batch->io_params.size(); ++index) {
            const auto& param = gds_batch->io_params[index];
            if (param.opcode != opcode) continue;
            const size_t io_size = param.u.batch.size;
            if (batch_entries > 0 &&
                (batch_entries == max_entries || batch_bytes >= max_bytes ||
                 io_size > max_bytes - batch_bytes)) {
                ++batch_count;
                batch_entries = 0;
                batch_bytes = 0;
            }
            ++batch_entries;
            batch_bytes += io_size;
            if (batch_entries == max_entries) {
                ++batch_count;
                batch_entries = 0;
                batch_bytes = 0;
            }
        }
        if (batch_entries > 0) ++batch_count;
        return batch_count;
    };
    const size_t physical_batch_count =
        count_planned_batches(CUFILE_READ, read_batch_depth_,
                              max_read_batch_bytes_) +
        count_planned_batches(CUFILE_WRITE, write_batch_depth_,
                              max_write_batch_bytes_);
    const size_t small_request_count = static_cast<size_t>(std::count_if(
        request_list.begin(), request_list.end(),
        [](const Request& request) { return request.length <= 64 * 1024; }));
    TentMetrics::instance().recordGdsTransportSubmission(
        request_list.size(), num_params, physical_batch_count,
        small_request_count);
    if (physical_batch_count > 16) {
        LOG(WARNING)
            << "Large GDS transport submission: logical_requests="
            << request_list.size() << ", physical_ios=" << num_params
            << ", physical_batches=" << physical_batch_count
            << ". Enable the TENT runtime queue and keep "
               "runtime_queue/max_dispatch_owners bounded to avoid an "
               "unbounded cuFile batch backlog.";
    }
    LOG_EVERY_N(INFO, 256)
        << "GDS transport submission: logical_requests="
        << request_list.size() << ", physical_ios=" << num_params
        << ", unaggregated_physical_batches=" << physical_batch_count
        << ", small_requests=" << small_request_count
        << ", read_batch_depth=" << read_batch_depth_
        << ", write_batch_depth=" << write_batch_depth_
        << ", max_read_batch_bytes=" << max_read_batch_bytes_
        << ", max_write_batch_bytes=" << max_write_batch_bytes_
        << ", max_inflight_batches=" << max_inflight_batches_;
    const auto enqueued_at = std::chrono::steady_clock::now();
    for (size_t param_index = first_param_index;
         param_index < gds_batch->io_params.size(); ++param_index) {
        pending_ios_.push_back(PendingIo{gds_batch, param_index,
                                         batch_device_id,
                                         gds_batch->io_params[param_index],
                                         enqueued_at});
    }

    return dispatchPendingIoLocked();
}

Status GdsTransport::getTransferStatus(SubBatchRef batch, int task_id,
                                       TransferStatus& status) {
    auto gds_batch = dynamic_cast<GdsSubBatch*>(batch);
    if (!gds_batch)
        return Status::InvalidArgument("Invalid GDS sub-batch" LOC_MARK);

    std::lock_guard<std::mutex> scheduler_guard(scheduler_lock_);
    const size_t num_tasks = gds_batch->io_param_ranges.size();
    if (task_id < 0 || static_cast<size_t>(task_id) >= num_tasks)
        return Status::InvalidArgument("Invalid task ID");
    auto range = gds_batch->io_param_ranges[task_id];

    CHECK_STATUS(pollInflightIoLocked());

    status = TransferStatus{PENDING, 0};
    size_t complete_count = 0;
    size_t terminal_count = 0;
    for (size_t index = range.base; index < range.base + range.count; ++index) {
        auto s = gds_batch->io_statuses[index];
        if (s == COMPLETED) {
            complete_count++;
            terminal_count++;
        } else if (s != PENDING) {
            terminal_count++;
            status.s = s;
        }
        status.transferred_bytes +=
            gds_batch->io_transferred_bytes[index];
    }
    if (terminal_count != range.count)
        status.s = PENDING;
    else if (complete_count == range.count)
        status.s = COMPLETED;
    return Status::OK();
}

Status GdsTransport::addMemoryBuffer(BufferDesc& desc,
                                     const MemoryOptions& options) {
    (void)options;
    LocationParser location(desc.location);
    if (location.type() != "cuda") return Status::OK();
    if (location.index() < 0)
        return Status::InvalidArgument(
            "GDS CUDA buffer has no device index" LOC_MARK);

    CudaDeviceGuard device_guard;
    CHECK_STATUS(device_guard.activate(location.index()));
    auto result = cuFileBufRegister((void*)desc.addr, desc.length, 0);
    if (result.err != CU_FILE_SUCCESS) {
        TENT_RECORD_GDS_BUFFER_REGISTRATION_FAILED();
        LOG(ERROR) << "Failed to register GDS buffer: addr="
                   << reinterpret_cast<void*>(desc.addr)
                   << ", length=" << desc.length
                   << ", location=" << desc.location
                   << ", cuFile_error=" << result.err;
        return Status::InternalError(
            std::string("Failed to register GDS buffer: Code ") +
            std::to_string(result.err) + LOC_MARK);
    }
    auto restore_status = device_guard.restore();
    if (!restore_status.ok()) {
        cuFileBufDeregister((void*)desc.addr);
        return restore_status;
    }
    {
        std::lock_guard<std::mutex> lock(registered_buffers_lock_);
        registered_buffers_[static_cast<std::uintptr_t>(desc.addr)] =
            RegisteredBuffer{desc.length, location.index()};
    }
    desc.transports.push_back(GDS);
    return Status::OK();
}

Status GdsTransport::removeMemoryBuffer(BufferDesc& desc) {
    LocationParser location(desc.location);
    if (location.type() != "cuda") return Status::OK();
    if (location.index() < 0)
        return Status::InvalidArgument(
            "GDS CUDA buffer has no device index" LOC_MARK);

    CudaDeviceGuard device_guard;
    CHECK_STATUS(device_guard.activate(location.index()));
    auto result = cuFileBufDeregister((void*)desc.addr);
    {
        std::lock_guard<std::mutex> lock(registered_buffers_lock_);
        registered_buffers_.erase(static_cast<std::uintptr_t>(desc.addr));
    }
    if (result.err != CU_FILE_SUCCESS) {
        LOG(ERROR) << "Failed to deregister GDS buffer: addr="
                   << reinterpret_cast<void*>(desc.addr)
                   << ", length=" << desc.length
                   << ", location=" << desc.location
                   << ", cuFile_error=" << result.err;
        return Status::InternalError(
            std::string("Failed to deregister GDS buffer: Code ") +
            std::to_string(result.err) + LOC_MARK);
    }
    CHECK_STATUS(device_guard.restore());
    return Status::OK();
}

}  // namespace tent
}  // namespace mooncake
