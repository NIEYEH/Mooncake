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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>

#include <fcntl.h>
#include <unistd.h>

#include "tent/runtime/slab.h"
#include "tent/metrics/tent_metrics.h"

namespace mooncake {
namespace tent {
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
    : installed_(false), io_batch_depth_(0), max_io_size_(1ull << 20) {
    static std::once_flag g_once_flag;
    auto fork_init = []() { cuFileDriverOpen(); };
    std::call_once(g_once_flag, fork_init);
}

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
    const int configured_batch_depth =
        conf_->get("transports/gds/io_batch_depth", 32);
    if (configured_batch_depth <= 0)
        return Status::InvalidArgument(
            "GDS io_batch_depth must be greater than zero" LOC_MARK);
    io_batch_depth_ = static_cast<size_t>(configured_batch_depth);

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
    LOG(INFO) << "GDS transport limits: batch_depth=" << io_batch_depth_
              << ", max_io_size=" << max_io_size_;
    installed_ = true;
    caps.dram_to_file = true;
    caps.gpu_to_file = true;
    return Status::OK();
}

Status GdsTransport::uninstall() {
    if (installed_) {
        // Clean up all allocated sub-batches (if user forgot to free them)
        {
            std::lock_guard<std::mutex> lock(allocated_batches_lock_);
            for (auto* gds_batch : allocated_batches_) {
                // Destroy the batch handle (don't return to pool since we're
                // shutting down)
                cuFileBatchIODestroy(gds_batch->batch_handle->handle);
                delete gds_batch->batch_handle;
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

    // Get or create BatchHandle from pool
    BatchHandle* batch_handle = nullptr;
    {
        std::lock_guard<std::mutex> lock(handle_pool_lock_);
        if (!handle_pool_.empty()) {
            batch_handle = handle_pool_.back();
            handle_pool_.pop_back();
        }
    }

    // If pool is empty or handle size mismatch, create new handle (expensive
    // operation)
    if (!batch_handle ||
        batch_handle->max_nr != static_cast<int>(io_batch_depth_)) {
        // Destroy mismatched handle if exists
        if (batch_handle) {
            cuFileBatchIODestroy(batch_handle->handle);
            delete batch_handle;
        }

        batch_handle = new BatchHandle();
        batch_handle->max_nr = static_cast<int>(io_batch_depth_);
        // cuFileBatchIOSetUp is time-costly, so we reuse handles
        auto result = cuFileBatchIOSetUp(
            &batch_handle->handle, static_cast<int>(io_batch_depth_));
        if (result.err != CU_FILE_SUCCESS) {
            delete batch_handle;
            Slab<GdsSubBatch>::Get().deallocate(gds_batch);
            return Status::InternalError(
                std::string("Failed to setup GDS batch IO: Code ") +
                std::to_string(result.err) + LOC_MARK);
        }
    }

    gds_batch->batch_handle = batch_handle;
    gds_batch->max_size = max_size;
    gds_batch->io_events.resize(io_batch_depth_);
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

    // Remove from tracking list
    {
        std::lock_guard<std::mutex> lock(allocated_batches_lock_);
        auto it = std::find(allocated_batches_.begin(),
                            allocated_batches_.end(), gds_batch);
        if (it != allocated_batches_.end()) {
            allocated_batches_.erase(it);
        }
    }

    // Return the handle to pool for reuse (avoid expensive
    // cuFileBatchIODestroy) Note: Caller should ensure all IOs are completed
    // (via getTransferStatus) before calling freeSubBatch, as cuFile may still
    // access io_params otherwise
    {
        std::lock_guard<std::mutex> lock(handle_pool_lock_);
        handle_pool_.push_back(gds_batch->batch_handle);
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
                                        size_t& registered_offset) {
    if (!addr || length == 0) return false;

    const auto begin = reinterpret_cast<std::uintptr_t>(addr);
    if (length > std::numeric_limits<std::uintptr_t>::max() - begin)
        return false;
    const auto end = begin + length;

    std::lock_guard<std::mutex> lock(registered_buffers_lock_);
    auto it = registered_buffers_.upper_bound(begin);
    if (it == registered_buffers_.begin()) return false;
    --it;
    if (it->second > std::numeric_limits<std::uintptr_t>::max() - it->first)
        return false;
    if (begin < it->first || end > it->first + it->second) return false;
    registered_base = reinterpret_cast<void*>(it->first);
    registered_offset = begin - it->first;
    return true;
}

Status GdsTransport::resolveIoBuffer(const Request& request, void*& io_base,
                                     size_t& io_offset) {
    if (findRegisteredBuffer(request.source, request.length, io_base,
                             io_offset))
        return Status::OK();

    if (Platform::getLoader().getMemoryType(request.source) == MTYPE_CPU) {
        io_base = request.source;
        io_offset = 0;
        return Status::OK();
    }

    return Status::AddressNotRegistered(
        "GDS CUDA request is outside a cuFile-registered buffer" LOC_MARK);
}

Status GdsTransport::validateRequest(const Request& request) {
    if (request.length == 0)
        return Status::InvalidArgument(
            "GDS request length must be greater than zero" LOC_MARK);
    void* registered_base = nullptr;
    size_t registered_offset = 0;
    CHECK_STATUS(resolveIoBuffer(request, registered_base, registered_offset));
    (void)registered_base;
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

    std::lock_guard<std::mutex> status_guard(gds_batch->status_lock);
    if (gds_batch->io_param_ranges.size() > gds_batch->max_size ||
        request_list.size() >
            gds_batch->max_size - gds_batch->io_param_ranges.size())
        return Status::TooManyRequests(
            "Exceed GDS logical batch capacity" LOC_MARK);
    size_t num_params = 0;
    size_t first_param_index = gds_batch->io_params.size();
    std::vector<GdsFileContext*> contexts;
    contexts.reserve(request_list.size());
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

        const size_t request_params =
            1 + (request.length - 1) / max_io_size_;
        if (request_params > std::numeric_limits<size_t>::max() - num_params)
            return Status::TooManyRequests(
                "GDS request count overflows" LOC_MARK);
        num_params += request_params;

        GdsFileContext* context = findFileContext(request.target_id);
        if (!context || !context->ready())
            return Status::InvalidArgument("Invalid remote segment" LOC_MARK);
        contexts.push_back(context);
    }
    if (first_param_index > io_batch_depth_ ||
        num_params > io_batch_depth_ - first_param_index) {
        LOG(ERROR) << "GDS batch requires " << num_params
                   << " IO entries but only "
                   << (first_param_index > io_batch_depth_
                           ? 0
                           : io_batch_depth_ - first_param_index)
                   << " are available; logical_requests="
                   << request_list.size() << ", max_io_size="
                   << max_io_size_;
        return Status::TooManyRequests("Exceed batch capacity" LOC_MARK);
    }

    const size_t first_range_index = gds_batch->io_param_ranges.size();
    for (size_t request_index = 0; request_index < request_list.size();
         ++request_index) {
        const auto& request = request_list[request_index];
        GdsFileContext* context = contexts[request_index];
        void* io_base = nullptr;
        size_t io_offset = 0;
        auto io_buffer_status = resolveIoBuffer(request, io_base, io_offset);
        if (!io_buffer_status.ok()) return io_buffer_status;
        IOParamRange range{gds_batch->io_params.size(), 0};
        for (size_t offset = 0; offset < request.length;
             offset += max_io_size_) {
            size_t length = std::min(max_io_size_, request.length - offset);
            const size_t param_index = gds_batch->io_params.size();
            CUfileIOParams_t params{};
            params.mode = CUFILE_BATCH;
            params.opcode =
                (request.opcode == Request::READ) ? CUFILE_READ : CUFILE_WRITE;
            params.cookie = reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(param_index + 1));
            params.u.batch.devPtr_base = io_base;
            params.u.batch.devPtr_offset =
                static_cast<off_t>(io_offset + offset);
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

    auto result = cuFileBatchIOSubmit(
        gds_batch->batch_handle->handle, static_cast<unsigned>(num_params),
        &gds_batch->io_params[first_param_index], 0);
    if (result.err != CU_FILE_SUCCESS) {
        TENT_RECORD_GDS_BATCH_SUBMIT_FAILED();
        gds_batch->io_params.resize(first_param_index);
        gds_batch->io_statuses.resize(first_param_index);
        gds_batch->io_transferred_bytes.resize(first_param_index);
        gds_batch->io_param_ranges.resize(first_range_index);
        LOG(ERROR) << "Failed to submit GDS batch IO: request_count="
                   << request_list.size() << ", io_count=" << num_params
                   << ", cuFile_error=" << result.err;
        return Status::InternalError(
            std::string("Failed to submit GDS batch IO: Code ") +
            std::to_string(result.err) + LOC_MARK);
    }
    return Status::OK();
}

Status GdsTransport::getTransferStatus(SubBatchRef batch, int task_id,
                                       TransferStatus& status) {
    auto gds_batch = dynamic_cast<GdsSubBatch*>(batch);
    if (!gds_batch)
        return Status::InvalidArgument("Invalid GDS sub-batch" LOC_MARK);

    std::lock_guard<std::mutex> status_guard(gds_batch->status_lock);
    const size_t num_tasks = gds_batch->io_param_ranges.size();
    if (task_id < 0 || static_cast<size_t>(task_id) >= num_tasks)
        return Status::InvalidArgument("Invalid task ID");
    auto range = gds_batch->io_param_ranges[task_id];

    size_t pending_count = 0;
    for (auto io_status : gds_batch->io_statuses) {
        if (io_status == PENDING) ++pending_count;
    }
    if (pending_count != 0) {
        unsigned num_events = static_cast<unsigned>(std::min<size_t>(
            pending_count, gds_batch->io_events.size()));
        auto result = cuFileBatchIOGetStatus(
            gds_batch->batch_handle->handle, 0, &num_events,
            gds_batch->io_events.data(), nullptr);
        if (result.err != CU_FILE_SUCCESS)
            return Status::InternalError(
                std::string("Failed to get GDS batch status: Code ") +
                std::to_string(result.err) + LOC_MARK);

        for (unsigned event_index = 0; event_index < num_events;
             ++event_index) {
            const auto& event = gds_batch->io_events[event_index];
            const std::uintptr_t cookie =
                reinterpret_cast<std::uintptr_t>(event.cookie);
            if (cookie == 0 || cookie > gds_batch->io_statuses.size())
                return Status::InternalError(
                    "GDS completion returned an invalid cookie" LOC_MARK);

            const size_t param_index = cookie - 1;
            auto event_status = parseTransferStatus(event.status);
            if (event_status == COMPLETED) {
                const size_t expected =
                    gds_batch->io_params[param_index].u.batch.size;
                const size_t transferred = static_cast<size_t>(event.ret);
                if (transferred != expected) {
                    LOG(ERROR) << "Short GDS IO completion: io_index="
                               << param_index << ", expected=" << expected
                               << ", actual=" << transferred;
                    event_status = FAILED;
                } else {
                    gds_batch->io_transferred_bytes[param_index] = transferred;
                }
            } else if (event_status != PENDING) {
                LOG(ERROR) << "GDS IO completion failed: io_index="
                           << param_index << ", cuFile_status="
                           << static_cast<int>(event.status)
                           << ", return_value=" << event.ret
                           << ", file_offset="
                           << gds_batch->io_params[param_index]
                                  .u.batch.file_offset
                           << ", size="
                           << gds_batch->io_params[param_index].u.batch.size;
            }
            gds_batch->io_statuses[param_index] = event_status;
        }
        if (num_events != 0) batch->notifyProgress();
    }

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
    {
        std::lock_guard<std::mutex> lock(registered_buffers_lock_);
        registered_buffers_[static_cast<std::uintptr_t>(desc.addr)] =
            desc.length;
    }
    desc.transports.push_back(GDS);
    return Status::OK();
}

Status GdsTransport::removeMemoryBuffer(BufferDesc& desc) {
    LocationParser location(desc.location);
    if (location.type() != "cuda") return Status::OK();
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
    return Status::OK();
}

}  // namespace tent
}  // namespace mooncake
