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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "tent/runtime/slab.h"
#include "tent/metrics/tent_metrics.h"

namespace mooncake {
namespace tent {
namespace {

constexpr size_t kSafeUnregisteredIoSize = 960 * 1024;
constexpr size_t kDefaultReadWorkerThreads = 16;
constexpr size_t kDefaultWriteWorkerThreads = 4;
constexpr size_t kDefaultMaxInflightReads = 16;
constexpr size_t kDefaultMaxInflightWrites = 4;
constexpr size_t kMaxDirectIoWorkerThreads = 64;
constexpr size_t kDefaultSubmitRetryCount = 0;
constexpr size_t kDefaultAdaptiveSampleWindow = 128;
constexpr size_t kDefaultAdaptiveEvaluationInterval = 32;
constexpr size_t kDefaultAdaptiveRecoveryWindows = 3;
constexpr size_t kDefaultMinReadInflight = 4;
constexpr size_t kDefaultMinWriteInflight = 1;
constexpr double kDefaultAdaptiveDegradationRatio = 1.25;
constexpr double kDefaultAdaptiveRecoveryRatio = 1.05;
constexpr double kAdaptiveLoadedRecoveryRatio = 1.00;
constexpr size_t kDefaultWriteP99TargetUs = 10000;
constexpr size_t kDefaultWriteStarvationTimeoutUs = 100000;
constexpr size_t kSubmitRetryBackoffUs = 50;
constexpr size_t kIoSummarySampleCapacity = 4096;
constexpr auto kIoSummaryInterval = std::chrono::seconds(1);
constexpr auto kMinimumDegradedWarningInterval =
    std::chrono::seconds(10);

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

size_t gdsAvailableWorkerSlots(size_t effective_limit,
                               size_t inflight_ios) {
    return effective_limit > inflight_ios
               ? effective_limit - inflight_ios
               : 0;
}

size_t gdsEffectiveWriteLimit(size_t adaptive_write_limit,
                              bool reads_active) {
    if (adaptive_write_limit == 0) return 0;
    return reads_active ? std::min<size_t>(adaptive_write_limit, 1)
                        : adaptive_write_limit;
}

bool gdsReadWindowSaturated(size_t effective_read_limit,
                            size_t inflight_reads,
                            size_t inflight_writes,
                            bool write_is_at_minimum) {
    if (effective_read_limit == 0) return false;
    if (inflight_reads >= effective_read_limit) return true;
    // READ owns the latency-sensitive path. While WRITE can still be
    // reduced, mixed-device pressure is attributed to WRITE first. Once
    // WRITE is at its floor, account for its unavoidable shared occupancy
    // without using an overflow-prone addition.
    return write_is_at_minimum &&
           inflight_writes >= effective_read_limit - inflight_reads;
}

double gdsNearestRankP99(std::vector<double> samples) {
    if (samples.empty()) return 0.0;
    const size_t index = std::min(
        samples.size() - 1,
        static_cast<size_t>(std::ceil(samples.size() * 0.99)) - 1);
    std::nth_element(samples.begin(), samples.begin() + index,
                     samples.end());
    return samples[index];
}

bool gdsAdaptiveEvaluationReady(const GdsAdaptiveState& state,
                                size_t sample_window,
                                size_t evaluation_interval) {
    return sample_window > 0 && evaluation_interval > 0 &&
           state.recent_io_latency_us.size() >= sample_window &&
           state.completions_since_evaluation >= evaluation_interval;
}

GdsAdaptiveAction adjustGdsAdaptiveConcurrency(
    GdsAdaptiveState& state, double p99_us, size_t queued_ios,
    double degradation_ratio, double recovery_ratio,
    size_t recovery_windows) {
    // Only an effective window that was actually full is a device saturation
    // signal. Runtime backlog alone can be caused by READ priority or strict
    // dispatch backpressure and must not ratchet WRITE from 4 down to 1.
    const bool saturated = state.saturation_since_evaluation;
    state.saturation_since_evaluation = false;
    if (state.baseline_p99_us <= 0.0) {
        state.baseline_p99_us =
            std::max(state.target_p99_us, p99_us);
        return GdsAdaptiveAction::NONE;
    }

    const double reference_p99_us =
        std::max(state.target_p99_us, state.baseline_p99_us);
    const bool degraded =
        saturated && p99_us > reference_p99_us * degradation_ratio;
    if (degraded) {
        state.healthy_windows = 0;
        if (state.current_limit > state.minimum_limit) {
            state.degraded_at_minimum_windows = 0;
            const size_t reduction =
                std::max<size_t>(1, state.current_limit / 4);
            state.current_limit = std::max(
                state.minimum_limit, state.current_limit - reduction);
            return GdsAdaptiveAction::REDUCE;
        }
        // Freeze the baseline at the minimum: overload must never normalize
        // itself into a healthy signal and trigger a later increase.
        if (state.degraded_at_minimum_windows <
            std::numeric_limits<size_t>::max()) {
            ++state.degraded_at_minimum_windows;
        }
        return GdsAdaptiveAction::HOLD_AT_MINIMUM;
    }
    state.degraded_at_minimum_windows = 0;

    if (!saturated && queued_ios == 0) {
        const double baseline_weight =
            p99_us < reference_p99_us ? 0.20 : 0.02;
        state.baseline_p99_us = std::max(
            state.target_p99_us,
            reference_p99_us * (1.0 - baseline_weight) +
                p99_us * baseline_weight);
    }

    const double recovery_reference_p99_us =
        std::max(state.target_p99_us, state.baseline_p99_us);
    const bool idle_healthy =
        queued_ios == 0 &&
        p99_us <= recovery_reference_p99_us * recovery_ratio;
    const bool loaded_with_headroom =
        queued_ios > 0 &&
        p99_us <= recovery_reference_p99_us *
                      kAdaptiveLoadedRecoveryRatio;
    if ((idle_healthy || loaded_with_headroom) &&
        state.current_limit < state.configured_limit) {
        ++state.healthy_windows;
        const size_t required_windows =
            loaded_with_headroom &&
                    recovery_windows <=
                        std::numeric_limits<size_t>::max() / 2
                ? recovery_windows * 2
                : recovery_windows;
        if (state.healthy_windows >= required_windows) {
            ++state.current_limit;
            state.healthy_windows = 0;
            return GdsAdaptiveAction::RECOVER;
        }
    } else {
        state.healthy_windows = 0;
    }
    return GdsAdaptiveAction::NONE;
}

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

GdsTransport::GdsTransport()
    : installed_(false),
      max_io_size_(1ull << 20),
      read_worker_threads_(kDefaultReadWorkerThreads),
      write_worker_threads_(kDefaultWriteWorkerThreads),
      max_inflight_reads_(kDefaultMaxInflightReads),
      max_inflight_writes_(kDefaultMaxInflightWrites),
      submit_retry_count_(kDefaultSubmitRetryCount),
      adaptive_concurrency_enabled_(true),
      adaptive_sample_window_(kDefaultAdaptiveSampleWindow),
      adaptive_evaluation_interval_(kDefaultAdaptiveEvaluationInterval),
      adaptive_degradation_ratio_(kDefaultAdaptiveDegradationRatio),
      adaptive_recovery_ratio_(kDefaultAdaptiveRecoveryRatio),
      adaptive_recovery_windows_(kDefaultAdaptiveRecoveryWindows),
      write_starvation_timeout_(kDefaultWriteStarvationTimeoutUs) {}

GdsTransport::~GdsTransport() { uninstall(); }

size_t GdsTransport::runtimeQueueDispatchLimit(Request::OpCode opcode) const {
    std::lock_guard<std::mutex> scheduler_guard(scheduler_lock_);
    if (opcode == Request::READ) return read_adaptive_.current_limit;
    const bool reads_active =
        !pending_reads_.empty() || !inflight_direct_reads_.empty() ||
        runtime_queued_reads_.load(std::memory_order_relaxed) > 0;
    return gdsEffectiveWriteLimit(write_adaptive_.current_limit,
                                  reads_active);
}

void GdsTransport::updateRuntimeQueueDepth(size_t queued_reads,
                                           size_t queued_writes) {
    runtime_queued_reads_.store(queued_reads, std::memory_order_relaxed);
    runtime_queued_writes_.store(queued_writes, std::memory_order_relaxed);
}

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
    // The multi-entry driver API is intentionally and unconditionally absent.
    // On NVMe-oF/raw block deployments it can deterministically return 5030
    // for every io_count > 1; retrying and recursively splitting only adds
    // tail latency. Every entry below uses an independent synchronous call.
    const int configured_read_threads = conf_->get(
        "transports/gds/read_worker_threads",
        static_cast<int>(kDefaultReadWorkerThreads));
    const int configured_write_threads = conf_->get(
        "transports/gds/write_worker_threads",
        static_cast<int>(kDefaultWriteWorkerThreads));
    const int configured_inflight_reads = conf_->get(
        "transports/gds/max_inflight_reads",
        static_cast<int>(kDefaultMaxInflightReads));
    const int configured_inflight_writes = conf_->get(
        "transports/gds/max_inflight_writes",
        static_cast<int>(kDefaultMaxInflightWrites));
    if (configured_read_threads <= 0 || configured_write_threads <= 0 ||
        configured_inflight_reads <= 0 || configured_inflight_writes <= 0) {
        return Status::InvalidArgument(
            "GDS direct IO worker limits must be greater than zero" LOC_MARK);
    }
    read_worker_threads_ = std::min(
        static_cast<size_t>(configured_read_threads),
        kMaxDirectIoWorkerThreads);
    write_worker_threads_ = std::min(
        static_cast<size_t>(configured_write_threads),
        kMaxDirectIoWorkerThreads);
    max_inflight_reads_ = std::min(static_cast<size_t>(configured_inflight_reads),
                                   read_worker_threads_);
    max_inflight_writes_ =
        std::min(static_cast<size_t>(configured_inflight_writes),
                 write_worker_threads_);
    if (static_cast<size_t>(configured_inflight_reads) >
        max_inflight_reads_) {
        LOG(WARNING) << "GDS max_inflight_reads=" << configured_inflight_reads
                     << " exceeds read_worker_threads="
                     << read_worker_threads_ << "; clamping inflight to "
                     << max_inflight_reads_
                     << " so queued work is not counted as inflight";
    }
    if (static_cast<size_t>(configured_inflight_writes) >
        max_inflight_writes_) {
        LOG(WARNING) << "GDS max_inflight_writes="
                     << configured_inflight_writes
                     << " exceeds write_worker_threads="
                     << write_worker_threads_ << "; clamping inflight to "
                     << max_inflight_writes_
                     << " so queued work is not counted as inflight";
    }
    const int configured_submit_retries =
        conf_->get("transports/gds/submit_retry_count",
                   static_cast<int>(kDefaultSubmitRetryCount));
    const int configured_sample_window = conf_->get(
        "transports/gds/adaptive_sample_window",
        static_cast<int>(kDefaultAdaptiveSampleWindow));
    const int configured_evaluation_interval = conf_->get(
        "transports/gds/adaptive_evaluation_interval",
        static_cast<int>(kDefaultAdaptiveEvaluationInterval));
    const int configured_recovery_windows = conf_->get(
        "transports/gds/adaptive_recovery_windows",
        static_cast<int>(kDefaultAdaptiveRecoveryWindows));
    const int configured_min_read_inflight = conf_->get(
        "transports/gds/adaptive_min_read_inflight",
        static_cast<int>(kDefaultMinReadInflight));
    const int configured_min_write_inflight = conf_->get(
        "transports/gds/adaptive_min_write_inflight",
        static_cast<int>(kDefaultMinWriteInflight));
    const int configured_read_p99_target_us = conf_->get(
        "transports/gds/adaptive_read_p99_target_us", 0);
    const int configured_write_p99_target_us = conf_->get(
        "transports/gds/adaptive_write_p99_target_us",
        static_cast<int>(kDefaultWriteP99TargetUs));
    const int configured_write_starvation_us = conf_->get(
        "transports/gds/write_starvation_timeout_us",
        static_cast<int>(kDefaultWriteStarvationTimeoutUs));
    adaptive_concurrency_enabled_ =
        conf_->get("transports/gds/adaptive_concurrency", true);
    adaptive_degradation_ratio_ = conf_->get(
        "transports/gds/adaptive_p99_degradation_ratio",
        kDefaultAdaptiveDegradationRatio);
    adaptive_recovery_ratio_ = conf_->get(
        "transports/gds/adaptive_p99_recovery_ratio",
        kDefaultAdaptiveRecoveryRatio);
    if (configured_submit_retries < 0 || configured_sample_window <= 0 ||
        configured_evaluation_interval <= 0 ||
        configured_evaluation_interval > configured_sample_window ||
        configured_recovery_windows <= 0 ||
        configured_min_read_inflight <= 0 ||
        configured_min_write_inflight <= 0 ||
        configured_read_p99_target_us < 0 ||
        configured_write_p99_target_us < 0 ||
        configured_write_starvation_us <= 0 ||
        adaptive_degradation_ratio_ <= 1.0 ||
        adaptive_recovery_ratio_ < 1.0 ||
        adaptive_recovery_ratio_ >= adaptive_degradation_ratio_) {
        return Status::InvalidArgument(
            "Invalid GDS direct IO or adaptive concurrency configuration"
            LOC_MARK);
    }
    submit_retry_count_ = static_cast<size_t>(configured_submit_retries);
    adaptive_sample_window_ = static_cast<size_t>(configured_sample_window);
    adaptive_evaluation_interval_ =
        static_cast<size_t>(configured_evaluation_interval);
    adaptive_recovery_windows_ =
        static_cast<size_t>(configured_recovery_windows);
    write_starvation_timeout_ =
        std::chrono::microseconds(configured_write_starvation_us);
    // install() may follow uninstall() on the same object. Discard every
    // rolling sample and recovery counter from the previous driver session.
    read_adaptive_ = GdsAdaptiveState{};
    write_adaptive_ = GdsAdaptiveState{};
    read_adaptive_.configured_limit = max_inflight_reads_;
    read_adaptive_.current_limit = max_inflight_reads_;
    read_adaptive_.minimum_limit =
        std::min(static_cast<size_t>(configured_min_read_inflight),
                 max_inflight_reads_);
    read_adaptive_.target_p99_us = configured_read_p99_target_us;
    read_adaptive_.baseline_p99_us = configured_read_p99_target_us;
    write_adaptive_.configured_limit = max_inflight_writes_;
    write_adaptive_.current_limit = max_inflight_writes_;
    write_adaptive_.minimum_limit =
        std::min(static_cast<size_t>(configured_min_write_inflight),
                 max_inflight_writes_);
    write_adaptive_.target_p99_us = configured_write_p99_target_us;
    write_adaptive_.baseline_p99_us = configured_write_p99_target_us;
    pending_reads_.clear();
    pending_writes_.clear();
    pending_read_bytes_ = 0;
    pending_write_bytes_ = 0;
    inflight_direct_reads_.clear();
    inflight_direct_writes_.clear();
    active_read_workers_ = 0;
    active_write_workers_ = 0;
    next_direct_io_id_ = 1;
    read_window_blocked_ = false;
    write_window_blocked_ = false;
    runtime_queued_reads_.store(0, std::memory_order_relaxed);
    runtime_queued_writes_.store(0, std::memory_order_relaxed);
    read_io_summary_ = IoSummaryDirection{};
    write_io_summary_ = IoSummaryDirection{};
    for (auto* samples : {&read_io_summary_.queue_wait_us,
                          &read_io_summary_.io_latency_us,
                          &read_io_summary_.total_latency_us,
                          &write_io_summary_.queue_wait_us,
                          &write_io_summary_.io_latency_us,
                          &write_io_summary_.total_latency_us}) {
        samples->reserve(kIoSummarySampleCapacity);
    }
    io_summary_started_at_ = std::chrono::steady_clock::now();
    last_read_minimum_degraded_warning_at_ = {};
    last_write_minimum_degraded_warning_at_ = {};

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
    LOG(INFO) << "GDS transport limits: mode=parallel_single_io"
              << ", cufile_batch_api=disabled"
              << ", registered_max_io_size=" << max_io_size_
              << ", unregistered_max_io_size="
              << std::min(max_io_size_, kSafeUnregisteredIoSize)
              << ", read_worker_threads=" << read_worker_threads_
              << ", write_worker_threads=" << write_worker_threads_
              << ", max_inflight_reads=" << max_inflight_reads_
              << ", max_inflight_writes=" << max_inflight_writes_
              << ", submit_retry_count=" << submit_retry_count_
              << ", adaptive_concurrency="
              << adaptive_concurrency_enabled_
              << ", adaptive_min_read_inflight="
              << read_adaptive_.minimum_limit
              << ", adaptive_min_write_inflight="
              << write_adaptive_.minimum_limit
              << ", adaptive_sample_window=" << adaptive_sample_window_
              << ", adaptive_evaluation_interval="
              << adaptive_evaluation_interval_
              << ", adaptive_p99_degradation_ratio="
              << adaptive_degradation_ratio_
              << ", adaptive_p99_recovery_ratio="
              << adaptive_recovery_ratio_
              << ", adaptive_read_p99_target_us="
              << read_adaptive_.baseline_p99_us
              << ", adaptive_write_p99_target_us="
              << write_adaptive_.baseline_p99_us
              << ", write_starvation_timeout_us="
              << write_starvation_timeout_.count();
    shutting_down_.store(false, std::memory_order_release);
    read_thread_pool_ = std::make_unique<ThreadPool>(read_worker_threads_);
    write_thread_pool_ = std::make_unique<ThreadPool>(write_worker_threads_);
    updateIoMetricsLocked();
    installed_ = true;
    caps.dram_to_file = true;
    caps.gpu_to_file = true;
    return Status::OK();
}

Status GdsTransport::uninstall() {
    if (installed_) {
        // Stop admitting new direct IO, then drain/join both worker pools
        // before sub-batches are released. Do not hold the
        // scheduler mutex while joining: completing workers acquire it to
        // publish their final status.
        shutting_down_.store(true, std::memory_order_release);
        std::unique_ptr<ThreadPool> read_pool_to_stop;
        std::unique_ptr<ThreadPool> write_pool_to_stop;
        {
            std::lock_guard<std::mutex> scheduler_guard(scheduler_lock_);
            read_pool_to_stop = std::move(read_thread_pool_);
            write_pool_to_stop = std::move(write_thread_pool_);
        }
        read_pool_to_stop.reset();
        write_pool_to_stop.reset();
        {
            std::lock_guard<std::mutex> scheduler_guard(scheduler_lock_);
            pending_reads_.clear();
            pending_writes_.clear();
            pending_read_bytes_ = 0;
            pending_write_bytes_ = 0;
            inflight_direct_reads_.clear();
            inflight_direct_writes_.clear();
            active_read_workers_ = 0;
            active_write_workers_ = 0;
            read_window_blocked_ = false;
            write_window_blocked_ = false;
            updateIoMetricsLocked();
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
    gds_batch->io_param_ranges.clear();
    gds_batch->io_statuses.clear();
    gds_batch->io_statuses.reserve(max_size);
    gds_batch->io_transferred_bytes.clear();
    gds_batch->io_transferred_bytes.reserve(max_size);

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

bool GdsTransport::subBatchHasWorkLocked(const GdsSubBatch* batch) const {
    if (!batch) return false;
    if (std::any_of(pending_reads_.begin(), pending_reads_.end(),
                    [batch](const PendingIo& pending) {
                        return pending.owner == batch;
                    })) {
        return true;
    }
    if (std::any_of(pending_writes_.begin(), pending_writes_.end(),
                    [batch](const PendingIo& pending) {
                        return pending.owner == batch;
                    })) {
        return true;
    }
    if (std::any_of(
            inflight_direct_reads_.begin(),
            inflight_direct_reads_.end(),
            [batch](const auto& entry) {
                return entry.second && entry.second->owner == batch;
            })) {
        return true;
    }
    if (std::any_of(
            inflight_direct_writes_.begin(),
            inflight_direct_writes_.end(),
            [batch](const auto& entry) {
                return entry.second && entry.second->owner == batch;
            })) {
        return true;
    }
    return false;
}

Status GdsTransport::dispatchDirectIoLocked(bool write,
                                            size_t max_to_schedule) {
    ThreadPool* const thread_pool =
        write ? write_thread_pool_.get() : read_thread_pool_.get();
    auto& inflight_ios =
        write ? inflight_direct_writes_ : inflight_direct_reads_;
    auto& pending_ios = write ? pending_writes_ : pending_reads_;
    auto& pending_bytes = write ? pending_write_bytes_ : pending_read_bytes_;
    const auto& adaptive = write ? write_adaptive_ : read_adaptive_;
    const size_t worker_threads =
        write ? write_worker_threads_ : read_worker_threads_;
    const char* const operation_name = write ? "cuFileWrite" : "cuFileRead";

    if (!thread_pool) {
        return Status::InvalidEntry(
            "GDS parallel IO workers are not available" LOC_MARK);
    }
    if (shutting_down_.load(std::memory_order_acquire)) {
        return Status::InvalidEntry("GDS transport is shutting down" LOC_MARK);
    }

    const size_t available_slots = gdsAvailableWorkerSlots(
        adaptive.current_limit, inflight_ios.size());
    if (available_slots == 0) {
        markSaturatedIoLocked();
        return Status::OK();
    }
    const size_t schedule_count =
        std::min({available_slots, max_to_schedule, pending_ios.size()});
    if (schedule_count == 0) return Status::OK();

    std::vector<std::shared_ptr<DirectIo>> scheduled;
    scheduled.reserve(schedule_count);
    size_t scheduled_bytes = 0;
    while (scheduled.size() < schedule_count) {
        auto pending = std::move(pending_ios.front());
        pending_ios.pop_front();
        auto direct_io = std::make_shared<DirectIo>(DirectIo{
            next_direct_io_id_++, pending.owner, pending.param_index,
            pending.device_id, pending.operation, pending.enqueued_at});
        scheduled_bytes += pending.operation.size;
        assert(pending_bytes >= pending.operation.size);
        pending_bytes -= pending.operation.size;
        inflight_ios.emplace(direct_io->id, direct_io);
        scheduled.emplace_back(std::move(direct_io));
    }
    size_t enqueue_failures = 0;
    for (const auto& direct_io : scheduled) {
        try {
            (void)thread_pool->enqueue(
                [this, direct_io] { executeDirectIo(direct_io); });
        } catch (const std::exception& error) {
            ++enqueue_failures;
            inflight_ios.erase(direct_io->id);
            if (direct_io->owner &&
                direct_io->param_index <
                    direct_io->owner->io_statuses.size()) {
                direct_io->owner->io_statuses[direct_io->param_index] = FAILED;
                direct_io->owner->notifyProgress();
            }
            LOG(ERROR) << "Failed to enqueue parallel GDS "
                       << operation_name << ": " << error.what();
        }
    }
    markSaturatedIoLocked();

    if (!scheduled.empty()) {
        LOG_EVERY_N(INFO, 64)
            << "GDS parallel " << operation_name
            << " dispatch: scheduled_ios="
            << scheduled.size() - enqueue_failures
            << ", scheduled_bytes=" << scheduled_bytes
            << ", inflight_ios=" << inflight_ios.size()
            << ", worker_threads=" << worker_threads
            << ", effective_inflight_limit=" << adaptive.current_limit
            << ", queued_ios_after_dispatch=" << pending_ios.size();
    }
    updateIoMetricsLocked();
    return Status::OK();
}

void GdsTransport::executeDirectIo(std::shared_ptr<DirectIo> direct_io) {
    const bool write = direct_io->operation.write;
    const char* const operation_name = write ? "cuFileWrite" : "cuFileRead";
    TransferStatusEnum final_status = FAILED;
    size_t transferred_bytes = 0;
    ssize_t direct_result = -1;
    int direct_errno = 0;
    std::chrono::steady_clock::time_point worker_started_at;
    {
        std::lock_guard<std::mutex> scheduler_guard(scheduler_lock_);
        auto& inflight_ios =
            write ? inflight_direct_writes_ : inflight_direct_reads_;
        if (inflight_ios.find(direct_io->id) == inflight_ios.end()) return;
        auto& active_workers =
            write ? active_write_workers_ : active_read_workers_;
        ++active_workers;
        auto& summary = write ? write_io_summary_ : read_io_summary_;
        summary.peak_active_workers =
            std::max(summary.peak_active_workers, active_workers);
        worker_started_at = std::chrono::steady_clock::now();
        updateIoMetricsLocked();
    }
    const double queue_wait_seconds = std::chrono::duration<double>(
                                          worker_started_at -
                                          direct_io->enqueued_at)
                                          .count();

    CudaDeviceGuard device_guard;
    auto activation_status = device_guard.activate(direct_io->device_id);
    double io_seconds = 0.0;
    if (activation_status.ok()) {
        const auto io_started_at = std::chrono::steady_clock::now();
        for (size_t attempt = 0; attempt <= submit_retry_count_; ++attempt) {
            errno = 0;
            const auto& operation = direct_io->operation;
            direct_result =
                write ? cuFileWrite(operation.file_handle,
                                    operation.dev_ptr_base, operation.size,
                                    operation.file_offset,
                                    operation.dev_ptr_offset)
                      : cuFileRead(operation.file_handle,
                                   operation.dev_ptr_base, operation.size,
                                   operation.file_offset,
                                   operation.dev_ptr_offset);
            direct_errno = errno;
            if (direct_result == static_cast<ssize_t>(operation.size)) {
                final_status = COMPLETED;
                transferred_bytes = operation.size;
                break;
            }
            if (attempt < submit_retry_count_) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(kSubmitRetryBackoffUs));
            }
        }
        io_seconds = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - io_started_at)
                         .count();
        auto restore_status = device_guard.restore();
        if (!restore_status.ok()) {
            LOG(ERROR) << "Failed to restore CUDA device after parallel GDS "
                       << operation_name << ": "
                       << restore_status.ToString();
        }
    } else {
        LOG(ERROR) << "Failed to activate CUDA device for parallel GDS "
                   << operation_name << ": "
                   << activation_status.ToString();
    }

    std::lock_guard<std::mutex> scheduler_guard(scheduler_lock_);
    auto& inflight_ios =
        write ? inflight_direct_writes_ : inflight_direct_reads_;
    auto inflight_it = inflight_ios.find(direct_io->id);
    if (inflight_it == inflight_ios.end()) return;
    auto& active_workers =
        write ? active_write_workers_ : active_read_workers_;
    assert(active_workers > 0);
    --active_workers;
    if (direct_io->owner &&
        direct_io->param_index < direct_io->owner->io_statuses.size()) {
        direct_io->owner->io_transferred_bytes[direct_io->param_index] =
            transferred_bytes;
        direct_io->owner->io_statuses[direct_io->param_index] = final_status;
    }
    inflight_ios.erase(inflight_it);
    // Measure total latency only after acquiring the completion lock and
    // publishing the terminal state. This includes scheduler-lock contention
    // rather than reporting only worker service time.
    const double total_seconds = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() -
                                     direct_io->enqueued_at)
                                     .count();
    TentMetrics::instance().recordGdsDirectIo(
        !write, direct_io->operation.size, final_status == COMPLETED,
        queue_wait_seconds, io_seconds, total_seconds);
    recordIoSummaryLocked(
        write, transferred_bytes, final_status == COMPLETED,
        queue_wait_seconds * 1e6, io_seconds * 1e6,
        total_seconds * 1e6);
    if (final_status == COMPLETED) {
        LOG_EVERY_N(INFO, 256)
            << "Parallel " << operation_name
            << " completed: bytes=" << transferred_bytes
            << ", queue_wait_us=" << queue_wait_seconds * 1e6
            << ", cufile_latency_us=" << io_seconds * 1e6
            << ", total_latency_us=" << total_seconds * 1e6;
    } else {
        const auto& operation = direct_io->operation;
        LOG_EVERY_N(ERROR, 64)
            << "Parallel " << operation_name
            << " failed: result=" << direct_result
            << ", errno=" << direct_errno
            << ", device_id=" << direct_io->device_id
            << ", dev_ptr=" << operation.dev_ptr_base
            << ", dev_offset=" << operation.dev_ptr_offset
            << ", file_offset=" << operation.file_offset
            << ", size=" << operation.size
            << ", queue_wait_us=" << queue_wait_seconds * 1e6
            << ", cufile_latency_us=" << io_seconds * 1e6
            << ", total_latency_us=" << total_seconds * 1e6;
    }
    if (io_seconds > 0.0) {
        recordAdaptiveSampleLocked(
            write, io_seconds * 1e6,
            direct_io->dispatched_at_capacity);
    }
    if (direct_io->owner) direct_io->owner->notifyProgress();
    updateIoMetricsLocked();

    if (!shutting_down_.load(std::memory_order_acquire)) {
        auto dispatch_status = dispatchPendingIoLocked();
        if (!dispatch_status.ok()) {
            LOG(ERROR) << "Failed to refill GDS dispatch window: "
                       << dispatch_status.ToString();
        }
    }
}

Status GdsTransport::dispatchPendingIoLocked() {
    if (shutting_down_.load(std::memory_order_acquire)) {
        return Status::OK();
    }

    // Fill the READ window first. Each dequeued entry is submitted separately;
    // dispatchDirectIoLocked never places more work in ThreadPool than there
    // are effective worker slots.
    CHECK_STATUS(dispatchDirectIoLocked(
        false, std::numeric_limits<size_t>::max()));

    const auto now = std::chrono::steady_clock::now();
    const bool reads_waiting = !pending_reads_.empty();
    const bool reads_active =
        reads_waiting || !inflight_direct_reads_.empty();
    const bool oldest_write_starved =
        reads_active && !pending_writes_.empty() &&
        now - pending_writes_.front().enqueued_at >=
            write_starvation_timeout_;

    if (!reads_active) {
        CHECK_STATUS(dispatchDirectIoLocked(
            true, std::numeric_limits<size_t>::max()));
    } else if (oldest_write_starved && inflight_direct_writes_.empty()) {
        // READ remains strict first choice, but one old WRITE may pass after
        // the starvation timeout so a continuous restore workload cannot
        // permanently prevent cache persistence.
        CHECK_STATUS(dispatchDirectIoLocked(true, 1));
    }

    const bool read_window_blocked =
        !pending_reads_.empty() &&
        inflight_direct_reads_.size() >= read_adaptive_.current_limit;
    const bool write_window_blocked =
        !pending_writes_.empty() &&
        (inflight_direct_writes_.size() >=
             write_adaptive_.current_limit ||
         reads_active);

    if (read_window_blocked && !read_window_blocked_) {
        TentMetrics::instance().recordGdsDispatchWindowFull(
            pending_reads_.size(),
            inflight_direct_reads_.size() +
                inflight_direct_writes_.size());
        LOG_EVERY_N(INFO, 256)
            << "GDS READ dispatch window full: queued_ios="
            << pending_reads_.size()
            << ", inflight_reads=" << inflight_direct_reads_.size()
            << ", effective_limit=" << read_adaptive_.current_limit;
    }
    if (write_window_blocked && !write_window_blocked_) {
        TentMetrics::instance().recordGdsDispatchWindowFull(
            pending_writes_.size(),
            inflight_direct_reads_.size() +
                inflight_direct_writes_.size());
        LOG_EVERY_N(INFO, 256)
            << "GDS WRITE dispatch delayed: queued_ios="
            << pending_writes_.size()
            << ", inflight_writes=" << inflight_direct_writes_.size()
            << ", effective_limit=" << write_adaptive_.current_limit
            << ", reads_active=" << reads_active;
    }
    read_window_blocked_ = read_window_blocked;
    write_window_blocked_ = write_window_blocked;
    updateIoMetricsLocked();
    return Status::OK();
}

void GdsTransport::markSaturatedIoLocked() {
    const bool reads_active =
        !pending_reads_.empty() || !inflight_direct_reads_.empty() ||
        runtime_queued_reads_.load(std::memory_order_relaxed) > 0;
    const size_t effective_write_limit = gdsEffectiveWriteLimit(
        write_adaptive_.current_limit, reads_active);

    if (effective_write_limit > 0 &&
        inflight_direct_writes_.size() >= effective_write_limit) {
        // Under READ pressure the scheduler deliberately exposes only one
        // WRITE slot. Treat that real dispatch ceiling as saturation so slow
        // mixed WRITE samples can reduce the adaptive limit from 4/2 to 1.
        for (auto& entry : inflight_direct_writes_) {
            if (entry.second) entry.second->dispatched_at_capacity = true;
        }
    }

    if (gdsReadWindowSaturated(
            read_adaptive_.current_limit,
            inflight_direct_reads_.size(),
            inflight_direct_writes_.size(),
            write_adaptive_.current_limit <=
                write_adaptive_.minimum_limit)) {
        // Mark every READ sharing the full device window. WRITE pressure is
        // counted only after WRITE reached its floor, preserving READ-first
        // adaptation.
        for (auto& entry : inflight_direct_reads_) {
            if (entry.second) entry.second->dispatched_at_capacity = true;
        }
    }
}

void GdsTransport::recordIoSummaryLocked(
    bool write, size_t transferred_bytes, bool success,
    double queue_wait_us, double io_latency_us,
    double total_latency_us) {
    auto& summary = write ? write_io_summary_ : read_io_summary_;
    ++summary.completions;
    if (!success) ++summary.failures;
    if (transferred_bytes <=
        std::numeric_limits<size_t>::max() - summary.bytes) {
        summary.bytes += transferred_bytes;
    } else {
        summary.bytes = std::numeric_limits<size_t>::max();
    }
    summary.peak_active_workers = std::max(
        summary.peak_active_workers,
        write ? active_write_workers_ : active_read_workers_);

    const size_t sample_index =
        summary.samples_seen % kIoSummarySampleCapacity;
    auto append_sample = [sample_index](std::vector<double>& samples,
                                        double value) {
        if (samples.size() < kIoSummarySampleCapacity) {
            samples.push_back(value);
        } else {
            samples[sample_index] = value;
        }
    };
    append_sample(summary.queue_wait_us, queue_wait_us);
    append_sample(summary.io_latency_us, io_latency_us);
    append_sample(summary.total_latency_us, total_latency_us);
    ++summary.samples_seen;

    maybeLogIoSummaryLocked(std::chrono::steady_clock::now());
}

void GdsTransport::maybeLogIoSummaryLocked(
    std::chrono::steady_clock::time_point now) {
    if (io_summary_started_at_.time_since_epoch().count() == 0) {
        io_summary_started_at_ = now;
        return;
    }
    const auto elapsed = now - io_summary_started_at_;
    if (elapsed < kIoSummaryInterval) return;

    const double elapsed_seconds =
        std::chrono::duration<double>(elapsed).count();
    LOG(INFO)
        << "GDS IO 1s summary: window_ms=" << elapsed_seconds * 1000.0
        << ", READ{completions=" << read_io_summary_.completions
        << ", failures=" << read_io_summary_.failures
        << ", bytes=" << read_io_summary_.bytes
        << ", throughput_mib_s="
        << static_cast<double>(read_io_summary_.bytes) /
               (1024.0 * 1024.0 * elapsed_seconds)
        << ", queue_wait_p99_us="
        << gdsNearestRankP99(read_io_summary_.queue_wait_us)
        << ", cufile_p99_us="
        << gdsNearestRankP99(read_io_summary_.io_latency_us)
        << ", total_p99_us="
        << gdsNearestRankP99(read_io_summary_.total_latency_us)
        << ", active_workers=" << active_read_workers_
        << ", peak_active_workers="
        << read_io_summary_.peak_active_workers
        << ", inflight=" << inflight_direct_reads_.size()
        << ", internal_queued=" << pending_reads_.size()
        << ", runtime_queued_owners="
        << runtime_queued_reads_.load(std::memory_order_relaxed)
        << ", effective_limit=" << read_adaptive_.current_limit
        << "}, WRITE{completions=" << write_io_summary_.completions
        << ", failures=" << write_io_summary_.failures
        << ", bytes=" << write_io_summary_.bytes
        << ", throughput_mib_s="
        << static_cast<double>(write_io_summary_.bytes) /
               (1024.0 * 1024.0 * elapsed_seconds)
        << ", queue_wait_p99_us="
        << gdsNearestRankP99(write_io_summary_.queue_wait_us)
        << ", cufile_p99_us="
        << gdsNearestRankP99(write_io_summary_.io_latency_us)
        << ", total_p99_us="
        << gdsNearestRankP99(write_io_summary_.total_latency_us)
        << ", active_workers=" << active_write_workers_
        << ", peak_active_workers="
        << write_io_summary_.peak_active_workers
        << ", inflight=" << inflight_direct_writes_.size()
        << ", internal_queued=" << pending_writes_.size()
        << ", runtime_queued_owners="
        << runtime_queued_writes_.load(std::memory_order_relaxed)
        << ", effective_limit=" << write_adaptive_.current_limit
        << ", runtime_dispatch_limit="
        << gdsEffectiveWriteLimit(
               write_adaptive_.current_limit,
               !pending_reads_.empty() ||
                   !inflight_direct_reads_.empty() ||
                   runtime_queued_reads_.load(
                       std::memory_order_relaxed) > 0)
        << "}}";

    auto reset_summary = [](IoSummaryDirection& summary,
                            size_t active_workers) {
        summary.completions = 0;
        summary.failures = 0;
        summary.bytes = 0;
        summary.peak_active_workers = active_workers;
        summary.samples_seen = 0;
        summary.queue_wait_us.clear();
        summary.io_latency_us.clear();
        summary.total_latency_us.clear();
    };
    reset_summary(read_io_summary_, active_read_workers_);
    reset_summary(write_io_summary_, active_write_workers_);
    io_summary_started_at_ = now;
}

void GdsTransport::updateIoMetricsLocked() const {
    TentMetrics::instance().updateGdsIoState(
        pending_reads_.size(), pending_read_bytes_,
        pending_writes_.size(), pending_write_bytes_,
        active_read_workers_, active_write_workers_,
        read_adaptive_.current_limit, write_adaptive_.current_limit);
}

void GdsTransport::recordAdaptiveSampleLocked(
    bool write, double io_latency_us, bool dispatched_at_capacity) {
    auto& state = write ? write_adaptive_ : read_adaptive_;
    state.saturation_since_evaluation |= dispatched_at_capacity;
    state.recent_io_latency_us.push_back(io_latency_us);
    while (state.recent_io_latency_us.size() > adaptive_sample_window_) {
        state.recent_io_latency_us.pop_front();
    }
    ++state.completions_since_evaluation;

    if (!adaptive_concurrency_enabled_ ||
        !gdsAdaptiveEvaluationReady(
            state, adaptive_sample_window_,
            adaptive_evaluation_interval_)) {
        return;
    }
    state.completions_since_evaluation = 0;
    evaluateAdaptiveConcurrencyLocked(write);
}

void GdsTransport::evaluateAdaptiveConcurrencyLocked(bool write) {
    auto& state = write ? write_adaptive_ : read_adaptive_;
    if (state.recent_io_latency_us.empty()) return;

    const double p99_us = gdsNearestRankP99(std::vector<double>(
        state.recent_io_latency_us.begin(),
        state.recent_io_latency_us.end()));
    const auto& pending = write ? pending_writes_ : pending_reads_;
    const size_t runtime_queued =
        write ? runtime_queued_writes_.load(std::memory_order_relaxed)
              : runtime_queued_reads_.load(std::memory_order_relaxed);
    const size_t queued_ios =
        pending.size() > std::numeric_limits<size_t>::max() - runtime_queued
            ? std::numeric_limits<size_t>::max()
            : pending.size() + runtime_queued;
    TentMetrics::instance().observeGdsAdaptiveWindow(
        !write, p99_us / 1e6, queued_ios);
    const size_t old_limit = state.current_limit;
    const bool reached_effective_limit =
        state.saturation_since_evaluation;
    const auto action = adjustGdsAdaptiveConcurrency(
        state, p99_us, queued_ios, adaptive_degradation_ratio_,
        adaptive_recovery_ratio_, adaptive_recovery_windows_);
    if (action == GdsAdaptiveAction::NONE) return;

    state.recent_io_latency_us.clear();
    if (action == GdsAdaptiveAction::HOLD_AT_MINIMUM) {
        auto& last_warning =
            write ? last_write_minimum_degraded_warning_at_
                  : last_read_minimum_degraded_warning_at_;
        const auto now = std::chrono::steady_clock::now();
        if (last_warning.time_since_epoch().count() == 0 ||
            now - last_warning >= kMinimumDegradedWarningInterval) {
            LOG(WARNING)
                << "GDS adaptive concurrency remains degraded at minimum "
                << (write ? "WRITE" : "READ")
                << " inflight=" << state.current_limit
                << ", consecutive_degraded_windows="
                << state.degraded_at_minimum_windows
                << ", rolling_p99_us=" << p99_us
                << ", baseline_p99_us=" << state.baseline_p99_us
                << ", target_p99_us=" << state.target_p99_us
                << ", reached_effective_limit="
                << reached_effective_limit
                << ", internal_queued_ios=" << pending.size()
                << ", runtime_queued_owners=" << runtime_queued;
            last_warning = now;
        }
        return;
    }
    const bool reduced = action == GdsAdaptiveAction::REDUCE;
    TentMetrics::instance().recordGdsAdaptiveConcurrency(
        !write, reduced, state.current_limit);
    if (reduced) {
        LOG(WARNING) << "GDS adaptive concurrency reduced "
                     << (write ? "WRITE" : "READ")
                     << " effective inflight: " << old_limit << " -> "
                     << state.current_limit
                     << ", rolling_p99_us=" << p99_us
                     << ", baseline_p99_us=" << state.baseline_p99_us
                     << ", target_p99_us=" << state.target_p99_us
                     << ", reached_effective_limit="
                     << reached_effective_limit
                     << ", internal_queued_ios=" << pending.size()
                     << ", runtime_queued_owners=" << runtime_queued;
    } else {
        LOG(INFO) << "GDS adaptive concurrency cautiously recovered "
                  << (write ? "WRITE" : "READ")
                  << " effective inflight: " << old_limit << " -> "
                  << state.current_limit << ", rolling_p99_us=" << p99_us
                  << ", baseline_p99_us=" << state.baseline_p99_us
                  << ", target_p99_us=" << state.target_p99_us
                  << ", reached_effective_limit="
                  << reached_effective_limit
                  << ", internal_queued_ios=" << pending.size()
                  << ", runtime_queued_owners=" << runtime_queued;
    }
    updateIoMetricsLocked();
}

Status GdsTransport::pollInflightIoLocked() {
    return dispatchPendingIoLocked();
}

Status GdsTransport::validateRequest(const Request& request) {
    if (request.opcode != Request::READ &&
        request.opcode != Request::WRITE) {
        return Status::InvalidArgument(
            "GDS supports only READ and WRITE requests" LOC_MARK);
    }
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
    if (shutting_down_.load(std::memory_order_acquire)) {
        return Status::InvalidEntry("GDS transport is shutting down" LOC_MARK);
    }
    if (gds_batch->io_param_ranges.size() > gds_batch->max_size ||
        request_list.size() >
            gds_batch->max_size - gds_batch->io_param_ranges.size()) {
        return Status::TooManyRequests(
            "Exceed GDS logical batch capacity" LOC_MARK);
    }

    size_t num_ios = 0;
    std::vector<GdsFileContext*> contexts;
    std::vector<void*> io_bases;
    std::vector<size_t> io_offsets;
    std::vector<size_t> io_chunk_sizes;
    std::vector<int> device_ids;
    contexts.reserve(request_list.size());
    io_bases.reserve(request_list.size());
    io_offsets.reserve(request_list.size());
    io_chunk_sizes.reserve(request_list.size());
    device_ids.reserve(request_list.size());

    // Validate and resolve every logical request before mutating the sub-batch
    // or transport queues. A vector received from the runtime queue is merely
    // a convenient dequeue unit; it never becomes one physical cuFile batch.
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

        // Registered CUDA buffers can use the driver's full direct-I/O size.
        // The 960 KiB limit applies only to an unregistered/bounce-buffer path.
        const size_t io_chunk_size =
            request_device_id >= 0
                ? max_io_size_
                : std::min(max_io_size_, kSafeUnregisteredIoSize);
        const size_t request_ios =
            1 + (request.length - 1) / io_chunk_size;
        if (request_ios > std::numeric_limits<size_t>::max() - num_ios) {
            return Status::TooManyRequests(
                "GDS request count overflows" LOC_MARK);
        }
        num_ios += request_ios;

        GdsFileContext* context = findFileContext(request.target_id);
        if (!context || !context->ready()) {
            return Status::InvalidArgument("Invalid remote segment" LOC_MARK);
        }
        contexts.push_back(context);
        io_bases.push_back(io_base);
        io_offsets.push_back(io_offset);
        io_chunk_sizes.push_back(io_chunk_size);
        device_ids.push_back(request_device_id);
    }

    const auto enqueued_at = std::chrono::steady_clock::now();
    size_t read_io_count = 0;
    size_t write_io_count = 0;
    for (size_t request_index = 0; request_index < request_list.size();
         ++request_index) {
        const auto& request = request_list[request_index];
        const bool write = request.opcode == Request::WRITE;
        IOParamRange range{gds_batch->io_statuses.size(), 0};
        const size_t io_chunk_size = io_chunk_sizes[request_index];

        for (size_t offset = 0; offset < request.length;
             offset += io_chunk_size) {
            const size_t length =
                std::min(io_chunk_size, request.length - offset);
            const size_t param_index = gds_batch->io_statuses.size();
            IoOperation operation{
                contexts[request_index]->getHandle(),
                io_bases[request_index],
                length,
                static_cast<off_t>(request.target_offset + offset),
                static_cast<off_t>(io_offsets[request_index] + offset),
                write,
            };
            gds_batch->io_statuses.push_back(PENDING);
            gds_batch->io_transferred_bytes.push_back(0);
            PendingIo pending{gds_batch, param_index,
                              device_ids[request_index], operation,
                              enqueued_at};
            if (write) {
                pending_writes_.push_back(std::move(pending));
                pending_write_bytes_ += length;
                ++write_io_count;
            } else {
                pending_reads_.push_back(std::move(pending));
                pending_read_bytes_ += length;
                ++read_io_count;
            }
            ++range.count;
        }
        gds_batch->io_param_ranges.push_back(range);
    }

    const size_t small_request_count = static_cast<size_t>(std::count_if(
        request_list.begin(), request_list.end(),
        [](const Request& request) { return request.length <= 64 * 1024; }));
    TentMetrics::instance().recordGdsTransportSubmission(
        request_list.size(), num_ios, num_ios, small_request_count);
    LOG_EVERY_N(INFO, 256)
        << "GDS single-IO submission: logical_requests="
        << request_list.size() << ", physical_ios=" << num_ios
        << ", read_ios=" << read_io_count
        << ", write_ios=" << write_io_count
        << ", queued_reads=" << pending_reads_.size()
        << ", queued_writes=" << pending_writes_.size()
        << ", read_effective_inflight=" << read_adaptive_.current_limit
        << ", write_effective_inflight=" << write_adaptive_.current_limit;

    updateIoMetricsLocked();
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
