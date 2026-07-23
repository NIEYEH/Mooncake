// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "tent/common/config.h"
#include "tent/common/types.h"
#include "tent/transfer_engine.h"
#include "tent/transport/gds/gds_transport.h"

namespace mooncake {
namespace tent {
namespace {

constexpr char kTestPathEnv[] = "MOONCAKE_GDS_TEST_PATH";
constexpr char kReaderPathEnv[] = "MOONCAKE_GDS_TEST_READER_PATH";
constexpr char kWriteEnableEnv[] = "MOONCAKE_GDS_TEST_ALLOW_WRITE";
constexpr char kReadOnlyEnv[] = "MOONCAKE_GDS_TEST_READ_ONLY";
constexpr char kDeviceConfirmEnv[] = "MOONCAKE_GDS_TEST_DEVICE_CONFIRM";
constexpr char kOffsetEnv[] = "MOONCAKE_GDS_TEST_OFFSET";
constexpr char kLengthEnv[] = "MOONCAKE_GDS_TEST_LENGTH";
constexpr char kGpuIdEnv[] = "MOONCAKE_GDS_TEST_GPU_ID";

class ScopedFd {
   public:
    explicit ScopedFd(int fd) : fd_(fd) {}
    ~ScopedFd() {
        if (fd_ >= 0) close(fd_);
    }

    int get() const { return fd_; }

   private:
    int fd_;
};

class RegisteredCudaBuffer {
   public:
    RegisteredCudaBuffer() = default;
    ~RegisteredCudaBuffer() {
        if (registered_) (void)engine_->unregisterLocalMemory(ptr_, length_);
        if (ptr_) (void)cudaFree(ptr_);
    }

    RegisteredCudaBuffer(const RegisteredCudaBuffer&) = delete;
    RegisteredCudaBuffer& operator=(const RegisteredCudaBuffer&) = delete;

    bool initialize(TransferEngine& engine, size_t length, int gpu_id,
                    std::string& error) {
        auto cuda_status = cudaMalloc(&ptr_, length);
        if (cuda_status != cudaSuccess) {
            error = std::string("cudaMalloc failed: ") +
                    cudaGetErrorString(cuda_status);
            ptr_ = nullptr;
            return false;
        }

        MemoryOptions options;
        options.location = "cuda:" + std::to_string(gpu_id);
        auto status = engine.registerLocalMemory(ptr_, length, options);
        if (!status.ok()) {
            error = "registerLocalMemory failed: " + status.ToString();
            (void)cudaFree(ptr_);
            ptr_ = nullptr;
            return false;
        }

        engine_ = &engine;
        length_ = length;
        registered_ = true;
        return true;
    }

    void* get() const { return ptr_; }

   private:
    TransferEngine* engine_ = nullptr;
    void* ptr_ = nullptr;
    size_t length_ = 0;
    bool registered_ = false;
};

std::optional<uint64_t> parseUnsigned(const char* value) {
    if (!value || value[0] == '\0' || value[0] == '-') return std::nullopt;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 0);
    if (errno != 0 || end == value || *end != '\0') return std::nullopt;
    return static_cast<uint64_t>(parsed);
}

std::string canonicalPath(const std::string& path) {
    std::unique_ptr<char, decltype(&std::free)> resolved(
        realpath(path.c_str(), nullptr), &std::free);
    return resolved ? std::string(resolved.get()) : std::string();
}

std::shared_ptr<Config> makeGdsConfig() {
    auto config = std::make_shared<Config>();
    config->set("metadata_type", "p2p");
    config->set("metadata_servers", "");
    config->set("rpc_server_hostname", "127.0.0.1");
    config->set("rpc_server_port", "0");
    config->set("local_segment_name",
                "gds_block_io_test_" + std::to_string(getpid()));
    config->set("metrics/enabled", false);
    config->set("transports/tcp/enable", false);
    config->set("transports/shm/enable", false);
    config->set("transports/rdma/enable", false);
    config->set("transports/io_uring/enable", false);
    config->set("transports/nvlink/enable", false);
    config->set("transports/mnnvl/enable", false);
    config->set("transports/gds/enable", true);
    config->set("transports/gds/read_worker_threads", 16);
    config->set("transports/gds/write_worker_threads", 4);
    config->set("transports/gds/max_inflight_reads", 16);
    config->set("transports/gds/max_inflight_writes", 4);
    config->set("transports/gds/submit_retry_count", 0);
    config->set("transports/gds/write_starvation_timeout_us", 500000);
    config->set("transports/gds/adaptive_concurrency", true);
    config->set("transports/gds/adaptive_sample_window", 128);
    config->set("transports/gds/adaptive_evaluation_interval", 32);
    config->set("transports/gds/adaptive_recovery_windows", 3);
    config->set("transports/gds/adaptive_min_read_inflight", 4);
    config->set("transports/gds/adaptive_min_write_inflight", 1);
    config->set("transports/gds/adaptive_p99_degradation_ratio", 1.25);
    config->set("transports/gds/adaptive_p99_recovery_ratio", 1.05);
    config->set("transports/gds/adaptive_read_p99_target_us", 60000);
    config->set("transports/gds/adaptive_write_p99_target_us", 0);
    return config;
}

std::vector<uint8_t> makePattern(size_t length, uint8_t seed) {
    std::vector<uint8_t> pattern(length);
    for (size_t i = 0; i < length; ++i) {
        pattern[i] = static_cast<uint8_t>(seed + i * 131u + (i >> 7));
    }
    return pattern;
}

bool runTransfersOnCurrentThread(TransferEngine& engine,
                                 const std::vector<Request>& requests,
                                 TransferStatus& final_status,
                                 std::string& error) {
    BatchID batch = engine.allocateBatch(requests.size());
    if (batch == 0) {
        error = "allocateBatch failed";
        return false;
    }

    auto status = engine.submitTransfer(batch, requests);
    if (!status.ok()) {
        error = "submitTransfer failed: " + status.ToString();
        (void)engine.freeBatch(batch);
        return false;
    }

    final_status = TransferStatus{INITIAL, 0};
    constexpr int kPollLimit = 60000;
    for (int poll = 0; poll < kPollLimit; ++poll) {
        status = engine.getTransferStatus(batch, final_status);
        if (!status.ok()) {
            error = "getTransferStatus failed: " + status.ToString();
            (void)engine.freeBatch(batch);
            return false;
        }
        if (final_status.s != PENDING && final_status.s != INITIAL) {
            status = engine.freeBatch(batch);
            if (!status.ok()) {
                error = "freeBatch failed: " + status.ToString();
                return false;
            }
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    error = "GDS transfer did not finish within 60 seconds";
    (void)engine.freeBatch(batch);
    return false;
}

bool runTransfers(TransferEngine& engine,
                  const std::vector<Request>& requests,
                  TransferStatus& final_status, std::string& error) {
    bool success = false;
    // Exercise the same fresh-worker-thread context used by vLLM batch_put.
    std::thread worker([&] {
        success = runTransfersOnCurrentThread(engine, requests, final_status,
                                              error);
    });
    worker.join();
    return success;
}

bool runTransfer(TransferEngine& engine, const Request& request,
                 TransferStatus& final_status, std::string& error) {
    return runTransfers(engine, {request}, final_status, error);
}

bool runIndependentTransfers(TransferEngine& engine,
                             const std::vector<Request>& requests,
                             std::vector<TransferStatus>& final_statuses,
                             std::string& error) {
    std::vector<BatchID> batches;
    batches.reserve(requests.size());
    final_statuses.assign(requests.size(), TransferStatus{INITIAL, 0});

    auto free_all = [&] {
        for (auto batch : batches) (void)engine.freeBatch(batch);
        batches.clear();
    };

    for (const auto& request : requests) {
        BatchID batch = engine.allocateBatch(1);
        if (batch == 0) {
            error = "allocateBatch failed for independent GDS request";
            free_all();
            return false;
        }
        batches.push_back(batch);
        auto status = engine.submitTransfer(batch, {request});
        if (!status.ok()) {
            error = "submitTransfer failed for independent GDS request: " +
                    status.ToString();
            free_all();
            return false;
        }
    }

    constexpr int kPollLimit = 60000;
    for (int poll = 0; poll < kPollLimit; ++poll) {
        bool all_terminal = true;
        for (size_t index = 0; index < batches.size(); ++index) {
            if (final_statuses[index].s != PENDING &&
                final_statuses[index].s != INITIAL) {
                continue;
            }
            auto status = engine.getTransferStatus(batches[index],
                                                   final_statuses[index]);
            if (!status.ok()) {
                error = "getTransferStatus failed for independent GDS "
                        "request: " +
                        status.ToString();
                free_all();
                return false;
            }
            if (final_statuses[index].s == PENDING ||
                final_statuses[index].s == INITIAL) {
                all_terminal = false;
            }
        }
        if (all_terminal) {
            for (auto batch : batches) {
                auto status = engine.freeBatch(batch);
                if (!status.ok()) {
                    error = "freeBatch failed for independent GDS request: " +
                            status.ToString();
                    return false;
                }
            }
            batches.clear();
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    error = "Independent GDS transfers did not finish within 60 seconds";
    free_all();
    return false;
}

Request makeRequest(Request::OpCode opcode, void* buffer, SegmentID segment,
                    uint64_t offset, size_t length) {
    Request request{};
    request.opcode = opcode;
    request.source = buffer;
    request.target_id = segment;
    request.target_offset = offset;
    request.length = length;
    request.transport_hint = GDS;
    return request;
}

TEST(GdsAdaptiveConcurrencyTest,
     ReducesUnderOverloadAndRecoversOnlyWithoutBacklog) {
    GdsAdaptiveState state;
    state.configured_limit = 16;
    state.current_limit = 16;
    state.minimum_limit = 4;
    state.baseline_p99_us = 100.0;

    EXPECT_EQ(adjustGdsAdaptiveConcurrency(state, 150.0, 64, 1.25, 1.05,
                                           3),
              GdsAdaptiveAction::REDUCE);
    EXPECT_EQ(state.current_limit, 12u);
    EXPECT_DOUBLE_EQ(state.baseline_p99_us, 100.0);

    // Even a healthy-looking P99 cannot raise concurrency while a direction
    // remains saturated and backlogged.
    for (int window = 0; window < 8; ++window) {
        EXPECT_EQ(adjustGdsAdaptiveConcurrency(state, 90.0, 64, 1.25,
                                               1.05, 3),
                  GdsAdaptiveAction::NONE);
    }
    EXPECT_EQ(state.current_limit, 12u);

    EXPECT_EQ(adjustGdsAdaptiveConcurrency(state, 90.0, 0, 1.25, 1.05, 3),
              GdsAdaptiveAction::NONE);
    EXPECT_EQ(adjustGdsAdaptiveConcurrency(state, 90.0, 0, 1.25, 1.05, 3),
              GdsAdaptiveAction::NONE);
    EXPECT_EQ(adjustGdsAdaptiveConcurrency(state, 90.0, 0, 1.25, 1.05, 3),
              GdsAdaptiveAction::RECOVER);
    EXPECT_EQ(state.current_limit, 13u);

    for (int window = 0; window < 30; ++window) {
        (void)adjustGdsAdaptiveConcurrency(state, 90.0, 0, 1.25, 1.05, 3);
    }
    EXPECT_EQ(state.current_limit, state.configured_limit);
}

TEST(GdsAdaptiveConcurrencyTest,
     BurstSaturationReducesAfterInflightAndQueuesAlreadyDrain) {
    GdsAdaptiveState state;
    state.configured_limit = 16;
    state.current_limit = 16;
    state.minimum_limit = 4;
    state.baseline_p99_us = 100.0;

    // The completion that triggers evaluation can observe no remaining
    // inflight or queued IO even though earlier samples filled the window.
    state.saturation_since_evaluation = true;
    EXPECT_EQ(adjustGdsAdaptiveConcurrency(state, 150.0, 0, 1.25, 1.05, 3),
              GdsAdaptiveAction::REDUCE);
    EXPECT_EQ(state.current_limit, 12u);
    EXPECT_FALSE(state.saturation_since_evaluation);
}

TEST(GdsAdaptiveConcurrencyTest, MinimumAndExistingInflightAreUnderflowSafe) {
    GdsAdaptiveState state;
    state.configured_limit = 16;
    state.current_limit = 4;
    state.minimum_limit = 4;
    state.baseline_p99_us = 100.0;

    EXPECT_EQ(adjustGdsAdaptiveConcurrency(state, 200.0, 64, 1.25, 1.05,
                                           3),
              GdsAdaptiveAction::NONE);
    EXPECT_EQ(state.current_limit, 4u);
    EXPECT_DOUBLE_EQ(state.baseline_p99_us, 100.0);
    EXPECT_EQ(gdsAvailableWorkerSlots(4, 8), 0u);
    EXPECT_EQ(gdsAvailableWorkerSlots(4, 4), 0u);
    EXPECT_EQ(gdsAvailableWorkerSlots(4, 3), 1u);
}

TEST(GdsBlockIoTest, DestructiveWriteReadAtTwoOffsets) {
    const char* path_env = std::getenv(kTestPathEnv);
    const char* write_env = std::getenv(kWriteEnableEnv);
    const char* offset_env = std::getenv(kOffsetEnv);
    const char* length_env = std::getenv(kLengthEnv);
    const char* gpu_env = std::getenv(kGpuIdEnv);
    const bool read_only =
        std::strcmp(std::getenv(kReadOnlyEnv)
                        ? std::getenv(kReadOnlyEnv)
                        : "",
                    "YES") == 0;
    if (!path_env || (!read_only &&
                      std::strcmp(write_env ? write_env : "", "YES") != 0) ||
        !offset_env || !length_env || !gpu_env) {
        GTEST_SKIP() << "Destructive GDS test is disabled. Set "
                     << kTestPathEnv << ", " << kWriteEnableEnv
                     << "=YES (unless " << kReadOnlyEnv << "=YES), "
                     << kDeviceConfirmEnv << ", " << kOffsetEnv << ", "
                     << kLengthEnv << ", and " << kGpuIdEnv;
    }

    std::string device_path(path_env);
    if (device_path.rfind(kLocalBlockSegmentPrefix, 0) == 0)
        device_path.erase(0, kLocalBlockSegmentPrefix.size());
    const std::string canonical_device = canonicalPath(device_path);
    ASSERT_FALSE(canonical_device.empty())
        << "realpath failed for " << device_path << ": "
        << std::strerror(errno);

    const char* confirm_env = std::getenv(kDeviceConfirmEnv);
    ASSERT_NE(confirm_env, nullptr)
        << kDeviceConfirmEnv << " must equal the canonical block device path";
    ASSERT_EQ(std::string(confirm_env), canonical_device)
        << kDeviceConfirmEnv << " must match `readlink -f " << device_path
        << "` exactly";

    struct stat device_stat {};
    ASSERT_EQ(stat(canonical_device.c_str(), &device_stat), 0)
        << std::strerror(errno);
    ASSERT_TRUE(S_ISBLK(device_stat.st_mode))
        << canonical_device << " is not a block device";

    std::string reader_device_path = device_path;
    if (const char* reader_path_env = std::getenv(kReaderPathEnv)) {
        reader_device_path = reader_path_env;
        if (reader_device_path.rfind(kLocalBlockSegmentPrefix, 0) == 0) {
            reader_device_path.erase(0, kLocalBlockSegmentPrefix.size());
        }
    }
    struct stat reader_device_stat {};
    ASSERT_EQ(stat(reader_device_path.c_str(), &reader_device_stat), 0)
        << std::strerror(errno);
    ASSERT_TRUE(S_ISBLK(reader_device_stat.st_mode))
        << reader_device_path << " is not a block device";
    ASSERT_EQ(reader_device_stat.st_rdev, device_stat.st_rdev)
        << kReaderPathEnv
        << " must resolve to the same NVMe namespace as " << kTestPathEnv;

    ScopedFd fd(open(canonical_device.c_str(), O_RDWR | O_DIRECT));
    ASSERT_GE(fd.get(), 0) << "open failed for " << canonical_device << ": "
                           << std::strerror(errno);
    uint64_t device_size = 0;
    int logical_block_size = 0;
    ASSERT_EQ(ioctl(fd.get(), BLKGETSIZE64, &device_size), 0)
        << std::strerror(errno);
    ASSERT_EQ(ioctl(fd.get(), BLKSSZGET, &logical_block_size), 0)
        << std::strerror(errno);
    ASSERT_GT(logical_block_size, 0);

    const auto offset = parseUnsigned(offset_env);
    const auto length = parseUnsigned(length_env);
    const auto gpu_id = parseUnsigned(gpu_env);
    ASSERT_TRUE(offset.has_value()) << "Invalid " << kOffsetEnv;
    ASSERT_TRUE(length.has_value()) << "Invalid " << kLengthEnv;
    ASSERT_TRUE(gpu_id.has_value()) << "Invalid " << kGpuIdEnv;
    ASSERT_GT(*length, 0u);
    ASSERT_LE(*length,
              static_cast<uint64_t>(std::numeric_limits<size_t>::max()));
    ASSERT_LE(*gpu_id,
              static_cast<uint64_t>(std::numeric_limits<int>::max()));

    const uint64_t alignment =
        std::max<uint64_t>(static_cast<uint64_t>(logical_block_size), 4096);
    ASSERT_GE(*offset, alignment)
        << "The test intentionally refuses to overwrite the first block";
    ASSERT_EQ(*offset % alignment, 0u);
    ASSERT_EQ(*length % alignment, 0u);
    ASSERT_LE(alignment,
              static_cast<uint64_t>(std::numeric_limits<size_t>::max()));
    ASSERT_LE(*length, static_cast<uint64_t>(
                           std::numeric_limits<size_t>::max()) -
                           alignment);
    ASSERT_LE(*length,
              (std::numeric_limits<uint64_t>::max() - *offset) / 2);
    const uint64_t test_range_end = *offset + *length * 2;
    ASSERT_LE(test_range_end, device_size)
        << "Two test regions do not fit on the block device";

    const int gpu = static_cast<int>(*gpu_id);
    ASSERT_EQ(cudaSetDevice(gpu), cudaSuccess);

    TransferEngine engine(makeGdsConfig());
    ASSERT_TRUE(engine.available()) << "failed to initialize TENT";

    SegmentID segment = 0;
    const std::string segment_uri = kLocalBlockSegmentPrefix + canonical_device;
    auto status = engine.openSegment(segment, segment_uri);
    ASSERT_TRUE(status.ok()) << status.ToString();

    SegmentID reader_segment = segment;
    const std::string reader_segment_uri =
        kLocalBlockSegmentPrefix + reader_device_path;
    if (reader_segment_uri != segment_uri) {
        status = engine.openSegment(reader_segment, reader_segment_uri);
        ASSERT_TRUE(status.ok()) << status.ToString();
    }

    SegmentInfo info;
    status = engine.getSegmentInfo(segment, info);
    ASSERT_TRUE(status.ok()) << status.ToString();
    ASSERT_EQ(info.type, SegmentInfo::File);
    ASSERT_EQ(info.buffers.size(), 1u);
    ASSERT_EQ(info.buffers[0].length, device_size);

    const size_t io_length = static_cast<size_t>(*length);
    const size_t buffer_offset = static_cast<size_t>(alignment);
    const size_t buffer_length = io_length + buffer_offset;
    RegisteredCudaBuffer source;
    RegisteredCudaBuffer destination;
    RegisteredCudaBuffer second_destination;
    std::string error;
    ASSERT_TRUE(source.initialize(engine, buffer_length, gpu, error)) << error;
    ASSERT_TRUE(destination.initialize(engine, buffer_length, gpu, error))
        << error;
    ASSERT_TRUE(
        second_destination.initialize(engine, buffer_length, gpu, error))
        << error;
    void* source_io = static_cast<char*>(source.get()) + buffer_offset;
    void* destination_io =
        static_cast<char*>(destination.get()) + buffer_offset;
    void* second_destination_io =
        static_cast<char*>(second_destination.get()) + buffer_offset;
    ASSERT_NE(source_io, source.get());
    ASSERT_NE(destination_io, destination.get());
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(source_io) % alignment, 0u);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(destination_io) % alignment,
              0u);

    const auto first_pattern = makePattern(io_length, 0x31);
    const auto second_pattern = makePattern(io_length, 0xa7);
    TransferStatus transfer_status{};
    if (!read_only) {
        ASSERT_EQ(cudaMemcpy(source_io, first_pattern.data(), io_length,
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        auto write_request = makeRequest(Request::WRITE, source_io, segment,
                                         *offset, io_length);
        auto second_write_request = write_request;
        second_write_request.target_offset = *offset + *length;
        ASSERT_TRUE(runTransfers(engine, {write_request, second_write_request},
                                 transfer_status, error))
            << error;
        ASSERT_EQ(transfer_status.s, COMPLETED);
        ASSERT_EQ(transfer_status.transferred_bytes, io_length * 2);

        ASSERT_EQ(cudaMemcpy(source_io, second_pattern.data(), io_length,
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        write_request.target_offset = *offset + *length;
        ASSERT_TRUE(
            runTransfer(engine, write_request, transfer_status, error))
            << error;
        ASSERT_EQ(transfer_status.s, COMPLETED);
    }

    std::vector<uint8_t> actual(io_length);
    ASSERT_EQ(cudaMemset(destination_io, 0, io_length), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto request = makeRequest(Request::READ, destination_io, reader_segment,
                               *offset, io_length);
    ASSERT_TRUE(runTransfer(engine, request, transfer_status, error)) << error;
    ASSERT_EQ(transfer_status.s, COMPLETED);
    ASSERT_EQ(cudaMemcpy(actual.data(), destination_io, io_length,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(std::memcmp(actual.data(), first_pattern.data(), io_length), 0);

    ASSERT_EQ(cudaMemset(destination_io, 0, io_length), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    request.target_offset = *offset + *length;
    ASSERT_TRUE(runTransfer(engine, request, transfer_status, error)) << error;
    ASSERT_EQ(transfer_status.s, COMPLETED);
    ASSERT_EQ(cudaMemcpy(actual.data(), destination_io, io_length,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(std::memcmp(actual.data(), second_pattern.data(), io_length), 0);

    // Submit one request per TransferEngine Batch. This is the shape produced
    // by concurrent Store/vLLM loads and verifies that the GDS transport-level
    // scheduler runs independent single IOs across logical SubBatches.
    ASSERT_EQ(cudaMemset(destination_io, 0, io_length), cudaSuccess);
    ASSERT_EQ(cudaMemset(second_destination_io, 0, io_length), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<TransferStatus> independent_statuses;
    ASSERT_TRUE(runIndependentTransfers(
        engine,
        {makeRequest(Request::READ, destination_io, reader_segment, *offset,
                     io_length),
         makeRequest(Request::READ, second_destination_io, reader_segment,
                     *offset + *length, io_length)},
        independent_statuses, error))
        << error;
    ASSERT_EQ(independent_statuses.size(), 2u);
    EXPECT_EQ(independent_statuses[0].s, COMPLETED);
    EXPECT_EQ(independent_statuses[1].s, COMPLETED);
    ASSERT_EQ(cudaMemcpy(actual.data(), destination_io, io_length,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(std::memcmp(actual.data(), first_pattern.data(), io_length), 0);
    ASSERT_EQ(cudaMemcpy(actual.data(), second_destination_io, io_length,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(std::memcmp(actual.data(), second_pattern.data(), io_length), 0);

    request = makeRequest(Request::READ, destination_io, segment,
                          *offset + 1, io_length);
    ASSERT_TRUE(runTransfer(engine, request, transfer_status, error)) << error;
    EXPECT_EQ(transfer_status.s, FAILED);

    request.target_offset = device_size;
    ASSERT_TRUE(runTransfer(engine, request, transfer_status, error)) << error;
    EXPECT_EQ(transfer_status.s, FAILED);

    if (reader_segment != segment) {
        status = engine.closeSegment(reader_segment);
        EXPECT_TRUE(status.ok()) << status.ToString();
    }
    status = engine.closeSegment(segment);
    EXPECT_TRUE(status.ok()) << status.ToString();
}

}  // namespace
}  // namespace tent
}  // namespace mooncake
