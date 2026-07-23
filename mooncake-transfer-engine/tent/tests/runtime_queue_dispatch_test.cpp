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

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "tent/common/config.h"
#include "tent/common/types.h"
#include "tent/runtime/segment.h"
#include "tent/runtime/transfer_engine_impl.h"
#include "tent/runtime/transport.h"

namespace mooncake {
namespace tent {
namespace {

class FakeSubBatch : public Transport::SubBatch {
   public:
    size_t size() const override { return task_count; }

    size_t task_count = 0;
    std::vector<Request> requests;
    std::vector<TransferStatus> statuses;
    std::vector<int> poll_counts;
};

class FakeTransport : public Transport {
   public:
    using PollStatusFactory =
        std::function<TransferStatus(const Request&, int)>;

    explicit FakeTransport(TransportType self_type,
                           PollStatusFactory poll_status_factory = {},
                           bool notify_on_submit = false,
                           bool fail_on_submit = false)
        : self_type_(self_type),
          poll_status_factory_(std::move(poll_status_factory)),
          notify_on_submit_(notify_on_submit),
          fail_on_submit_(fail_on_submit) {
        caps.dram_to_dram = true;
    }

    std::atomic<int> submit_calls{0};
    std::atomic<int> status_calls{0};
    std::atomic<size_t> max_requests_per_submit{0};
    std::atomic<int> first_submitted_opcode{-1};
    std::atomic<size_t> runtime_read_limit{
        std::numeric_limits<size_t>::max()};
    std::atomic<size_t> runtime_write_limit{
        std::numeric_limits<size_t>::max()};
    std::atomic<size_t> runtime_queued_reads{0};
    std::atomic<size_t> runtime_queued_writes{0};

    Status install(std::string&, std::shared_ptr<ControlService>,
                   std::shared_ptr<Topology>,
                   std::shared_ptr<Config> = nullptr) override {
        return Status::OK();
    }

    Status allocateSubBatch(SubBatchRef& batch, size_t) override {
        batch = new FakeSubBatch();
        return Status::OK();
    }

    Status freeSubBatch(SubBatchRef& batch) override {
        delete static_cast<FakeSubBatch*>(batch);
        batch = nullptr;
        return Status::OK();
    }

    Status submitTransferTasks(SubBatchRef batch,
                               const std::vector<Request>& requests) override {
        ++submit_calls;
        size_t previous_max = max_requests_per_submit.load();
        while (previous_max < requests.size() &&
               !max_requests_per_submit.compare_exchange_weak(
                   previous_max, requests.size())) {
        }
        if (!requests.empty()) {
            int no_opcode = -1;
            first_submitted_opcode.compare_exchange_strong(
                no_opcode, static_cast<int>(requests.front().opcode));
        }
        if (fail_on_submit_) {
            return Status::InternalError("injected submit failure" LOC_MARK);
        }
        auto* fake = static_cast<FakeSubBatch*>(batch);
        for (const auto& request : requests) {
            fake->requests.push_back(request);
            fake->statuses.push_back(
                {TransferStatusEnum::COMPLETED, request.length});
            fake->poll_counts.push_back(0);
            ++fake->task_count;
        }
        if (notify_on_submit_) batch->notifyProgress();
        return Status::OK();
    }

    Status getTransferStatus(SubBatchRef batch, int task_id,
                             TransferStatus& status) override {
        ++status_calls;
        auto* fake = static_cast<FakeSubBatch*>(batch);
        if (task_id < 0 || task_id >= (int)fake->statuses.size()) {
            return Status::InvalidArgument("bad task_id" LOC_MARK);
        }
        ++fake->poll_counts[task_id];
        if (poll_status_factory_) {
            status = poll_status_factory_(fake->requests[task_id],
                                          fake->poll_counts[task_id]);
        } else {
            status = fake->statuses[task_id];
        }
        return Status::OK();
    }

    Status addMemoryBuffer(BufferDesc& desc, const MemoryOptions&) override {
        desc.transports.push_back(self_type_);
        return Status::OK();
    }

    Status addMemoryBuffer(std::vector<BufferDesc>& desc_list,
                           const MemoryOptions& options) override {
        for (auto& desc : desc_list) {
            auto status = addMemoryBuffer(desc, options);
            if (!status.ok()) return status;
        }
        return Status::OK();
    }

    Status removeMemoryBuffer(BufferDesc&) override { return Status::OK(); }

    Status allocateLocalMemory(void** addr, size_t size,
                               MemoryOptions&) override {
        *addr = std::malloc(size);
        return *addr ? Status::OK()
                     : Status::InternalError("malloc failed" LOC_MARK);
    }

    Status freeLocalMemory(void* addr, size_t) override {
        std::free(addr);
        return Status::OK();
    }

    bool warmupMemory(void*, size_t) override { return false; }

    size_t runtimeQueueDispatchLimit(Request::OpCode opcode) const override {
        return opcode == Request::READ ? runtime_read_limit.load()
                                       : runtime_write_limit.load();
    }

    void updateRuntimeQueueDepth(size_t queued_reads,
                                 size_t queued_writes) override {
        runtime_queued_reads.store(queued_reads);
        runtime_queued_writes.store(queued_writes);
    }

    const char* getName() const override { return "<fake-rdma>"; }

   private:
    TransportType self_type_;
    PollStatusFactory poll_status_factory_;
    bool notify_on_submit_;
    bool fail_on_submit_;
};

std::shared_ptr<Config> makeRuntimeQueueConfig(size_t max_dispatch_owners,
                                               size_t max_dispatch_bytes,
                                               bool merge_requests = false) {
    auto cfg = std::make_shared<Config>();
    cfg->set("metadata_type", "p2p");
    cfg->set("metadata_servers", "");
    cfg->set("rpc_server_hostname", "127.0.0.1");
    cfg->set("rpc_server_port", "0");
    cfg->set("log_level", "warning");
    cfg->set("merge_requests", merge_requests);
    cfg->set("enable_runtime_queue", true);
    cfg->set("runtime_queue/max_outstanding_owners", 16UL);
    cfg->set("runtime_queue/max_outstanding_bytes", 1UL << 20);
    cfg->set("runtime_queue/max_dispatch_owners", max_dispatch_owners);
    cfg->set("runtime_queue/max_dispatch_bytes", max_dispatch_bytes);
    cfg->set("runtime_queue/staging_owner_reserve", 0UL);
    cfg->set("runtime_queue/staging_byte_reserve", 0UL);
    cfg->set("runtime_queue/progress_fallback_interval_us", 50000UL);

    cfg->set("transports/tcp/enable", false);
    cfg->set("transports/shm/enable", false);
    cfg->set("transports/rdma/enable", false);
    cfg->set("transports/io_uring/enable", false);
    cfg->set("transports/nvlink/enable", false);
    cfg->set("transports/mnnvl/enable", false);
    cfg->set("transports/gds/enable", false);
    cfg->set("transports/ascend_direct/enable", false);
    return cfg;
}

void installFakeRdma(TransferEngineImpl& engine,
                     const std::shared_ptr<FakeTransport>& fake_rdma) {
    std::string seg_name = engine.getSegmentName();
    ASSERT_TRUE(fake_rdma->install(seg_name, nullptr, nullptr).ok());
    engine.swapTransportForTest(RDMA, fake_rdma);
}

void installFakeGds(TransferEngineImpl& engine,
                    const std::shared_ptr<FakeTransport>& fake_gds) {
    std::string seg_name = engine.getSegmentName();
    ASSERT_TRUE(fake_gds->install(seg_name, nullptr, nullptr).ok());
    engine.swapTransportForTest(GDS, fake_gds);
}

Request makeLocalWrite(uint8_t* ptr, size_t length) {
    Request request;
    request.opcode = Request::WRITE;
    request.source = ptr;
    request.target_id = LOCAL_SEGMENT_ID;
    request.target_offset = reinterpret_cast<uint64_t>(ptr);
    request.length = length;
    request.transport_hint = RDMA;
    return request;
}

Request makeLocalGdsWrite(uint8_t* ptr, size_t length) {
    auto request = makeLocalWrite(ptr, length);
    request.transport_hint = GDS;
    return request;
}

Request makeLocalGdsRead(uint8_t* ptr, size_t length) {
    auto request = makeLocalGdsWrite(ptr, length);
    request.opcode = Request::READ;
    return request;
}

TEST(RuntimeQueueDispatch, RejectsOverfullBatchBeforePublishingTasks) {
    auto cfg = makeRuntimeQueueConfig(1, 1UL << 20);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    auto fake_rdma = std::make_shared<FakeTransport>(RDMA);
    installFakeRdma(engine, fake_rdma);

    constexpr size_t kReqLen = 4096;
    std::vector<uint8_t> buffer(kReqLen * 2, 0x11);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());

    BatchID batch = engine.allocateBatch(1);
    ASSERT_NE(batch, (BatchID)0);

    auto status = engine.submitTransfer(
        batch, {makeLocalWrite(buffer.data(), kReqLen),
                makeLocalWrite(buffer.data() + kReqLen, kReqLen)});
    EXPECT_TRUE(status.IsTooManyRequests()) << status.ToString();
    EXPECT_EQ(fake_rdma->submit_calls.load(), 0);

    TransferStatus task_status{};
    EXPECT_TRUE(
        engine.getTransferStatus(batch, 0, task_status).IsInvalidArgument());

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(
        engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, StreamsSubmitAboveOutstandingOwnerLimit) {
    constexpr size_t kOutstandingOwners = 16;
    constexpr size_t kReqLen = 4096;
    auto cfg = makeRuntimeQueueConfig(kOutstandingOwners, 1UL << 20);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    std::atomic<bool> complete{false};
    auto fake_rdma = std::make_shared<FakeTransport>(
        RDMA, [&complete](const Request& request, int) {
            if (!complete.load()) {
                return TransferStatus{TransferStatusEnum::PENDING, 0};
            }
            return TransferStatus{TransferStatusEnum::COMPLETED,
                                  request.length};
        });
    installFakeRdma(engine, fake_rdma);

    std::vector<uint8_t> buffer(kReqLen * (kOutstandingOwners + 1), 0x12);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());
    BatchID batch = engine.allocateBatch(kOutstandingOwners + 1);
    ASSERT_NE(batch, (BatchID)0);

    std::vector<Request> requests;
    for (size_t index = 0; index <= kOutstandingOwners; ++index) {
        requests.push_back(
            makeLocalWrite(buffer.data() + index * kReqLen, kReqLen));
    }
    auto status = engine.submitTransfer(batch, requests);
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(fake_rdma->submit_calls.load(), 1);

    TransferStatus task_status{};
    ASSERT_TRUE(engine
                    .getTransferStatus(batch, kOutstandingOwners, task_status)
                    .ok());
    EXPECT_EQ(task_status.s, TransferStatusEnum::PENDING);

    complete.store(true);
    ASSERT_TRUE(engine.getTransferStatus(batch, 0, task_status).ok());
    EXPECT_EQ(task_status.s, TransferStatusEnum::COMPLETED);
    EXPECT_EQ(fake_rdma->submit_calls.load(), 2);
    ASSERT_TRUE(engine
                    .getTransferStatus(batch, kOutstandingOwners, task_status)
                    .ok());
    EXPECT_EQ(task_status.s, TransferStatusEnum::COMPLETED);

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(
        engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, StreamsSubmitAboveOutstandingByteLimit) {
    constexpr size_t kReqLen = 4096;
    constexpr size_t kOutstandingBytes = 2 * kReqLen;
    auto cfg = makeRuntimeQueueConfig(2, kOutstandingBytes);
    cfg->set("runtime_queue/max_outstanding_bytes", kOutstandingBytes);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    std::atomic<bool> complete{false};
    auto fake_rdma = std::make_shared<FakeTransport>(
        RDMA, [&complete](const Request& request, int) {
            if (!complete.load()) {
                return TransferStatus{TransferStatusEnum::PENDING, 0};
            }
            return TransferStatus{TransferStatusEnum::COMPLETED,
                                  request.length};
        });
    installFakeRdma(engine, fake_rdma);

    std::vector<uint8_t> buffer(3 * kReqLen, 0x13);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());
    BatchID batch = engine.allocateBatch(3);
    ASSERT_NE(batch, (BatchID)0);

    std::vector<Request> requests;
    for (size_t index = 0; index < 3; ++index) {
        requests.push_back(
            makeLocalWrite(buffer.data() + index * kReqLen, kReqLen));
    }
    auto status = engine.submitTransfer(batch, requests);
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(fake_rdma->submit_calls.load(), 1);

    TransferStatus task_status{};
    ASSERT_TRUE(engine.getTransferStatus(batch, 2, task_status).ok());
    EXPECT_EQ(task_status.s, TransferStatusEnum::PENDING);

    complete.store(true);
    ASSERT_TRUE(engine.getTransferStatus(batch, 0, task_status).ok());
    EXPECT_EQ(task_status.s, TransferStatusEnum::COMPLETED);
    EXPECT_EQ(fake_rdma->submit_calls.load(), 2);
    ASSERT_TRUE(engine.getTransferStatus(batch, 2, task_status).ok());
    EXPECT_EQ(task_status.s, TransferStatusEnum::COMPLETED);

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(
        engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, RejectsOwnerLargerThanDispatchByteWindow) {
    constexpr size_t kReqLen = 4096;
    auto cfg = makeRuntimeQueueConfig(1, kReqLen - 1);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    auto fake_rdma = std::make_shared<FakeTransport>(RDMA);
    installFakeRdma(engine, fake_rdma);

    std::vector<uint8_t> buffer(kReqLen, 0x77);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());

    BatchID batch = engine.allocateBatch(1);
    ASSERT_NE(batch, (BatchID)0);

    auto status =
        engine.submitTransfer(batch, {makeLocalWrite(buffer.data(), kReqLen)});
    EXPECT_TRUE(status.IsTooManyRequests()) << status.ToString();
    EXPECT_EQ(fake_rdma->submit_calls.load(), 0);

    TransferStatus task_status{};
    EXPECT_TRUE(
        engine.getTransferStatus(batch, 0, task_status).IsInvalidArgument());

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(
        engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, RejectsEmptyDispatchWindowConfig) {
    auto cfg = makeRuntimeQueueConfig(0, 1UL << 20);
    TransferEngineImpl engine(cfg);

    EXPECT_FALSE(engine.available());

    cfg = makeRuntimeQueueConfig(1, 0);
    TransferEngineImpl byte_window_engine(cfg);

    EXPECT_FALSE(byte_window_engine.available());
}

TEST(RuntimeQueueDispatch, RejectsDispatchWindowAboveOutstandingLimits) {
    auto cfg = makeRuntimeQueueConfig(2, 1UL << 20);
    cfg->set("runtime_queue/max_outstanding_owners", 1UL);
    TransferEngineImpl owner_window_engine(cfg);
    EXPECT_FALSE(owner_window_engine.available());

    cfg = makeRuntimeQueueConfig(1, 1UL << 20);
    cfg->set("runtime_queue/max_outstanding_bytes", 512UL << 10);
    TransferEngineImpl byte_window_engine(cfg);
    EXPECT_FALSE(byte_window_engine.available());

    cfg = makeRuntimeQueueConfig(4, 1UL << 20);
    cfg->set("runtime_queue/max_dispatch_read_owners", 5UL);
    TransferEngineImpl direction_window_engine(cfg);
    EXPECT_FALSE(direction_window_engine.available());
}

TEST(RuntimeQueueDispatch, ImplicitDirectionWindowsFollowGlobalWindow) {
    auto cfg = makeRuntimeQueueConfig(8, 1UL << 20);
    EXPECT_FALSE(cfg->contains("runtime_queue/max_dispatch_read_owners"));
    EXPECT_FALSE(cfg->contains("runtime_queue/max_dispatch_write_owners"));

    TransferEngineImpl engine(cfg);
    EXPECT_TRUE(engine.available());
}

TEST(RuntimeQueueDispatch, DispatchesOnlyOneWindowOnSubmit) {
    auto cfg = makeRuntimeQueueConfig(1, 1UL << 20);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    auto fake_rdma = std::make_shared<FakeTransport>(RDMA);
    installFakeRdma(engine, fake_rdma);

    constexpr size_t kReqLen = 4096;
    std::vector<uint8_t> buffer(kReqLen * 2, 0x22);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());

    BatchID batch = engine.allocateBatch(2);
    ASSERT_NE(batch, (BatchID)0);

    ASSERT_TRUE(
        engine
            .submitTransfer(batch,
                            {makeLocalWrite(buffer.data(), kReqLen),
                             makeLocalWrite(buffer.data() + kReqLen, kReqLen)})
            .ok());
    EXPECT_EQ(fake_rdma->submit_calls.load(), 1);

    TransferStatus first{};
    ASSERT_TRUE(engine.getTransferStatus(batch, 0, first).ok());
    EXPECT_EQ(first.s, TransferStatusEnum::COMPLETED);
    EXPECT_EQ(fake_rdma->submit_calls.load(), 2);

    TransferStatus second{};
    ASSERT_TRUE(engine.getTransferStatus(batch, 1, second).ok());
    EXPECT_EQ(second.s, TransferStatusEnum::COMPLETED);

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(
        engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, PreservesGroupedSubmitForNonGdsTransport) {
    constexpr size_t kRequestCount = 4;
    constexpr size_t kReqLen = 4096;
    auto cfg = makeRuntimeQueueConfig(kRequestCount, 1UL << 20);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    auto fake_rdma = std::make_shared<FakeTransport>(RDMA);
    installFakeRdma(engine, fake_rdma);

    std::vector<uint8_t> buffer(kReqLen * kRequestCount, 0x44);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());
    BatchID batch = engine.allocateBatch(kRequestCount);
    ASSERT_NE(batch, (BatchID)0);

    std::vector<Request> requests;
    for (size_t index = 0; index < kRequestCount; ++index) {
        requests.push_back(
            makeLocalWrite(buffer.data() + index * kReqLen, kReqLen));
    }
    ASSERT_TRUE(engine.submitTransfer(batch, requests).ok());
    EXPECT_EQ(fake_rdma->submit_calls.load(), 1);
    EXPECT_EQ(fake_rdma->max_requests_per_submit.load(), kRequestCount);

    for (size_t index = 0; index < kRequestCount; ++index) {
        TransferStatus status{};
        ASSERT_TRUE(engine.getTransferStatus(batch, index, status).ok());
        EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);
        EXPECT_EQ(status.transferred_bytes, kReqLen);
    }

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, BulkPicksButSubmitsEachGdsOwnerIndividually) {
    constexpr size_t kRequestCount = 4;
    constexpr size_t kReqLen = 4096;
    auto cfg = makeRuntimeQueueConfig(kRequestCount, 1UL << 20);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    auto fake_gds = std::make_shared<FakeTransport>(GDS);
    installFakeGds(engine, fake_gds);

    std::vector<uint8_t> buffer(kReqLen * kRequestCount, 0x47);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());
    BatchID batch = engine.allocateBatch(kRequestCount);
    ASSERT_NE(batch, (BatchID)0);

    std::vector<Request> requests;
    for (size_t index = 0; index < kRequestCount; ++index) {
        requests.push_back(
            makeLocalGdsWrite(buffer.data() + index * kReqLen, kReqLen));
    }
    ASSERT_TRUE(engine.submitTransfer(batch, requests).ok());
    EXPECT_EQ(fake_gds->submit_calls.load(), kRequestCount);
    EXPECT_EQ(fake_gds->max_requests_per_submit.load(), 1u);

    for (size_t index = 0; index < kRequestCount; ++index) {
        TransferStatus status{};
        ASSERT_TRUE(engine.getTransferStatus(batch, index, status).ok());
        EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);
        EXPECT_EQ(status.transferred_bytes, kReqLen);
    }

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, FailedGdsSubmitDoesNotStrandBulkPickedOwners) {
    constexpr size_t kRequestCount = 4;
    constexpr size_t kReqLen = 4096;
    auto cfg = makeRuntimeQueueConfig(kRequestCount, 1UL << 20);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    auto fake_gds = std::make_shared<FakeTransport>(
        GDS, FakeTransport::PollStatusFactory{}, false, true);
    installFakeGds(engine, fake_gds);

    std::vector<uint8_t> buffer(kReqLen * kRequestCount, 0x48);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());
    BatchID batch = engine.allocateBatch(kRequestCount);
    ASSERT_NE(batch, (BatchID)0);

    std::vector<Request> requests;
    for (size_t index = 0; index < kRequestCount; ++index) {
        requests.push_back(
            makeLocalGdsWrite(buffer.data() + index * kReqLen, kReqLen));
    }
    ASSERT_TRUE(engine.submitTransfer(batch, requests).ok());
    EXPECT_EQ(fake_gds->submit_calls.load(), kRequestCount);

    for (size_t index = 0; index < kRequestCount; ++index) {
        TransferStatus status{};
        ASSERT_TRUE(engine.getTransferStatus(batch, index, status).ok());
        EXPECT_EQ(status.s, TransferStatusEnum::FAILED);
    }

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, GdsWriteBurstRespectsDirectionWindow) {
    constexpr size_t kRequestCount = 8;
    constexpr size_t kWriteWindow = 4;
    constexpr size_t kReqLen = 4096;
    auto cfg = makeRuntimeQueueConfig(16, 1UL << 20);
    cfg->set("runtime_queue/max_dispatch_read_owners", 16UL);
    cfg->set("runtime_queue/max_dispatch_write_owners", kWriteWindow);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    std::atomic<bool> complete_writes{false};
    auto fake_gds = std::make_shared<FakeTransport>(
        GDS, [&complete_writes](const Request& request, int) {
            if (!complete_writes.load()) {
                return TransferStatus{TransferStatusEnum::PENDING, 0};
            }
            return TransferStatus{TransferStatusEnum::COMPLETED,
                                  request.length};
        });
    installFakeGds(engine, fake_gds);

    std::vector<uint8_t> buffer(kReqLen * kRequestCount, 0x49);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());
    BatchID batch = engine.allocateBatch(kRequestCount);
    ASSERT_NE(batch, (BatchID)0);

    std::vector<Request> requests;
    for (size_t index = 0; index < kRequestCount; ++index) {
        requests.push_back(
            makeLocalGdsWrite(buffer.data() + index * kReqLen, kReqLen));
    }
    ASSERT_TRUE(engine.submitTransfer(batch, requests).ok());
    EXPECT_EQ(fake_gds->submit_calls.load(), kWriteWindow);
    EXPECT_EQ(fake_gds->max_requests_per_submit.load(), 1u);

    complete_writes.store(true);
    for (size_t index = 0; index < kRequestCount; ++index) {
        TransferStatus status{};
        ASSERT_TRUE(engine.getTransferStatus(batch, index, status).ok());
        EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);
    }
    EXPECT_EQ(fake_gds->submit_calls.load(), kRequestCount);

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, StreamsLargeGdsSubmitThroughAdmissionWindow) {
    constexpr size_t kRequestCount = 20;
    constexpr size_t kWriteWindow = 4;
    constexpr size_t kReqLen = 4096;
    auto cfg = makeRuntimeQueueConfig(kWriteWindow, 1UL << 20);
    cfg->set("runtime_queue/max_dispatch_read_owners", kWriteWindow);
    cfg->set("runtime_queue/max_dispatch_write_owners", kWriteWindow);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    std::atomic<bool> complete{false};
    auto fake_gds = std::make_shared<FakeTransport>(
        GDS, [&complete](const Request& request, int) {
            if (!complete.load()) {
                return TransferStatus{TransferStatusEnum::PENDING, 0};
            }
            return TransferStatus{TransferStatusEnum::COMPLETED,
                                  request.length};
        });
    installFakeGds(engine, fake_gds);

    std::vector<uint8_t> buffer(kReqLen * kRequestCount, 0x4c);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());
    BatchID batch = engine.allocateBatch(kRequestCount);
    ASSERT_NE(batch, (BatchID)0);

    std::vector<Request> requests;
    for (size_t index = 0; index < kRequestCount; ++index) {
        requests.push_back(
            makeLocalGdsWrite(buffer.data() + index * kReqLen, kReqLen));
    }
    ASSERT_TRUE(engine.submitTransfer(batch, requests).ok());
    EXPECT_EQ(fake_gds->submit_calls.load(), kWriteWindow);
    EXPECT_EQ(fake_gds->max_requests_per_submit.load(), 1u);

    TransferStatus status{};
    ASSERT_TRUE(
        engine.getTransferStatus(batch, kRequestCount - 1, status).ok());
    EXPECT_EQ(status.s, TransferStatusEnum::PENDING);

    complete.store(true);
    for (size_t index = 0; index < kRequestCount; ++index) {
        ASSERT_TRUE(engine.getTransferStatus(batch, index, status).ok());
        EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);
    }
    EXPECT_EQ(fake_gds->submit_calls.load(), kRequestCount);

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(
        engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, GdsAdaptiveLimitClampsRuntimeReadWindow) {
    constexpr size_t kRequestCount = 8;
    constexpr size_t kInitialLimit = 4;
    constexpr size_t kReducedLimit = 2;
    constexpr size_t kReqLen = 4096;
    auto cfg = makeRuntimeQueueConfig(kRequestCount, 1UL << 20);
    cfg->set("runtime_queue/max_dispatch_read_owners", kRequestCount);
    // Keep the event worker from racing this deterministic manual-progress
    // test; production still uses the configured fallback interval.
    cfg->set("runtime_queue/progress_fallback_interval_us", 60000000UL);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    std::atomic<bool> complete_reads{false};
    auto fake_gds = std::make_shared<FakeTransport>(
        GDS, [&complete_reads](const Request& request, int) {
            if (!complete_reads.load()) {
                return TransferStatus{TransferStatusEnum::PENDING, 0};
            }
            return TransferStatus{TransferStatusEnum::COMPLETED,
                                  request.length};
        });
    fake_gds->runtime_read_limit.store(kInitialLimit);
    installFakeGds(engine, fake_gds);

    std::vector<uint8_t> buffer(kReqLen * kRequestCount, 0x4a);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());
    BatchID batch = engine.allocateBatch(kRequestCount);
    ASSERT_NE(batch, (BatchID)0);

    std::vector<Request> requests;
    for (size_t index = 0; index < kRequestCount; ++index) {
        requests.push_back(
            makeLocalGdsRead(buffer.data() + index * kReqLen, kReqLen));
    }
    ASSERT_TRUE(engine.submitTransfer(batch, requests).ok());
    EXPECT_EQ(fake_gds->submit_calls.load(), kInitialLimit);

    // Model a P99-triggered transport reduction while the original window is
    // still full. No new owner may enter until the live count drains below
    // the reduced limit.
    fake_gds->runtime_read_limit.store(kReducedLimit);
    complete_reads.store(true);
    for (size_t index = 0; index < kInitialLimit; ++index) {
        TransferStatus status{};
        ASSERT_TRUE(engine.getTransferStatus(batch, index, status).ok());
        EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);
    }
    EXPECT_EQ(fake_gds->submit_calls.load(),
              kInitialLimit + kReducedLimit);

    for (size_t index = kInitialLimit; index < kRequestCount; ++index) {
        TransferStatus status{};
        ASSERT_TRUE(engine.getTransferStatus(batch, index, status).ok());
        EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);
    }
    EXPECT_EQ(fake_gds->submit_calls.load(), kRequestCount);

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, PublishesExternalGdsBacklogByDirection) {
    constexpr size_t kReqLen = 4096;
    auto cfg = makeRuntimeQueueConfig(1, 1UL << 20);
    cfg->set("runtime_queue/progress_fallback_interval_us", 60000000UL);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    std::atomic<bool> complete{false};
    auto fake_gds = std::make_shared<FakeTransport>(
        GDS, [&complete](const Request& request, int) {
            if (!complete.load()) {
                return TransferStatus{TransferStatusEnum::PENDING, 0};
            }
            return TransferStatus{TransferStatusEnum::COMPLETED,
                                  request.length};
        });
    installFakeGds(engine, fake_gds);

    std::vector<uint8_t> buffer(kReqLen * 3, 0x4b);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());
    BatchID batch = engine.allocateBatch(3);
    ASSERT_NE(batch, (BatchID)0);

    ASSERT_TRUE(
        engine
            .submitTransfer(batch,
                            {makeLocalGdsWrite(buffer.data(), kReqLen),
                             makeLocalGdsRead(buffer.data() + kReqLen,
                                              kReqLen),
                             makeLocalGdsRead(buffer.data() + 2 * kReqLen,
                                              kReqLen)})
            .ok());
    EXPECT_EQ(fake_gds->first_submitted_opcode.load(),
              static_cast<int>(Request::READ));
    EXPECT_EQ(fake_gds->runtime_queued_reads.load(), 1u);
    EXPECT_EQ(fake_gds->runtime_queued_writes.load(), 1u);

    complete.store(true);
    TransferStatus first_read{};
    ASSERT_TRUE(engine.getTransferStatus(batch, 1, first_read).ok());
    EXPECT_EQ(first_read.s, TransferStatusEnum::COMPLETED);
    EXPECT_EQ(fake_gds->runtime_queued_reads.load(), 0u);
    EXPECT_EQ(fake_gds->runtime_queued_writes.load(), 1u);

    TransferStatus second_read{};
    ASSERT_TRUE(engine.getTransferStatus(batch, 2, second_read).ok());
    EXPECT_EQ(second_read.s, TransferStatusEnum::COMPLETED);
    EXPECT_EQ(fake_gds->runtime_queued_reads.load(), 0u);
    EXPECT_EQ(fake_gds->runtime_queued_writes.load(), 0u);

    TransferStatus write{};
    ASSERT_TRUE(engine.getTransferStatus(batch, 0, write).ok());
    EXPECT_EQ(write.s, TransferStatusEnum::COMPLETED);

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, PrioritizesReadOwnersWithinBulkPick) {
    constexpr size_t kReqLen = 4096;
    auto cfg = makeRuntimeQueueConfig(1, 1UL << 20);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    auto fake_gds = std::make_shared<FakeTransport>(GDS);
    installFakeGds(engine, fake_gds);

    std::vector<uint8_t> buffer(kReqLen * 2, 0x45);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());
    BatchID batch = engine.allocateBatch(2);
    ASSERT_NE(batch, (BatchID)0);

    ASSERT_TRUE(
        engine
            .submitTransfer(batch,
                            {makeLocalGdsWrite(buffer.data(), kReqLen),
                             makeLocalGdsRead(buffer.data() + kReqLen,
                                              kReqLen)})
            .ok());
    EXPECT_EQ(fake_gds->first_submitted_opcode.load(),
              static_cast<int>(Request::READ));

    TransferStatus read_status{};
    ASSERT_TRUE(engine.getTransferStatus(batch, 1, read_status).ok());
    EXPECT_EQ(read_status.s, TransferStatusEnum::COMPLETED);
    TransferStatus write_status{};
    ASSERT_TRUE(engine.getTransferStatus(batch, 0, write_status).ok());
    EXPECT_EQ(write_status.s, TransferStatusEnum::COMPLETED);

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, EnforcesDispatchByteWindowDuringBulkPick) {
    constexpr size_t kReqLen = 4096;
    auto cfg = makeRuntimeQueueConfig(2, kReqLen);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    std::atomic<bool> complete_first{false};
    auto fake_rdma = std::make_shared<FakeTransport>(
        RDMA, [&complete_first](const Request& request, int) {
            if (!complete_first.load()) {
                return TransferStatus{TransferStatusEnum::PENDING, 0};
            }
            return TransferStatus{TransferStatusEnum::COMPLETED,
                                  request.length};
        });
    installFakeRdma(engine, fake_rdma);

    std::vector<uint8_t> buffer(kReqLen * 2, 0x46);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());
    BatchID batch = engine.allocateBatch(2);
    ASSERT_NE(batch, (BatchID)0);

    ASSERT_TRUE(
        engine
            .submitTransfer(batch,
                            {makeLocalWrite(buffer.data(), kReqLen),
                             makeLocalWrite(buffer.data() + kReqLen, kReqLen)})
            .ok());
    EXPECT_EQ(fake_rdma->submit_calls.load(), 1);
    EXPECT_EQ(fake_rdma->max_requests_per_submit.load(), 1u);

    TransferStatus first{};
    ASSERT_TRUE(engine.getTransferStatus(batch, 0, first).ok());
    EXPECT_EQ(first.s, TransferStatusEnum::PENDING);
    EXPECT_EQ(fake_rdma->submit_calls.load(), 1);

    complete_first.store(true);
    ASSERT_TRUE(engine.getTransferStatus(batch, 0, first).ok());
    EXPECT_EQ(first.s, TransferStatusEnum::COMPLETED);
    EXPECT_EQ(fake_rdma->submit_calls.load(), 2);

    TransferStatus second{};
    ASSERT_TRUE(engine.getTransferStatus(batch, 1, second).ok());
    EXPECT_EQ(second.s, TransferStatusEnum::COMPLETED);

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, KeepsDispatchWindowUntilOwnerIsTerminal) {
    auto cfg = makeRuntimeQueueConfig(1, 1UL << 20);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    std::atomic<bool> complete_first{false};
    auto fake_rdma = std::make_shared<FakeTransport>(
        RDMA, [&complete_first](const Request& request, int) {
            if (!complete_first.load()) {
                return TransferStatus{TransferStatusEnum::PENDING, 0};
            }
            return TransferStatus{TransferStatusEnum::COMPLETED,
                                  request.length};
        });
    installFakeRdma(engine, fake_rdma);

    constexpr size_t kReqLen = 4096;
    std::vector<uint8_t> buffer(kReqLen * 2, 0x55);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());

    BatchID batch = engine.allocateBatch(2);
    ASSERT_NE(batch, (BatchID)0);

    ASSERT_TRUE(
        engine
            .submitTransfer(batch,
                            {makeLocalWrite(buffer.data(), kReqLen),
                             makeLocalWrite(buffer.data() + kReqLen, kReqLen)})
            .ok());
    EXPECT_EQ(fake_rdma->submit_calls.load(), 1);

    TransferStatus first{};
    ASSERT_TRUE(engine.getTransferStatus(batch, 0, first).ok());
    EXPECT_EQ(first.s, TransferStatusEnum::PENDING);
    EXPECT_EQ(fake_rdma->submit_calls.load(), 1);

    TransferStatus second{};
    ASSERT_TRUE(engine.getTransferStatus(batch, 1, second).ok());
    EXPECT_EQ(second.s, TransferStatusEnum::PENDING);
    EXPECT_EQ(fake_rdma->submit_calls.load(), 1);

    complete_first.store(true);
    ASSERT_TRUE(engine.getTransferStatus(batch, 0, first).ok());
    EXPECT_EQ(first.s, TransferStatusEnum::COMPLETED);
    EXPECT_EQ(fake_rdma->submit_calls.load(), 2);

    ASSERT_TRUE(engine.getTransferStatus(batch, 1, second).ok());
    EXPECT_EQ(second.s, TransferStatusEnum::COMPLETED);

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(
        engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, ProgressWorkerRefillsWindowFromTransportNotify) {
    auto cfg = makeRuntimeQueueConfig(1, 1UL << 20);
    cfg->set("enable_progress_worker", true);
    cfg->set("runtime_queue/progress_fallback_interval_us", 0UL);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    auto fake_rdma = std::make_shared<FakeTransport>(
        RDMA, FakeTransport::PollStatusFactory{}, true);
    installFakeRdma(engine, fake_rdma);

    constexpr size_t kReqLen = 4096;
    std::vector<uint8_t> buffer(kReqLen * 2, 0x66);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());

    BatchID batch = engine.allocateBatch(2);
    ASSERT_NE(batch, (BatchID)0);

    ASSERT_TRUE(
        engine
            .submitTransfer(batch,
                            {makeLocalWrite(buffer.data(), kReqLen),
                             makeLocalWrite(buffer.data() + kReqLen, kReqLen)})
            .ok());
    EXPECT_EQ(fake_rdma->submit_calls.load(), 1);

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
    while (std::chrono::steady_clock::now() < deadline &&
           fake_rdma->submit_calls.load() < 2) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_EQ(fake_rdma->submit_calls.load(), 2);

    TransferStatus status{};
    ASSERT_TRUE(engine.getTransferStatus(batch, status).ok());
    EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(
        engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, RuntimeQueueDrainsWithoutUserPolling) {
    auto cfg = makeRuntimeQueueConfig(1, 1UL << 20);
    cfg->set("runtime_queue/progress_fallback_interval_us", 1000UL);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    auto fake_rdma = std::make_shared<FakeTransport>(
        RDMA, [](const Request& request, int poll_count) {
            if (poll_count == 1) {
                return TransferStatus{TransferStatusEnum::PENDING, 0};
            }
            return TransferStatus{TransferStatusEnum::COMPLETED,
                                  request.length};
        });
    installFakeRdma(engine, fake_rdma);

    constexpr size_t kReqLen = 4096;
    std::vector<uint8_t> buffer(kReqLen * 2, 0x88);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());

    BatchID batch = engine.allocateBatch(2);
    ASSERT_NE(batch, (BatchID)0);

    ASSERT_TRUE(
        engine
            .submitTransfer(batch,
                            {makeLocalWrite(buffer.data(), kReqLen),
                             makeLocalWrite(buffer.data() + kReqLen, kReqLen)})
            .ok());
    EXPECT_EQ(fake_rdma->submit_calls.load(), 1);

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
    while (std::chrono::steady_clock::now() < deadline &&
           (fake_rdma->submit_calls.load() < 2 ||
            fake_rdma->status_calls.load() < 4)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_EQ(fake_rdma->submit_calls.load(), 2);
    EXPECT_GE(fake_rdma->status_calls.load(), 4);

    TransferStatus status{};
    ASSERT_TRUE(engine.getTransferStatus(batch, status).ok());
    EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(
        engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, EarlyFreeReclaimsAfterQueuedCompletion) {
    auto cfg = makeRuntimeQueueConfig(1, 1UL << 20);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    auto fake_rdma = std::make_shared<FakeTransport>(
        RDMA, [](const Request& request, int poll_count) {
            if (poll_count == 1) {
                return TransferStatus{TransferStatusEnum::PENDING, 0};
            }
            return TransferStatus{TransferStatusEnum::COMPLETED,
                                  request.length};
        });
    installFakeRdma(engine, fake_rdma);

    constexpr size_t kReqLen = 4096;
    std::vector<uint8_t> buffer(kReqLen, 0x33);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());

    BatchID batch = engine.allocateBatch(1);
    ASSERT_NE(batch, (BatchID)0);
    ASSERT_TRUE(
        engine.submitTransfer(batch, {makeLocalWrite(buffer.data(), kReqLen)})
            .ok());

    ASSERT_TRUE(engine.freeBatch(batch).ok());

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
    while (std::chrono::steady_clock::now() < deadline &&
           fake_rdma->status_calls.load() < 2) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_GE(fake_rdma->status_calls.load(), 2);

    TransferStatus status{};
    EXPECT_TRUE(engine.getTransferStatus(batch, 0, status).IsInvalidArgument());
    EXPECT_TRUE(
        engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

TEST(RuntimeQueueDispatch, PollingDerivedTaskCompletesMergedOwner) {
    auto cfg = makeRuntimeQueueConfig(1, 1UL << 20, true);
    TransferEngineImpl engine(cfg);
    ASSERT_TRUE(engine.available());

    auto fake_rdma = std::make_shared<FakeTransport>(RDMA);
    installFakeRdma(engine, fake_rdma);

    constexpr size_t kReqLen = 4096;
    std::vector<uint8_t> buffer(kReqLen * 2, 0x44);
    ASSERT_TRUE(engine.registerLocalMemory(buffer.data(), buffer.size()).ok());

    BatchID batch = engine.allocateBatch(2);
    ASSERT_NE(batch, (BatchID)0);

    ASSERT_TRUE(
        engine
            .submitTransfer(batch,
                            {makeLocalWrite(buffer.data(), kReqLen),
                             makeLocalWrite(buffer.data() + kReqLen, kReqLen)})
            .ok());
    EXPECT_EQ(fake_rdma->submit_calls.load(), 1);

    TransferStatus derived{};
    ASSERT_TRUE(engine.getTransferStatus(batch, 1, derived).ok());
    EXPECT_EQ(derived.s, TransferStatusEnum::COMPLETED);

    TransferStatus owner{};
    ASSERT_TRUE(engine.getTransferStatus(batch, 0, owner).ok());
    EXPECT_EQ(owner.s, TransferStatusEnum::COMPLETED);

    EXPECT_TRUE(engine.freeBatch(batch).ok());
    EXPECT_TRUE(
        engine.unregisterLocalMemory(buffer.data(), buffer.size()).ok());
}

}  // namespace
}  // namespace tent
}  // namespace mooncake
