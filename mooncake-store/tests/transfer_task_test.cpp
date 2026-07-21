// transfer_task_test.cpp
#include "transfer_task.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#include "types.h"

namespace mooncake {

// Test fixture for TransferTask tests
// TODO: Currently, this test does not cover TransferSubmitter and
// TransferEngine integration. Will add more tests in the future.
class TransferTaskTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Initialize glog for logging
        google::InitGoogleLogging("TransferTaskTest");
        FLAGS_logtostderr = 1;  // Output logs to stderr
    }

    void TearDown() override {
        // Cleanup glog
        google::ShutdownGoogleLogging();
    }
};

// Test basic MemcpyOperation functionality
TEST_F(TransferTaskTest, MemcpyOperationBasic) {
    const size_t data_size = 1024;
    std::vector<char> src_data(data_size, 'A');
    std::vector<char> dest_data(data_size, 'B');

    // Create memcpy operation
    MemcpyOperation op(dest_data.data(), src_data.data(), data_size);

    // Verify operation parameters
    EXPECT_EQ(op.dest, dest_data.data());
    EXPECT_EQ(op.src, src_data.data());
    EXPECT_EQ(op.size, data_size);

    // Perform memcpy manually to test
    std::memcpy(op.dest, op.src, op.size);

    // Verify data was copied correctly
    EXPECT_EQ(dest_data, src_data);
    for (size_t i = 0; i < data_size; ++i) {
        EXPECT_EQ(dest_data[i], 'A');
    }
}

// Test MemcpyOperationState functionality
TEST_F(TransferTaskTest, MemcpyOperationState) {
    auto state = std::make_shared<MemcpyOperationState>();

    // Initially not completed
    EXPECT_FALSE(state->is_completed());
    EXPECT_EQ(state->get_strategy(), TransferStrategy::LOCAL_MEMCPY);

    // Set completed with success
    state->set_completed(ErrorCode::OK);
    EXPECT_TRUE(state->is_completed());
    EXPECT_EQ(state->get_result(), ErrorCode::OK);
}

// Test MemcpyWorkerPool basic functionality
TEST_F(TransferTaskTest, MemcpyWorkerPoolBasic) {
    MemcpyWorkerPool pool;

    const size_t data_size = 512;
    std::vector<char> src_data(data_size, 'X');
    std::vector<char> dest_data(data_size, 'Y');

    auto state = std::make_shared<MemcpyOperationState>();

    // Create memcpy operations
    std::vector<MemcpyOperation> operations;
    operations.emplace_back(dest_data.data(), src_data.data(), data_size);

    // Create and submit task
    MemcpyTask task(std::move(operations), state);
    pool.submitTask(std::move(task));

    // Wait for completion
    state->wait_for_completion();

    // Verify completion and result
    EXPECT_TRUE(state->is_completed());
    EXPECT_EQ(state->get_result(), ErrorCode::OK);

    // Verify data was copied correctly
    for (size_t i = 0; i < data_size; ++i) {
        EXPECT_EQ(dest_data[i], 'X');
    }
}

// Test multiple memcpy operations in one task
TEST_F(TransferTaskTest, MemcpyWorkerPoolMultipleOperations) {
    MemcpyWorkerPool pool;

    const size_t num_ops = 3;
    const size_t data_size = 256;

    std::vector<std::vector<char>> src_buffers(num_ops);
    std::vector<std::vector<char>> dest_buffers(num_ops);

    // Initialize source buffers with different patterns
    for (size_t i = 0; i < num_ops; ++i) {
        src_buffers[i].resize(data_size, 'A' + i);
        dest_buffers[i].resize(data_size, 'Z');
    }

    auto state = std::make_shared<MemcpyOperationState>();

    // Create multiple memcpy operations
    std::vector<MemcpyOperation> operations;
    for (size_t i = 0; i < num_ops; ++i) {
        operations.emplace_back(dest_buffers[i].data(), src_buffers[i].data(),
                                data_size);
    }

    // Create and submit task
    MemcpyTask task(std::move(operations), state);
    pool.submitTask(std::move(task));

    // Wait for completion
    state->wait_for_completion();

    // Verify completion and result
    EXPECT_TRUE(state->is_completed());
    EXPECT_EQ(state->get_result(), ErrorCode::OK);

    // Verify all data was copied correctly
    for (size_t i = 0; i < num_ops; ++i) {
        for (size_t j = 0; j < data_size; ++j) {
            EXPECT_EQ(dest_buffers[i][j], 'A' + i);
        }
    }
}

// Test the locality decision used by TransferSubmitter::isLocalTransfer.
// Same-host different-process pairs share an IP but have distinct ports;
// they must NOT be treated as locally addressable, otherwise memcpy in the
// caller process would dereference a virtual address belonging to a peer
// process and segfault.
TEST_F(TransferTaskTest, IsSameProcessEndpoint) {
    // Empty inputs -> not same-process (cannot prove locality).
    EXPECT_FALSE(TransferSubmitter::isSameProcessEndpoint("", ""));
    EXPECT_FALSE(
        TransferSubmitter::isSameProcessEndpoint("", "192.168.1.10:12345"));
    EXPECT_FALSE(
        TransferSubmitter::isSameProcessEndpoint("192.168.1.10:12345", ""));

    // Identical ip:port -> same process.
    EXPECT_TRUE(TransferSubmitter::isSameProcessEndpoint("192.168.1.10:12345",
                                                         "192.168.1.10:12345"));

    // Same host, different port -> different process, NOT local.
    // This is the regression case fixed by this change.
    EXPECT_FALSE(TransferSubmitter::isSameProcessEndpoint(
        "192.168.1.10:12345", "192.168.1.10:12346"));

    // Different hosts -> not local.
    EXPECT_FALSE(TransferSubmitter::isSameProcessEndpoint(
        "192.168.1.10:12345", "192.168.1.11:12345"));

    // Hostname endpoints (non-P2P metadata mode) compare as full strings.
    EXPECT_TRUE(TransferSubmitter::isSameProcessEndpoint("host-a", "host-a"));
    EXPECT_FALSE(TransferSubmitter::isSameProcessEndpoint("host-a", "host-b"));
}

TEST_F(TransferTaskTest, BuildGdsSsdTransferRequests) {
    GdsSsdDescriptor descriptor;
    descriptor.segment_name = "gds_pool";
    descriptor.segment_uri = "block:///dev/mooncake/gds_pool";
    descriptor.offset = 8192;
    descriptor.object_size = 8192;
    descriptor.block_size = 4096;
    descriptor.allocation_alignment = 4096;

    std::vector<Slice> slices = {
        {reinterpret_cast<void*>(static_cast<uintptr_t>(0x10000)), 4096},
        {reinterpret_cast<void*>(static_cast<uintptr_t>(0x12000)), 4096},
    };
    std::vector<TransferRequest> requests;
    ASSERT_TRUE(TransferSubmitter::buildGdsSsdTransferRequests(
        descriptor, slices, TransferRequest::WRITE, requests));
    ASSERT_EQ(requests.size(), 2u);
    EXPECT_EQ(requests[0].opcode, TransferRequest::WRITE);
    EXPECT_EQ(requests[0].source, slices[0].ptr);
    EXPECT_EQ(requests[0].target_id, 0u);
    EXPECT_EQ(requests[0].target_offset, descriptor.offset);
    EXPECT_EQ(requests[0].length, 4096u);
    EXPECT_EQ(requests[0].transport_hint, 5);
    EXPECT_EQ(requests[1].target_offset, descriptor.offset + 4096);

    ASSERT_TRUE(TransferSubmitter::buildGdsSsdTransferRequests(
        descriptor, slices, TransferRequest::READ, requests));
    EXPECT_EQ(requests[0].opcode, TransferRequest::READ);

    // KV-cache blocks can begin at a non-4 KiB offset inside a larger
    // cuFile-registered allocation. TENT resolves the registered base and
    // passes this delta through CUfileIOParams_t::devPtr_offset.
    slices = {
        {reinterpret_cast<void*>(static_cast<uintptr_t>(0x10800)), 4096},
        {reinterpret_cast<void*>(static_cast<uintptr_t>(0x12800)), 4096},
    };
    ASSERT_TRUE(TransferSubmitter::buildGdsSsdTransferRequests(
        descriptor, slices, TransferRequest::WRITE, requests));
    ASSERT_EQ(requests.size(), 2u);
    EXPECT_EQ(requests[0].source, slices[0].ptr);
    EXPECT_EQ(requests[1].source, slices[1].ptr);

    // RealClient splits at kMaxSliceSize (16 MiB - 16). Adjacent pieces must
    // be coalesced before applying GDS alignment requirements.
    GdsSsdDescriptor split_descriptor = descriptor;
    split_descriptor.object_size = 16 * 1024 * 1024;
    constexpr uintptr_t kSplitBufferAddress = 0x20000;
    slices = {
        {reinterpret_cast<void*>(kSplitBufferAddress), kMaxSliceSize},
        {reinterpret_cast<void*>(kSplitBufferAddress + kMaxSliceSize), 16},
    };
    ASSERT_TRUE(TransferSubmitter::buildGdsSsdTransferRequests(
        split_descriptor, slices, TransferRequest::WRITE, requests));
    ASSERT_EQ(requests.size(), 1u);
    EXPECT_EQ(requests[0].source, slices[0].ptr);
    EXPECT_EQ(requests[0].length, split_descriptor.object_size);
    EXPECT_EQ(requests[0].target_offset, split_descriptor.offset);

    slices = {
        {reinterpret_cast<void*>(static_cast<uintptr_t>(0x30000)), 4096},
        {reinterpret_cast<void*>(static_cast<uintptr_t>(0x31000)), 4096},
    };
    ASSERT_TRUE(TransferSubmitter::buildGdsSsdTransferRequests(
        descriptor, slices, TransferRequest::WRITE, requests));
    ASSERT_EQ(requests.size(), 1u);
    EXPECT_EQ(requests[0].source, slices[0].ptr);
    EXPECT_EQ(requests[0].length, descriptor.object_size);

    GdsSsdDescriptor large_descriptor = descriptor;
    constexpr size_t kLargeSliceCount = 8192;
    large_descriptor.object_size = kLargeSliceCount * 4096;
    slices.clear();
    slices.reserve(kLargeSliceCount);
    constexpr uintptr_t kLargeBufferAddress = 0x100000;
    for (size_t index = 0; index < kLargeSliceCount; ++index) {
        slices.push_back(
            {reinterpret_cast<void*>(kLargeBufferAddress + index * 4096),
             4096});
    }
    ASSERT_TRUE(TransferSubmitter::buildGdsSsdTransferRequests(
        large_descriptor, slices, TransferRequest::WRITE, requests));
    ASSERT_EQ(requests.size(), 2u);
    EXPECT_EQ(requests[0].source, slices[0].ptr);
    EXPECT_EQ(requests[0].length, kMaxSliceSize + 16);
    EXPECT_EQ(requests[1].target_offset,
              large_descriptor.offset + kMaxSliceSize + 16);
    EXPECT_EQ(requests[1].length, kMaxSliceSize + 16);
}

TEST_F(TransferTaskTest, RejectsInvalidGdsSsdTransferRequests) {
    GdsSsdDescriptor descriptor;
    descriptor.segment_name = "gds_pool";
    descriptor.segment_uri = "block:///dev/mooncake/gds_pool";
    descriptor.offset = 0;
    descriptor.object_size = 8192;
    descriptor.block_size = 4096;
    descriptor.allocation_alignment = 4096;

    const Slice first{
        reinterpret_cast<void*>(static_cast<uintptr_t>(0x10000)), 4096};
    const Slice second{
        reinterpret_cast<void*>(static_cast<uintptr_t>(0x12000)), 4096};
    std::vector<TransferRequest> requests;

    EXPECT_FALSE(TransferSubmitter::buildGdsSsdTransferRequests(
        descriptor, {}, TransferRequest::WRITE, requests));
    EXPECT_TRUE(requests.empty());

    EXPECT_FALSE(TransferSubmitter::buildGdsSsdTransferRequests(
        descriptor, {{first.ptr, 0}, second}, TransferRequest::WRITE,
        requests));
    EXPECT_FALSE(TransferSubmitter::buildGdsSsdTransferRequests(
        descriptor, {first}, TransferRequest::WRITE, requests));
    EXPECT_FALSE(TransferSubmitter::buildGdsSsdTransferRequests(
        descriptor,
        {{first.ptr, 4080},
         {reinterpret_cast<void*>(static_cast<uintptr_t>(0x12000)), 4112}},
        TransferRequest::WRITE, requests));

    GdsSsdDescriptor overlap_descriptor = descriptor;
    overlap_descriptor.object_size = 12288;
    EXPECT_FALSE(TransferSubmitter::buildGdsSsdTransferRequests(
        overlap_descriptor,
        {{first.ptr, 8192},
         {reinterpret_cast<void*>(static_cast<uintptr_t>(0x11000)), 4096}},
        TransferRequest::WRITE, requests));

    GdsSsdDescriptor overflow_descriptor = descriptor;
    overflow_descriptor.offset = std::numeric_limits<uint64_t>::max() - 4095;
    EXPECT_FALSE(TransferSubmitter::buildGdsSsdTransferRequests(
        overflow_descriptor, {first, second}, TransferRequest::WRITE,
        requests));

    GdsSsdDescriptor invalid_uri_descriptor = descriptor;
    invalid_uri_descriptor.segment_uri = "file:///tmp/gds_pool";
    EXPECT_FALSE(TransferSubmitter::buildGdsSsdTransferRequests(
        invalid_uri_descriptor, {first, second}, TransferRequest::WRITE,
        requests));
    EXPECT_FALSE(TransferSubmitter::buildGdsSsdTransferRequests(
        descriptor, {first, second},
        static_cast<TransferRequest::OpCode>(7), requests));
}

// Test TransferStrategy enum and stream operator
TEST_F(TransferTaskTest, TransferStrategyEnum) {
    // Test enum values
    EXPECT_EQ(static_cast<int>(TransferStrategy::LOCAL_MEMCPY), 0);
    EXPECT_EQ(static_cast<int>(TransferStrategy::TRANSFER_ENGINE), 1);
    EXPECT_EQ(static_cast<int>(TransferStrategy::FILE_READ), 2);
    EXPECT_EQ(static_cast<int>(TransferStrategy::EMPTY), 3);
    EXPECT_EQ(static_cast<int>(TransferStrategy::SPDK_NVMF), 4);

    // Test stream operator
    std::ostringstream oss;
    oss << TransferStrategy::LOCAL_MEMCPY;
    EXPECT_EQ(oss.str(), "LOCAL_MEMCPY");

    oss.str("");
    oss << TransferStrategy::TRANSFER_ENGINE;
    EXPECT_EQ(oss.str(), "TRANSFER_ENGINE");

    oss.str("");
    oss << TransferStrategy::SPDK_NVMF;
    EXPECT_EQ(oss.str(), "SPDK_NVMF");

    oss.str("");
    oss << TransferStrategy::FILE_READ;
    EXPECT_EQ(oss.str(), "FILE_READ");

    oss.str("");
    oss << TransferStrategy::EMPTY;
    EXPECT_EQ(oss.str(), "EMPTY");
}

}  // namespace mooncake

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
