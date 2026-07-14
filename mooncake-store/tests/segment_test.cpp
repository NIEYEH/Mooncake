#include "segment.h"
#include "master_metric_manager.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <boost/functional/hash.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef USE_GDS
#include <cufile.h>

#include "tent/common/config.h"
#include "tent/transfer_engine.h"
#endif

namespace mooncake {

namespace {

#if defined(__linux__) && defined(USE_GDS)
constexpr char kGdsTestPathEnv[] = "MOONCAKE_GDS_TEST_PATH";
constexpr size_t kGdsTestFileSize = 4 * 1024 * 1024;

class ScopedFd {
   public:
    explicit ScopedFd(int fd) : fd_(fd) {}
    ~ScopedFd() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;

    int get() const { return fd_; }

   private:
    int fd_;
};

class GdsTestTarget {
   public:
    ~GdsTestTarget() {
        if (remove_on_destroy && !path.empty()) {
            unlink(path.c_str());
        }
    }

    GdsTestTarget(const GdsTestTarget&) = delete;
    GdsTestTarget& operator=(const GdsTestTarget&) = delete;

    static std::unique_ptr<GdsTestTarget> Prepare(const std::string& input,
                                                  std::string& error) {
        auto target = std::unique_ptr<GdsTestTarget>(new GdsTestTarget());
        std::string requested_scheme;
        target->path = input;

        constexpr char kBlockPrefix[] = "block://";
        constexpr char kFilePrefix[] = "file://";
        if (input.rfind(kBlockPrefix, 0) == 0) {
            requested_scheme = "block";
            target->path = input.substr(std::strlen(kBlockPrefix));
        } else if (input.rfind(kFilePrefix, 0) == 0) {
            requested_scheme = "file";
            target->path = input.substr(std::strlen(kFilePrefix));
        }

        if (target->path.empty()) {
            error = "test path is empty";
            return nullptr;
        }

        struct stat st{};
        if (stat(target->path.c_str(), &st) != 0) {
            error = "stat failed for '" + target->path + "': " +
                    std::strerror(errno);
            return nullptr;
        }

        if (S_ISDIR(st.st_mode)) {
            if (requested_scheme == "block") {
                error = "block:// target is a directory: " + target->path;
                return nullptr;
            }
            if (!target->path.empty() && target->path.back() != '/') {
                target->path += '/';
            }
            target->path += ".mooncake-gds-segment-test-" +
                            std::to_string(static_cast<uint64_t>(getpid()));

            ScopedFd fd(open(target->path.c_str(),
                             O_CREAT | O_EXCL | O_RDWR | O_DIRECT, 0600));
            if (fd.get() < 0) {
                error = "failed to create an O_DIRECT test file in the "
                        "mount directory '" +
                        target->path + "': " + std::strerror(errno);
                return nullptr;
            }
            target->remove_on_destroy = true;
            if (ftruncate(fd.get(), kGdsTestFileSize) != 0) {
                error = "failed to size test file '" + target->path +
                        "': " + std::strerror(errno);
                return nullptr;
            }
            target->uri = std::string(kFilePrefix) + target->path;
            target->expected_size = kGdsTestFileSize;
            return target;
        }

        const bool is_block = S_ISBLK(st.st_mode);
        const bool is_regular_file = S_ISREG(st.st_mode);
        if (!is_block && !is_regular_file) {
            error = "target is neither a block device, regular file, nor "
                    "directory: " +
                    target->path;
            return nullptr;
        }
        if (requested_scheme == "block" && !is_block) {
            error = "block:// target is not a block device: " + target->path;
            return nullptr;
        }
        if (requested_scheme == "file" && !is_regular_file) {
            error = "file:// target is not a regular file: " + target->path;
            return nullptr;
        }

        ScopedFd fd(open(target->path.c_str(), O_RDWR | O_DIRECT));
        if (fd.get() < 0) {
            error = "failed to open '" + target->path +
                    "' with O_RDWR | O_DIRECT: " + std::strerror(errno);
            return nullptr;
        }

        target->uri = std::string(is_block ? kBlockPrefix : kFilePrefix) +
                      target->path;
        target->is_block_device = is_block;
        target->expected_size = is_regular_file
                                    ? static_cast<uint64_t>(st.st_size)
                                    : 0;
        return target;
    }

    std::string path;
    std::string uri;
    uint64_t expected_size{0};
    bool is_block_device{false};
    bool remove_on_destroy{false};

   private:
    GdsTestTarget() = default;
};

bool RegisterGdsStorageHandle(const std::string& path, std::string& error) {
    ScopedFd fd(open(path.c_str(), O_RDWR | O_DIRECT));
    if (fd.get() < 0) {
        error = "failed to reopen '" + path +
                "' for cuFile registration: " + std::strerror(errno);
        return false;
    }

    CUfileDescr_t descriptor{};
    descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descriptor.handle.fd = fd.get();

    CUfileHandle_t handle = nullptr;
    const auto result = cuFileHandleRegister(&handle, &descriptor);
    if (result.err != CU_FILE_SUCCESS) {
        error = "cuFileHandleRegister failed for '" + path +
                "' with error code " + std::to_string(result.err);
        return false;
    }
    cuFileHandleDeregister(handle);
    return true;
}
#endif

}  // namespace

// Test fixture for Segment tests
class SegmentTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Initialize glog for logging
        google::InitGoogleLogging("EvictionStrategyTest");
        FLAGS_logtostderr = 1;  // Output logs to stderr
    }

    void TearDown() override {
        // Cleanup glog
        google::ShutdownGoogleLogging();
    }

    void ValidateMountedSegments(const SegmentManager& segment_manager,
                                 const std::vector<Segment>& segments,
                                 const std::vector<UUID>& client_ids) {
        // validate client_segments_ and mounted_segments_
        size_t total_num = 0;
        for (const auto& it : segment_manager.client_segments_) {
            total_num += it.second.size();
        }
        ASSERT_EQ(total_num, segments.size());
        ASSERT_EQ(segment_manager.mounted_segments_.size(), segments.size());
        for (size_t i = 0; i < client_ids.size(); i++) {
            auto client_it =
                segment_manager.client_segments_.find(client_ids[i]);
            ASSERT_NE(client_it, segment_manager.client_segments_.end());
            auto segment_it =
                std::find(client_it->second.begin(), client_it->second.end(),
                          segments[i].id);
            ASSERT_NE(segment_it, client_it->second.end());
            ASSERT_EQ(*segment_it, segments[i].id);

            ASSERT_NE(segment_manager.mounted_segments_.find(segments[i].id),
                      segment_manager.mounted_segments_.end());
            MountedSegment seg =
                segment_manager.mounted_segments_.at(segments[i].id);
            ASSERT_EQ(seg.segment.id, segments[i].id);
            ASSERT_EQ(seg.segment.name, segments[i].name);
            ASSERT_EQ(seg.segment.size, segments[i].size);
            ASSERT_EQ(seg.segment.base, segments[i].base);
            ASSERT_EQ(seg.status, SegmentStatus::OK);
            ASSERT_EQ(seg.buf_allocator->getSegmentName(), segments[i].name);
            ASSERT_EQ(seg.buf_allocator->capacity(), segments[i].size);
        }

        // validate allocator manager
        const auto& allocator_manager = segment_manager.allocator_manager_;

        total_num = 0;
        for (const auto& name : allocator_manager.getNames()) {
            auto allocators = allocator_manager.getAllocators(name);
            ASSERT_NE(allocators, nullptr);
            total_num += allocators->size();
        }
        ASSERT_EQ(total_num, segments.size());

        for (const auto& segment : segments) {
            auto allocators = allocator_manager.getAllocators(segment.name);
            ASSERT_NE(allocators, nullptr);

            // validate allocator exist in allocator_manager
            MountedSegment mounted_segment =
                segment_manager.mounted_segments_.at(segment.id);
            auto allocator = mounted_segment.buf_allocator;
            ASSERT_NE(std::find(allocators->begin(), allocators->end(),
                                mounted_segment.buf_allocator),
                      allocators->end());
        }
    }

    void ValidateMountedSegment(const SegmentManager& segment_manager,
                                const Segment segment, const UUID& client_id) {
        std::vector<Segment> segments;
        segments.push_back(segment);
        std::vector<UUID> client_ids;
        client_ids.push_back(client_id);
        ValidateMountedSegments(segment_manager, segments, client_ids);
    }

    bool HasAllocatorForSegment(const SegmentManager& segment_manager,
                                const UUID& segment_id) {
        const auto mounted_it =
            segment_manager.mounted_segments_.find(segment_id);
        if (mounted_it == segment_manager.mounted_segments_.end()) {
            return false;
        }

        const auto& mounted_segment = mounted_it->second;
        const auto* allocators =
            segment_manager.allocator_manager_.getAllocators(
                mounted_segment.segment.name);
        if (allocators == nullptr) {
            return false;
        }

        return std::find(allocators->begin(), allocators->end(),
                         mounted_segment.buf_allocator) != allocators->end();
    }

    void ValidateMountedLocalDiskSegments(
        const SegmentManager& segment_manager,
        const std::vector<std::shared_ptr<LocalDiskSegment>>& segments,
        const std::vector<UUID>& client_ids) {
        size_t total_num = segment_manager.client_local_disk_segment_.size();
        ASSERT_EQ(total_num, segments.size());
        for (size_t i = 0; i < client_ids.size(); i++) {
            LOG(INFO) << "Mounted local disk segment " << client_ids[i];
            auto client_it =
                segment_manager.client_local_disk_segment_.find(client_ids[i]);
            ASSERT_NE(client_it,
                      segment_manager.client_local_disk_segment_.end());
            const auto& segment = segments.at(i);
            ASSERT_EQ(client_it->second->enable_offloading,
                      segment->enable_offloading);
        }
    }

    void ValidateMountedLocalDiskSegments(
        const SegmentManager& segment_manager,
        const std::shared_ptr<LocalDiskSegment>& segment,
        const UUID& client_id) {
        std::vector<std::shared_ptr<LocalDiskSegment>> segments;
        segments.push_back(segment);
        std::vector<UUID> client_ids;
        client_ids.push_back(client_id);
        ValidateMountedLocalDiskSegments(segment_manager, segments, client_ids);
    }
};

// Mount Segment Operations Tests:
TEST_F(SegmentTest, MountSegmentSuccess) {
    SegmentManager segment_manager;
    // Create a valid segment and client ID
    Segment segment;
    segment.id = generate_uuid();
    segment.name = "test_segment";
    segment.size = 1024 * 1024 * 16;
    segment.base = 0x100000000;

    UUID client_id = generate_uuid();

    // Get segment access and attempt to mount
    auto segment_access = segment_manager.getSegmentAccess();
    ASSERT_EQ(segment_access.MountSegment(segment, client_id), ErrorCode::OK);

    // Verify segment is properly mounted
    ValidateMountedSegment(segment_manager, segment, client_id);
}

// MountSegmentDuplicate Tests:
// 1. MountSegment with the same segment id. The second mount operation return
// SEGMENT_ALREADY_EXISTS.
// 2. MountSegment with different segment id and the same segment name should be
// considered as different segments. Validate the status of SegmentManager use
// ValidateMountedSegments function.
TEST_F(SegmentTest, MountSegmentDuplicate) {
    SegmentManager segment_manager;
    // Create a valid segment and client ID
    Segment segment;
    segment.id = generate_uuid();
    segment.name = "test_segment";
    segment.size = 1024 * 1024 * 16;
    segment.base = 0x100000000;

    UUID client_id = generate_uuid();

    // Get segment access and mount first time
    auto segment_access = segment_manager.getSegmentAccess();
    ASSERT_EQ(segment_access.MountSegment(segment, client_id), ErrorCode::OK);

    // Verify first mount
    ValidateMountedSegment(segment_manager, segment, client_id);

    // Test duplicate mount - mount the same segment again
    ASSERT_EQ(segment_access.MountSegment(segment, client_id),
              ErrorCode::SEGMENT_ALREADY_EXISTS);

    // Verify state remains the same after duplicate mount
    ValidateMountedSegment(segment_manager, segment, client_id);

    // Create a new segment with same name but different ID
    Segment segment2;
    segment2.id = generate_uuid();  // Different ID
    segment2.name = segment.name;   // Same name
    segment2.size = segment.size * 2;
    segment2.base = segment.base + segment.size;

    // Mount the second segment
    ASSERT_EQ(segment_access.MountSegment(segment2, client_id), ErrorCode::OK);

    // Verify both segments are mounted correctly
    std::vector<Segment> segments = {segment, segment2};
    std::vector<UUID> client_ids = {client_id, client_id};
    ValidateMountedSegments(segment_manager, segments, client_ids);
}

// UnmountSegmentSuccess:
// 1. Mount a segment and then unmount it. Unmount operation return success.
// 2. Use ValidateMountedSegments function to validate the status of
// SegmentManager.
TEST_F(SegmentTest, UnmountSegmentSuccess) {
    SegmentManager segment_manager;

    // Create and mount a segment
    Segment segment;
    segment.id = generate_uuid();
    segment.name = "test_segment";
    segment.size = 1024 * 1024 * 16;
    segment.base = 0x100000000;

    UUID client_id = generate_uuid();

    // Get segment access and mount
    auto segment_access = segment_manager.getSegmentAccess();
    ASSERT_EQ(segment_access.MountSegment(segment, client_id), ErrorCode::OK);

    // Verify segment is mounted correctly
    ValidateMountedSegment(segment_manager, segment, client_id);

    // Prepare unmount
    size_t metrics_dec_capacity = 0;
    ASSERT_EQ(
        segment_access.PrepareUnmountSegment(segment.id, metrics_dec_capacity),
        ErrorCode::OK);
    ASSERT_EQ(metrics_dec_capacity, segment.size);

    // Commit unmount
    ASSERT_EQ(segment_access.CommitUnmountSegment(segment.id, client_id,
                                                  metrics_dec_capacity),
              ErrorCode::OK);

    // Verify segment is unmounted correctly
    std::vector<Segment> empty_segment_vec;
    std::vector<UUID> empty_client_ids_vec;
    ValidateMountedSegments(segment_manager, empty_segment_vec,
                            empty_client_ids_vec);
}

// UnmountSegmentDuplicate:
// 1. Mount a segment and then unmount it twice. The second unmount operation
// returns SEGMENT_NOT_FOUND.
// 2. Only use ValidateMountedSegments function to validate the status of
// SegmentManager. Do not use other interfaces for validation.
TEST_F(SegmentTest, UnmountSegmentDuplicate) {
    SegmentManager segment_manager;

    // Create and mount a segment
    Segment segment;
    segment.id = generate_uuid();
    segment.name = "test_segment";
    segment.size = 1024 * 1024 * 16;
    segment.base = 0x100000000;

    UUID client_id = generate_uuid();

    // Get segment access and mount
    auto segment_access = segment_manager.getSegmentAccess();
    ASSERT_EQ(segment_access.MountSegment(segment, client_id), ErrorCode::OK);

    // Verify initial mounted state
    ValidateMountedSegment(segment_manager, segment, client_id);

    // First unmount
    size_t metrics_dec_capacity = 0;
    ASSERT_EQ(
        segment_access.PrepareUnmountSegment(segment.id, metrics_dec_capacity),
        ErrorCode::OK);
    ASSERT_EQ(segment_access.CommitUnmountSegment(segment.id, client_id,
                                                  metrics_dec_capacity),
              ErrorCode::OK);

    // Verify segment is unmounted after first unmount
    std::vector<Segment> empty_segment_vec;
    std::vector<UUID> empty_client_ids_vec;
    ValidateMountedSegments(segment_manager, empty_segment_vec,
                            empty_client_ids_vec);

    // Second unmount attempt
    metrics_dec_capacity = 0;
    ASSERT_EQ(
        segment_access.PrepareUnmountSegment(segment.id, metrics_dec_capacity),
        ErrorCode::SEGMENT_NOT_FOUND);

    // Verify segment remains unmounted after second unmount
    ValidateMountedSegments(segment_manager, empty_segment_vec,
                            empty_client_ids_vec);
}

TEST_F(SegmentTest, SegmentLifecycleStatusControlsAllocation) {
    SegmentManager segment_manager;

    Segment segment;
    segment.id = generate_uuid();
    segment.name = "status_segment";
    segment.size = 1024 * 1024 * 16;
    segment.base = 0x100000000;

    UUID client_id = generate_uuid();

    auto segment_access = segment_manager.getSegmentAccess();
    ASSERT_EQ(segment_access.MountSegment(segment, client_id), ErrorCode::OK);

    SegmentStatus status = SegmentStatus::UNDEFINED;
    ASSERT_EQ(segment_access.GetSegmentStatusByName(segment.name, status),
              ErrorCode::OK);
    EXPECT_EQ(status, SegmentStatus::OK);
    EXPECT_TRUE(segment_access.IsSegmentAllocatable(segment.name));
    EXPECT_TRUE(HasAllocatorForSegment(segment_manager, segment.id));

    ASSERT_EQ(segment_access.SetSegmentStatusByName(segment.name,
                                                    SegmentStatus::DRAINING),
              ErrorCode::OK);
    ASSERT_EQ(segment_access.GetSegmentStatusByName(segment.name, status),
              ErrorCode::OK);
    EXPECT_EQ(status, SegmentStatus::DRAINING);
    EXPECT_FALSE(segment_access.IsSegmentAllocatable(segment.name));
    EXPECT_FALSE(HasAllocatorForSegment(segment_manager, segment.id));

    ASSERT_EQ(segment_access.SetSegmentStatusByName(segment.name,
                                                    SegmentStatus::DRAINED),
              ErrorCode::OK);
    ASSERT_EQ(segment_access.GetSegmentStatusByName(segment.name, status),
              ErrorCode::OK);
    EXPECT_EQ(status, SegmentStatus::DRAINED);
    EXPECT_FALSE(segment_access.IsSegmentAllocatable(segment.name));
    EXPECT_FALSE(HasAllocatorForSegment(segment_manager, segment.id));

    ASSERT_EQ(
        segment_access.SetSegmentStatusByName(segment.name, SegmentStatus::OK),
        ErrorCode::OK);
    ASSERT_EQ(segment_access.GetSegmentStatusByName(segment.name, status),
              ErrorCode::OK);
    EXPECT_EQ(status, SegmentStatus::OK);
    EXPECT_TRUE(segment_access.IsSegmentAllocatable(segment.name));
    EXPECT_TRUE(HasAllocatorForSegment(segment_manager, segment.id));
}

TEST_F(SegmentTest, HostOrderedSegmentsTracksMountStatusAndUnmount) {
    SegmentManager segment_manager;

    Segment segment0;
    segment0.id = generate_uuid();
    segment0.name = "host0_segment";
    segment0.size = 1024 * 1024 * 16;
    segment0.base = 0x100000000;
    segment0.host_id = "host0";

    Segment segment1;
    segment1.id = generate_uuid();
    segment1.name = "host1_segment";
    segment1.size = 1024 * 1024 * 16;
    segment1.base = 0x200000000;
    segment1.host_id = "host1";

    UUID client_id = generate_uuid();

    {
        auto segment_access = segment_manager.getSegmentAccess();
        ASSERT_EQ(segment_access.MountSegment(segment0, client_id),
                  ErrorCode::OK);
        ASSERT_EQ(segment_access.MountSegment(segment1, client_id),
                  ErrorCode::OK);
    }

    {
        auto allocator_access = segment_manager.getAllocatorAccess();
        auto ordered =
            allocator_access.GetHostOrderedSegments("host1", "test_key");
        ASSERT_GE(ordered.size(), 2u);
        EXPECT_EQ(ordered[0], segment1.name);
    }

    {
        auto segment_access = segment_manager.getSegmentAccess();
        ASSERT_EQ(segment_access.SetSegmentStatusByName(
                      segment1.name, SegmentStatus::DRAINING),
                  ErrorCode::OK);
    }

    {
        auto allocator_access = segment_manager.getAllocatorAccess();
        auto ordered =
            allocator_access.GetHostOrderedSegments("host1", "test_key");
        ASSERT_EQ(ordered.size(), 1u);
        EXPECT_EQ(ordered[0], segment0.name);
    }

    {
        auto segment_access = segment_manager.getSegmentAccess();
        ASSERT_EQ(segment_access.SetSegmentStatusByName(segment1.name,
                                                        SegmentStatus::OK),
                  ErrorCode::OK);
        size_t metrics_dec_capacity = 0;
        ASSERT_EQ(segment_access.PrepareUnmountSegment(segment1.id,
                                                       metrics_dec_capacity),
                  ErrorCode::OK);
        ASSERT_EQ(segment_access.CommitUnmountSegment(segment1.id, client_id,
                                                      metrics_dec_capacity),
                  ErrorCode::OK);
    }

    {
        auto allocator_access = segment_manager.getAllocatorAccess();
        auto ordered =
            allocator_access.GetHostOrderedSegments("host1", "test_key");
        ASSERT_EQ(ordered.size(), 1u);
        EXPECT_EQ(ordered[0], segment0.name);
    }
}

TEST_F(SegmentTest, HostOrderedSegmentsKeepsNameUntilLastSameNameSegmentGone) {
    SegmentManager segment_manager;

    Segment segment0;
    segment0.id = generate_uuid();
    segment0.name = "shared_host_segment";
    segment0.size = 1024 * 1024 * 16;
    segment0.base = 0x100000000;
    segment0.host_id = "host1";

    Segment segment1;
    segment1.id = generate_uuid();
    segment1.name = segment0.name;
    segment1.size = 1024 * 1024 * 16;
    segment1.base = 0x200000000;
    segment1.host_id = "host1";

    UUID client_id = generate_uuid();
    {
        auto segment_access = segment_manager.getSegmentAccess();
        ASSERT_EQ(segment_access.MountSegment(segment0, client_id),
                  ErrorCode::OK);
        ASSERT_EQ(segment_access.MountSegment(segment1, client_id),
                  ErrorCode::OK);
    }

    {
        auto allocator_access = segment_manager.getAllocatorAccess();
        auto ordered =
            allocator_access.GetHostOrderedSegments("host1", "test_key");
        ASSERT_EQ(ordered.size(), 1u);
        EXPECT_EQ(ordered[0], segment0.name);
    }

    {
        auto segment_access = segment_manager.getSegmentAccess();
        size_t metrics_dec_capacity = 0;
        ASSERT_EQ(segment_access.PrepareUnmountSegment(segment0.id,
                                                       metrics_dec_capacity),
                  ErrorCode::OK);
        ASSERT_EQ(segment_access.CommitUnmountSegment(segment0.id, client_id,
                                                      metrics_dec_capacity),
                  ErrorCode::OK);
    }

    {
        auto allocator_access = segment_manager.getAllocatorAccess();
        auto ordered =
            allocator_access.GetHostOrderedSegments("host1", "test_key");
        ASSERT_EQ(ordered.size(), 1u);
        EXPECT_EQ(ordered[0], segment1.name);
    }
}

TEST_F(SegmentTest, HostOrderedSegmentsRotateWithinSameHostByKey) {
    SegmentManager segment_manager;

    Segment segment_a;
    segment_a.id = generate_uuid();
    segment_a.name = "host1_segment_a";
    segment_a.size = 1024 * 1024 * 16;
    segment_a.base = 0x100000000;
    segment_a.host_id = "host1";

    Segment segment_b;
    segment_b.id = generate_uuid();
    segment_b.name = "host1_segment_b";
    segment_b.size = 1024 * 1024 * 16;
    segment_b.base = 0x200000000;
    segment_b.host_id = "host1";

    UUID client_id = generate_uuid();
    {
        auto segment_access = segment_manager.getSegmentAccess();
        ASSERT_EQ(segment_access.MountSegment(segment_a, client_id),
                  ErrorCode::OK);
        ASSERT_EQ(segment_access.MountSegment(segment_b, client_id),
                  ErrorCode::OK);
    }

    const std::string key = "stable_rotation_key";
    std::vector<std::string> sorted_segments = {segment_a.name, segment_b.name};
    std::sort(sorted_segments.begin(), sorted_segments.end());
    const size_t start = std::hash<std::string>{}(key) % sorted_segments.size();

    auto allocator_access = segment_manager.getAllocatorAccess();
    auto ordered = allocator_access.GetHostOrderedSegments("host1", key);
    ASSERT_EQ(ordered.size(), 2u);
    EXPECT_EQ(ordered[0], sorted_segments[start]);
    EXPECT_EQ(ordered[1],
              sorted_segments[(start + 1) % sorted_segments.size()]);
}

TEST_F(SegmentTest, PrepareUnmountDrainedSegment) {
    SegmentManager segment_manager;

    Segment segment;
    segment.id = generate_uuid();
    segment.name = "drained_segment";
    segment.size = 1024 * 1024 * 16;
    segment.base = 0x100000000;

    UUID client_id = generate_uuid();

    auto segment_access = segment_manager.getSegmentAccess();
    ASSERT_EQ(segment_access.MountSegment(segment, client_id), ErrorCode::OK);
    ASSERT_EQ(segment_access.SetSegmentStatusByName(segment.name,
                                                    SegmentStatus::DRAINED),
              ErrorCode::OK);

    size_t metrics_dec_capacity = 0;
    ASSERT_EQ(
        segment_access.PrepareUnmountSegment(segment.id, metrics_dec_capacity),
        ErrorCode::OK);
    ASSERT_EQ(segment_access.CommitUnmountSegment(segment.id, client_id,
                                                  metrics_dec_capacity),
              ErrorCode::OK);

    std::vector<Segment> empty_segments;
    std::vector<UUID> empty_client_ids;
    ValidateMountedSegments(segment_manager, empty_segments, empty_client_ids);
}

// ReMountSegmentSuccess:
// 1. Mount a segment A;
// 2. Remount two segments: A and B where A is already mounted and B is a new
// segment. The remount operation return success.
// 3. Only use ValidateMountedSegments function to validate the status of
// SegmentManager. Do not use other interfaces for validation.
TEST_F(SegmentTest, ReMountSegmentSuccess) {
    SegmentManager segment_manager;

    // Create and mount segment A
    Segment segment_a;
    segment_a.id = generate_uuid();
    segment_a.name = "test_segment_a";
    segment_a.size = 1024 * 1024 * 16;
    segment_a.base = 0x100000000;

    UUID client_id = generate_uuid();

    // Get segment access and mount segment A
    auto segment_access = segment_manager.getSegmentAccess();
    ASSERT_EQ(segment_access.MountSegment(segment_a, client_id), ErrorCode::OK);

    // Verify segment A is mounted correctly
    ValidateMountedSegment(segment_manager, segment_a, client_id);

    // Create segment B
    Segment segment_b;
    segment_b.id = generate_uuid();
    segment_b.name = "test_segment_b";
    segment_b.size = 1024 * 1024 * 32;
    segment_b.base = 0x200000000;

    // Remount both segments A and B
    std::vector<Segment> segments_to_remount = {segment_a, segment_b};
    ASSERT_EQ(segment_access.ReMountSegment(segments_to_remount, client_id),
              ErrorCode::OK);

    // Verify both segments are mounted correctly
    std::vector<UUID> client_ids = {client_id, client_id};
    ValidateMountedSegments(segment_manager, segments_to_remount, client_ids);
}

// ReMountUnmountingSegment:
// 1. Mount a segment A;
// 2. PrepareUnmount segment A;
// 3. Remount segment A. The remount operation return
// UNAVAILABLE_IN_CURRENT_STATUS.
// 4. CommitUnmount segment A;
// 5. Only use ValidateMountedSegments function to validate the status of
// SegmentManager. Do not use other interfaces for validation.
TEST_F(SegmentTest, ReMountUnmountingSegment) {
    SegmentManager segment_manager;

    // Create and mount segment A
    Segment segment_a;
    segment_a.id = generate_uuid();
    segment_a.name = "test_segment_a";
    segment_a.size = 1024 * 1024 * 16;
    segment_a.base = 0x100000000;

    UUID client_id = generate_uuid();

    // Get segment access and mount segment A
    auto segment_access = segment_manager.getSegmentAccess();
    ASSERT_EQ(segment_access.MountSegment(segment_a, client_id), ErrorCode::OK);

    // Verify segment A is mounted correctly
    ValidateMountedSegment(segment_manager, segment_a, client_id);

    // Prepare unmount segment A
    size_t metrics_dec_capacity = 0;
    ASSERT_EQ(segment_access.PrepareUnmountSegment(segment_a.id,
                                                   metrics_dec_capacity),
              ErrorCode::OK);

    // Attempt to remount segment A while it's in UNMOUNTING state
    std::vector<Segment> segments_to_remount = {segment_a};
    ASSERT_EQ(segment_access.ReMountSegment(segments_to_remount, client_id),
              ErrorCode::UNAVAILABLE_IN_CURRENT_STATUS);

    // Complete the unmount process
    ASSERT_EQ(segment_access.CommitUnmountSegment(segment_a.id, client_id,
                                                  metrics_dec_capacity),
              ErrorCode::OK);

    // Verify segment is completely unmounted
    std::vector<Segment> empty_segment_vec;
    std::vector<UUID> empty_client_ids_vec;
    ValidateMountedSegments(segment_manager, empty_segment_vec,
                            empty_client_ids_vec);
}

// QuerySegments:
// 1. Create and mount 10 different segments with different names and different
// client ids;
// 2. Test GetClientSegments, verify the return value is correct.
// 3. Test GetAllSegments, verify the return value is correct.
// 4. Test QuerySegments, verify the return value is correct.
TEST_F(SegmentTest, QuerySegments) {
    SegmentManager segment_manager;
    auto segment_access = segment_manager.getSegmentAccess();

    // Create 10 different segments with different names and client IDs
    std::vector<Segment> segments;
    std::vector<UUID> client_ids;
    std::unordered_map<UUID, UUID, boost::hash<UUID>> expected_client_segments;

    for (int i = 0; i < 10; i++) {
        // Create segment
        Segment segment;
        segment.id = generate_uuid();
        segment.name = "test_segment_" + std::to_string(i);
        segment.size = 1024 * 1024 * 16;
        segment.base =
            0x100000000 + (i * 0x100000000);  // Different base addresses

        // Create client ID
        UUID client_id = generate_uuid();

        // Mount segment
        ASSERT_EQ(segment_access.MountSegment(segment, client_id),
                  ErrorCode::OK);

        // Store for verification
        segments.push_back(segment);
        client_ids.push_back(client_id);
        expected_client_segments[client_id] = segment.id;
    }

    // Verify all segments are mounted correctly
    ValidateMountedSegments(segment_manager, segments, client_ids);

    // Test GetClientSegments for each client
    for (size_t i = 0; i < client_ids.size(); i++) {
        std::vector<Segment> client_segments;
        ASSERT_EQ(
            segment_access.GetClientSegments(client_ids[i], client_segments),
            ErrorCode::OK);

        // Verify correct number of segments
        ASSERT_EQ(client_segments.size(), 1);

        // Verify all expected segments are present
        ASSERT_EQ(client_segments[0].id,
                  expected_client_segments[client_ids[i]]);
    }

    // Test GetAllSegments
    std::vector<std::string> all_segments;
    ASSERT_EQ(segment_access.GetAllSegments(all_segments), ErrorCode::OK);

    // Verify correct number of segments
    ASSERT_EQ(all_segments.size(), segments.size());

    // Verify all segment names are present
    for (const auto& segment : segments) {
        ASSERT_NE(
            std::find(all_segments.begin(), all_segments.end(), segment.name),
            all_segments.end());
    }

    // Test QuerySegments for each segment
    for (const auto& segment : segments) {
        size_t used = 0, capacity = 0;
        ASSERT_EQ(segment_access.QuerySegments(segment.name, used, capacity),
                  ErrorCode::OK);

        // Verify capacity matches segment size
        ASSERT_EQ(capacity, segment.size);

        // Verify used space is 0 for newly mounted segments
        ASSERT_EQ(used, 0);
    }

    // Test QuerySegments for non-existent segment
    size_t used = 0, capacity = 0;
    ASSERT_EQ(
        segment_access.QuerySegments("non_existent_segment", used, capacity),
        ErrorCode::SEGMENT_NOT_FOUND);
    ASSERT_EQ(used, 0);
    ASSERT_EQ(capacity, 0);
}

// Mount Local Disk Segment Operations Tests:
TEST_F(SegmentTest, MountLocalDiskSegmentSuccess) {
    SegmentManager segment_manager;
    // Create a valid local disk segment and client ID
    auto segment = std::make_shared<LocalDiskSegment>(true);
    UUID client_id = generate_uuid();

    // Get segment access and attempt to mount
    auto segment_access = segment_manager.getSegmentAccess();
    ASSERT_EQ(segment_access.MountLocalDiskSegment(client_id, true),
              ErrorCode::OK);

    // Verify segment is properly mounted
    ValidateMountedLocalDiskSegments(segment_manager, segment, client_id);
}

// MountLocalDiskSegmentDuplicate Tests:
// 1. MountLocalDiskSegment with the same segment id. The second mount operation
// return SEGMENT_ALREADY_EXISTS.
// 2. MountLocalDiskSegment with different segment id and the same segment name
// should be considered as different segments. Validate the status of
// SegmentManager use ValidateMountedLocalDiskSegments function.
TEST_F(SegmentTest, MountLocalDiskSegmentDuplicate) {
    SegmentManager segment_manager;
    // Create a valid segment and client ID
    auto segment = std::make_shared<LocalDiskSegment>(true);
    UUID client_id = generate_uuid();

    // Get segment access and mount first time
    auto segment_access = segment_manager.getSegmentAccess();
    ASSERT_EQ(segment_access.MountLocalDiskSegment(client_id, true),
              ErrorCode::OK);

    // Verify first mount
    ValidateMountedLocalDiskSegments(segment_manager, segment, client_id);

    // Test duplicate mount - mount the same segment again
    ASSERT_EQ(segment_access.MountLocalDiskSegment(client_id, true),
              ErrorCode::SEGMENT_ALREADY_EXISTS);

    // Verify state remains the same after duplicate mount
    ValidateMountedLocalDiskSegments(segment_manager, segment, client_id);

    // Create a new segment with same name but different ID
    auto segment2 = std::make_shared<LocalDiskSegment>(true);
    UUID client_id2 = generate_uuid();

    // Mount the second segment
    ASSERT_EQ(segment_access.MountLocalDiskSegment(client_id2, true),
              ErrorCode::OK);

    // Verify both segments are mounted correctly
    std::vector<std::shared_ptr<LocalDiskSegment>> segments = {segment,
                                                               segment2};
    std::vector<UUID> client_ids = {client_id, client_id2};
    ValidateMountedLocalDiskSegments(segment_manager, segments, client_ids);
}


TEST_F(SegmentTest, GdsSsdSegmentManagerAllocatesResolvesAndReleases) {
    GdsSsdSegmentManager gds_segment_manager;

    GdsSsdSegment segment;
    segment.id = generate_uuid();
    segment.name = "gds_pool_0";
    segment.base = 0;
    segment.size = 1024 * 1024;
    segment.block_size = 4096;
    segment.allocation_alignment = 4096;
    segment.namespace_id = "nvme-ns-0";

    GdsSsdAccessor accessor;
    accessor.client_host = "host-a";
    accessor.segment_uri = "block:///dev/disk/by-id/mooncake-gds-0";
    accessor.namespace_id = segment.namespace_id;
    accessor.size = segment.size;
    accessor.block_size = segment.block_size;
    accessor.allocation_alignment = segment.allocation_alignment;
    accessor.gpu_device_ids = {0, 1};
    accessor.numa_node = 0;
    accessor.alive = true;
    segment.accessors.push_back(accessor);

    auto access = gds_segment_manager.getGdsSsdSegmentAccess();
    ASSERT_EQ(access.MountSegment(segment), ErrorCode::OK);

    using FailureReason =
        MasterMetricManager::GdsSsdAllocationFailureReason;
    auto& metrics = MasterMetricManager::instance();
    const int64_t success_before = metrics.get_gds_allocation_success();
    const int64_t no_accessor_before =
        metrics.get_gds_allocation_failure(FailureReason::NO_ACCESSOR);
    const int64_t no_segment_before =
        metrics.get_gds_allocation_failure(FailureReason::NO_SEGMENT);
    const int64_t alignment_before =
        metrics.get_gds_allocation_failure(FailureReason::ALIGNMENT);
    const int64_t capacity_before =
        metrics.get_gds_allocation_failure(FailureReason::CAPACITY);

    auto allocation =
        access.AllocateReplica(8192, {"gds_pool_0"}, {}, "host-a");
    ASSERT_TRUE(allocation.has_value());
    EXPECT_EQ(metrics.get_gds_allocation_success(), success_before + 1);
    EXPECT_EQ(metrics.get_segment_gds_pending_size(segment.name), 8192);
    EXPECT_EQ(metrics.get_segment_gds_used_size(segment.name), 0);

    auto inaccessible_allocation =
        access.AllocateReplica(4096, {}, {}, "host-b");
    EXPECT_FALSE(inaccessible_allocation.has_value());
    EXPECT_EQ(inaccessible_allocation.error(), ErrorCode::NO_AVAILABLE_HANDLE);
    EXPECT_EQ(metrics.get_gds_allocation_failure(FailureReason::NO_ACCESSOR),
              no_accessor_before + 1);

    auto missing_segment_allocation = access.AllocateReplica(
        4096, {"missing-gds-segment"}, {}, "host-a");
    EXPECT_FALSE(missing_segment_allocation.has_value());
    EXPECT_EQ(missing_segment_allocation.error(),
              ErrorCode::NO_AVAILABLE_HANDLE);
    EXPECT_EQ(metrics.get_gds_allocation_failure(FailureReason::NO_SEGMENT),
              no_segment_before + 1);

    auto unaligned_allocation =
        access.AllocateReplica(4097, {}, {}, "host-a");
    EXPECT_FALSE(unaligned_allocation.has_value());
    EXPECT_EQ(unaligned_allocation.error(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(metrics.get_gds_allocation_failure(FailureReason::ALIGNMENT),
              alignment_before + 1);

    auto oversized_allocation =
        access.AllocateReplica(2 * 1024 * 1024, {}, {}, "host-a");
    EXPECT_FALSE(oversized_allocation.has_value());
    EXPECT_EQ(oversized_allocation.error(), ErrorCode::NO_AVAILABLE_HANDLE);
    EXPECT_EQ(metrics.get_gds_allocation_failure(FailureReason::CAPACITY),
              capacity_before + 1);
    const GdsSsdReplicaMeta meta = allocation.value();
    EXPECT_EQ(meta.segment_id, segment.id);
    EXPECT_EQ(meta.segment_name, segment.name);
    EXPECT_EQ(meta.object_size, 8192);
    EXPECT_EQ(meta.block_size, segment.block_size);
    EXPECT_EQ(meta.allocation_alignment, segment.allocation_alignment);

    GdsSsdDescriptor descriptor;
    EXPECT_EQ(access.ResolveDescriptor(meta, "host-a", descriptor),
              ErrorCode::OK);
    EXPECT_EQ(descriptor.segment_id, segment.id);
    EXPECT_EQ(descriptor.segment_name, segment.name);
    EXPECT_EQ(descriptor.segment_uri, accessor.segment_uri);
    EXPECT_EQ(descriptor.offset, meta.offset);
    EXPECT_EQ(descriptor.object_size, meta.object_size);
    EXPECT_EQ(access.ResolveDescriptor(meta, "host-b", descriptor),
              ErrorCode::NO_AVAILABLE_HANDLE);

    size_t used = 0;
    size_t capacity = 0;
    ASSERT_EQ(access.QuerySegments(segment.name, used, capacity),
              ErrorCode::OK);
    EXPECT_EQ(used, 8192);
    EXPECT_EQ(capacity, segment.size);

    EXPECT_EQ(access.CommitReplicas({meta}), ErrorCode::OK);
    EXPECT_EQ(metrics.get_segment_gds_pending_size(segment.name), 0);
    EXPECT_EQ(metrics.get_segment_gds_used_size(segment.name), 8192);
    EXPECT_EQ(access.CommitReplicas({meta}), ErrorCode::OK);
    EXPECT_EQ(metrics.get_segment_gds_used_size(segment.name), 8192);

    EXPECT_EQ(access.ReleaseReplica(meta), ErrorCode::OK);
    EXPECT_EQ(metrics.get_segment_gds_pending_size(segment.name), 0);
    EXPECT_EQ(metrics.get_segment_gds_used_size(segment.name), 0);
    ASSERT_EQ(access.QuerySegments(segment.name, used, capacity),
              ErrorCode::OK);
    EXPECT_EQ(used, 0);
    EXPECT_EQ(capacity, segment.size);
}

TEST_F(SegmentTest, GdsSsdMetricsAreReleasedWithManager) {
    const std::string segment_name = "gds_metrics_lifetime";
    auto& metrics = MasterMetricManager::instance();
    ASSERT_EQ(metrics.get_segment_total_gds_capacity(segment_name), 0);
    ASSERT_EQ(metrics.get_segment_allocated_gds_size(segment_name), 0);
    ASSERT_EQ(metrics.get_segment_gds_pending_size(segment_name), 0);
    ASSERT_EQ(metrics.get_segment_gds_used_size(segment_name), 0);

    {
        GdsSsdSegmentManager manager;
        GdsSsdSegment segment;
        segment.id = generate_uuid();
        segment.name = segment_name;
        segment.size = 16 * 1024;
        segment.block_size = 4096;
        segment.allocation_alignment = 4096;
        segment.namespace_id = "nvme-ns-metrics-lifetime";

        GdsSsdAccessor accessor;
        accessor.client_host = "host-a";
        accessor.segment_uri = "block:///dev/mooncake/gds_metrics_lifetime";
        accessor.namespace_id = segment.namespace_id;
        accessor.size = segment.size;
        accessor.block_size = segment.block_size;
        accessor.allocation_alignment = segment.allocation_alignment;
        accessor.alive = true;
        segment.accessors.push_back(accessor);

        {
            auto access = manager.getGdsSsdSegmentAccess();
            ASSERT_EQ(access.MountSegment(segment), ErrorCode::OK);
            auto committed = access.AllocateReplica(4096, {}, {}, "host-a");
            auto pending = access.AllocateReplica(4096, {}, {}, "host-a");
            ASSERT_TRUE(committed.has_value());
            ASSERT_TRUE(pending.has_value());
            ASSERT_EQ(access.CommitReplicas({committed.value()}),
                      ErrorCode::OK);
        }

        EXPECT_EQ(metrics.get_segment_total_gds_capacity(segment_name),
                  segment.size);
        EXPECT_EQ(metrics.get_segment_allocated_gds_size(segment_name), 8192);
        EXPECT_EQ(metrics.get_segment_gds_pending_size(segment_name), 4096);
        EXPECT_EQ(metrics.get_segment_gds_used_size(segment_name), 4096);
    }

    EXPECT_EQ(metrics.get_segment_total_gds_capacity(segment_name), 0);
    EXPECT_EQ(metrics.get_segment_allocated_gds_size(segment_name), 0);
    EXPECT_EQ(metrics.get_segment_gds_pending_size(segment_name), 0);
    EXPECT_EQ(metrics.get_segment_gds_used_size(segment_name), 0);
}

TEST_F(SegmentTest, GdsSsdConcurrentAllocationsDoNotOverlap) {
    constexpr size_t kThreadCount = 16;
    constexpr size_t kAllocationSize = 4096;

    GdsSsdSegmentManager manager;
    GdsSsdSegment segment;
    segment.id = generate_uuid();
    segment.name = "gds_concurrent_allocations";
    segment.size = kThreadCount * kAllocationSize;
    segment.block_size = kAllocationSize;
    segment.allocation_alignment = kAllocationSize;
    segment.namespace_id = "nvme-ns-concurrent";

    GdsSsdAccessor accessor;
    accessor.client_host = "host-a";
    accessor.segment_uri = "block:///dev/mooncake/gds_concurrent";
    accessor.namespace_id = segment.namespace_id;
    accessor.size = segment.size;
    accessor.block_size = segment.block_size;
    accessor.allocation_alignment = segment.allocation_alignment;
    accessor.alive = true;
    segment.accessors.push_back(accessor);

    auto access = manager.getGdsSsdSegmentAccess();
    ASSERT_EQ(access.MountSegment(segment), ErrorCode::OK);

    std::atomic<bool> start{false};
    std::mutex results_mutex;
    std::vector<GdsSsdReplicaMeta> allocations;
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (size_t i = 0; i < kThreadCount; ++i) {
        threads.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            auto allocation =
                access.AllocateReplica(kAllocationSize, {}, {}, "host-a");
            ASSERT_TRUE(allocation.has_value());
            std::lock_guard<std::mutex> lock(results_mutex);
            allocations.push_back(allocation.value());
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& thread : threads) {
        thread.join();
    }

    ASSERT_EQ(allocations.size(), kThreadCount);
    std::unordered_set<uint64_t> offsets;
    for (const auto& allocation : allocations) {
        EXPECT_TRUE(offsets.insert(allocation.offset).second);
        EXPECT_EQ(allocation.offset % kAllocationSize, 0u);
        EXPECT_LE(allocation.offset + allocation.object_size, segment.size);
    }

    ASSERT_EQ(access.CommitReplicas(allocations), ErrorCode::OK);
    for (const auto& allocation : allocations) {
        ASSERT_EQ(access.ReleaseReplica(allocation), ErrorCode::OK);
    }
    auto reused = access.AllocateReplica(kAllocationSize, {}, {}, "host-a");
    ASSERT_TRUE(reused.has_value());
    EXPECT_EQ(reused->offset, 0u);
    EXPECT_EQ(access.ReleaseReplica(reused.value()), ErrorCode::OK);
}

TEST_F(SegmentTest, GdsSsdAccessorLifecycleAndValidation) {
    GdsSsdSegmentManager manager;
    GdsSsdSegment segment;
    segment.id = generate_uuid();
    segment.name = "gds_accessors";
    segment.base = 4096;
    segment.size = 1024 * 1024;
    segment.block_size = 512;
    segment.allocation_alignment = 4096;
    segment.namespace_id = "nvme-ns-accessors";

    auto access = manager.getGdsSsdSegmentAccess();
    ASSERT_EQ(access.MountSegment(segment), ErrorCode::OK);

    GdsSsdAccessor accessor;
    accessor.client_host = "host-a";
    accessor.segment_uri = "block:///dev/disk/by-id/gds-a";
    accessor.namespace_id = segment.namespace_id;
    accessor.size = segment.base + segment.size;
    accessor.block_size = segment.block_size;
    accessor.allocation_alignment = segment.allocation_alignment;
    accessor.gpu_device_ids = {0};
    accessor.alive = true;

    ASSERT_EQ(access.RegisterAccessor(segment.id, accessor), ErrorCode::OK);
    auto found = access.GetAccessor(segment.id, "host-a");
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->segment_uri, accessor.segment_uri);

    GdsSsdAccessor updated = accessor;
    updated.segment_uri = "block:///dev/mooncake/gds-a";
    updated.gpu_device_ids = {1, 2};
    ASSERT_EQ(access.RegisterAccessor(segment.id, updated), ErrorCode::OK);
    found = access.GetAccessor(segment.id, "host-a");
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->segment_uri, updated.segment_uri);
    EXPECT_EQ(found->gpu_device_ids, updated.gpu_device_ids);

    GdsSsdAccessor invalid = accessor;
    invalid.namespace_id = "another-namespace";
    EXPECT_EQ(access.RegisterAccessor(segment.id, invalid),
              ErrorCode::INVALID_PARAMS);
    invalid = accessor;
    invalid.size = segment.base + segment.size - 1;
    EXPECT_EQ(access.RegisterAccessor(segment.id, invalid),
              ErrorCode::INVALID_PARAMS);
    invalid = accessor;
    invalid.block_size = 4096;
    EXPECT_EQ(access.RegisterAccessor(segment.id, invalid),
              ErrorCode::INVALID_PARAMS);
    invalid = accessor;
    invalid.segment_uri = "block://";
    EXPECT_EQ(access.RegisterAccessor(segment.id, invalid),
              ErrorCode::INVALID_PARAMS);

    EXPECT_EQ(access.UnregisterAccessor(segment.id, "host-a"), ErrorCode::OK);
    EXPECT_EQ(access.UnregisterAccessor(segment.id, "host-a"), ErrorCode::OK);
    found = access.GetAccessor(segment.id, "host-a");
    ASSERT_FALSE(found.has_value());
    EXPECT_EQ(found.error(), ErrorCode::NO_AVAILABLE_HANDLE);

    auto allocation = access.AllocateReplica(4096, {}, {}, "host-a");
    ASSERT_FALSE(allocation.has_value());
    EXPECT_EQ(allocation.error(), ErrorCode::NO_AVAILABLE_HANDLE);

    updated.alive = true;
    EXPECT_EQ(access.RegisterAccessor(segment.id, updated), ErrorCode::OK);
    EXPECT_TRUE(access.GetAccessor(segment.id, "host-a").has_value());
}

TEST_F(SegmentTest, GdsSsdRejectsInvalidSegmentsAndReplicaRelease) {
    GdsSsdSegmentManager manager;
    GdsSsdSegment segment;
    segment.id = generate_uuid();
    segment.name = "gds_validation";
    segment.base = 0;
    segment.size = 1024 * 1024;
    segment.block_size = 4096;
    segment.allocation_alignment = 4096;
    segment.namespace_id = "nvme-ns-validation";

    auto access = manager.getGdsSsdSegmentAccess();
    GdsSsdSegment invalid = segment;
    invalid.base = 1;
    EXPECT_EQ(access.MountSegment(invalid), ErrorCode::INVALID_PARAMS);

    invalid = segment;
    invalid.allocation_alignment = 512;
    EXPECT_EQ(access.MountSegment(invalid), ErrorCode::INVALID_PARAMS);

    invalid = segment;
    invalid.base = std::numeric_limits<uint64_t>::max() - 4095;
    invalid.size = 8192;
    EXPECT_EQ(access.MountSegment(invalid), ErrorCode::INVALID_PARAMS);

    GdsSsdAccessor accessor;
    accessor.client_host = "host-a";
    accessor.segment_uri = "block:///dev/disk/by-id/gds-validation";
    accessor.namespace_id = segment.namespace_id;
    accessor.size = segment.size;
    accessor.block_size = segment.block_size;
    accessor.allocation_alignment = segment.allocation_alignment;
    accessor.alive = true;
    segment.accessors.push_back(accessor);
    ASSERT_EQ(access.MountSegment(segment), ErrorCode::OK);

    auto allocation = access.AllocateReplica(4096, {}, {}, "host-a");
    ASSERT_TRUE(allocation.has_value());
    GdsSsdReplicaMeta forged = allocation.value();
    forged.block_size = 512;
    EXPECT_EQ(access.ReleaseReplica(forged), ErrorCode::INVALID_REPLICA);

    size_t used = 0;
    size_t capacity = 0;
    ASSERT_EQ(access.QuerySegments(segment.name, used, capacity),
              ErrorCode::OK);
    EXPECT_EQ(used, 4096);

    EXPECT_EQ(access.ReleaseReplica(allocation.value()), ErrorCode::OK);
    EXPECT_EQ(access.ReleaseReplica(allocation.value()),
              ErrorCode::INVALID_REPLICA);
}

TEST_F(SegmentTest, GdsSsdLocalNvmeOfPathCanBeOpenedByTent) {
#if !defined(__linux__)
    GTEST_SKIP() << "GDS storage path probing requires Linux";
#elif !defined(USE_GDS)
    GTEST_SKIP() << "Rebuild with USE_CUDA=ON and cuFile available to run the "
                    "GDS storage path probe";
#else
    const char* configured_path = std::getenv(kGdsTestPathEnv);
    if (configured_path == nullptr || configured_path[0] == '\0') {
        GTEST_SKIP() << "Set " << kGdsTestPathEnv
                     << " to a locally visible NVMe-oF block device, regular "
                        "file, or mounted directory";
    }

    std::string prepare_error;
    auto target = GdsTestTarget::Prepare(configured_path, prepare_error);
    ASSERT_NE(target, nullptr) << prepare_error;

    auto config = std::make_shared<tent::Config>();
    config->set("metadata_type", "p2p");
    config->set("metadata_servers", "");
    config->set("rpc_server_hostname", "127.0.0.1");
    config->set("rpc_server_port", "0");
    config->set("local_segment_name", "gds_path_probe");
    config->set("metrics/enabled", false);
    config->set("transports/tcp/enable", false);
    config->set("transports/shm/enable", false);
    config->set("transports/rdma/enable", false);
    config->set("transports/io_uring/enable", false);
    config->set("transports/nvlink/enable", false);
    config->set("transports/mnnvl/enable", false);
    config->set("transports/gds/enable", true);

    tent::TransferEngine engine(config);
    ASSERT_TRUE(engine.available()) << "failed to initialize TENT";

    tent::SegmentID segment_id = 0;
    auto status = engine.openSegment(segment_id, target->uri);
    ASSERT_TRUE(status.ok()) << status.ToString();

    tent::SegmentInfo info;
    status = engine.getSegmentInfo(segment_id, info);
    ASSERT_TRUE(status.ok())
        << "TENT could not resolve/open " << target->uri << ": "
        << status.ToString();
    ASSERT_EQ(info.type, tent::SegmentInfo::File);
    ASSERT_EQ(info.buffers.size(), 1u);
    if (target->is_block_device) {
        EXPECT_GT(info.buffers[0].length, 0u);
    } else {
        EXPECT_EQ(info.buffers[0].length, target->expected_size);
    }

    std::string cufile_error;
    EXPECT_TRUE(RegisterGdsStorageHandle(target->path, cufile_error))
        << cufile_error;

    status = engine.closeSegment(segment_id);
    EXPECT_TRUE(status.ok()) << status.ToString();
#endif
}
}  // namespace mooncake
