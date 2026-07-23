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

#include "tent/runtime/transfer_engine_impl.h"
#include "tent/runtime/control_plane.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>

#include "tent/common/config.h"
#include "tent/common/status.h"
#include "tent/runtime/control_plane.h"
#include "tent/runtime/segment.h"
#include "tent/runtime/segment_tracker.h"
#include "tent/runtime/progress_worker.h"
#include "tent/runtime/proxy_manager.h"
#include "tent/runtime/transport.h"
#include "tent/runtime/topology.h"
#include "tent/runtime/platform.h"
#include "tent/runtime/slab.h"
#include "tent/common/utils/ip.h"
#include "tent/common/utils/random.h"
#include "tent/metrics/tent_metrics.h"
#include "tent/metrics/config_loader.h"

namespace mooncake {
namespace tent {

namespace {
constexpr uint8_t kRedisMaxDbIndex = 255;
constexpr uint8_t kRedisDefaultDbIndex = 0;
// The default must absorb normal overlapping vLLM BatchPut bursts. The
// backlog is still finite, but only an explicitly tighter deployment setting
// should turn routine 126/192-owner submits into synchronous rejection.
constexpr size_t kDefaultMinWaitingOwners = 4096;
constexpr size_t kDefaultMinWaitingBytes = size_t{8} << 30;
constexpr size_t kRuntimeQueueSummarySampleCapacity = 4096;
constexpr auto kRuntimeQueueSummaryInterval = std::chrono::seconds(1);

size_t saturatingMultiply(size_t value, size_t multiplier) {
    if (value > std::numeric_limits<size_t>::max() / multiplier) {
        return std::numeric_limits<size_t>::max();
    }
    return value * multiplier;
}

size_t saturatingAdd(size_t lhs, size_t rhs) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return std::numeric_limits<size_t>::max();
    }
    return lhs + rhs;
}

Status checkedDispatchOwnerSum(size_t read_owners, size_t write_owners,
                               size_t& total) {
    if (write_owners >
        std::numeric_limits<size_t>::max() - read_owners) {
        return Status::InvalidArgument(
            "runtime queue directional dispatch window overflow" LOC_MARK);
    }
    total = read_owners + write_owners;
    return Status::OK();
}

double runtimeQueueNearestRankP99(std::vector<double> samples) {
    if (samples.empty()) return 0.0;
    // Samples are capped at 4096, so this integer nearest-rank calculation
    // cannot overflow.
    const size_t rank = (samples.size() * 99 + 99) / 100;
    const size_t index = rank - 1;
    std::nth_element(samples.begin(), samples.begin() + index,
                     samples.end());
    return samples[index];
}
}  // namespace

struct Batch {
    Batch() : max_size(0) { sub_batch.fill(nullptr); }

    ~Batch() {}

    std::array<Transport::SubBatchRef, kSupportedTransportTypes> sub_batch;
    std::vector<TaskInfo> task_list;
    size_t max_size;
    size_t runtime_refs{0};
    bool free_requested{false};
    uint64_t queue_token{0};

    struct SubmitHook {
        size_t start_task_id{0};
        size_t end_task_id{0};  // [start, end)
        Notification notifi;
        bool fired{false};
        std::unordered_set<SegmentID> targets;
    };
    std::vector<SubmitHook> submit_hooks;
};

struct PreservedTentConfigOverrides {
    std::optional<std::string> metadata_type;
    std::optional<std::string> metadata_servers;
    std::optional<std::string> local_segment_name;
    std::optional<std::string> rpc_server_hostname;
    std::optional<json> rpc_server_port;
};

template <typename T>
std::optional<T> captureExplicitConfigValue(const Config& config,
                                            const std::string& key,
                                            const T& default_value) {
    if (!config.contains(key)) {
        return std::nullopt;
    }
    return config.get<T>(key, default_value);
}

std::optional<long long> tryParseConfigIntString(const std::string& value) {
    try {
        size_t parsed_chars = 0;
        long long parsed_value = std::stoll(value, &parsed_chars);
        if (parsed_chars == value.size()) {
            return parsed_value;
        }
    } catch (...) {
    }
    return std::nullopt;
}

Status validateRpcServerPortValue(long long value, const std::string& source,
                                  uint16_t& port) {
    constexpr long long kMinPort = 0;
    constexpr long long kMaxPort = std::numeric_limits<uint16_t>::max();
    if (value < kMinPort || value > kMaxPort) {
        return Status::InvalidArgument("Invalid rpc_server_port '" + source +
                                       "', expected value in range [0, " +
                                       std::to_string(kMaxPort) + "]" +
                                       LOC_MARK);
    }

    port = static_cast<uint16_t>(value);
    return Status::OK();
}

Status getRpcServerPortFromConfig(const Config& config, uint16_t default_value,
                                  uint16_t& port) {
    constexpr const char* kKey = "rpc_server_port";
    if (!config.contains(kKey)) {
        port = default_value;
        return Status::OK();
    }

    json raw_value = config.get<json>(kKey, json());
    if (raw_value.is_number_integer() || raw_value.is_number_unsigned()) {
        long long numeric_value = raw_value.get<long long>();
        return validateRpcServerPortValue(numeric_value,
                                          std::to_string(numeric_value), port);
    }

    if (raw_value.is_string()) {
        auto string_value = raw_value.get<std::string>();
        auto parsed_value = tryParseConfigIntString(string_value);
        if (!parsed_value.has_value()) {
            return Status::InvalidArgument(
                "Invalid rpc_server_port '" + string_value +
                "', expected integer in range [0, 65535]" LOC_MARK);
        }
        return validateRpcServerPortValue(*parsed_value, string_value, port);
    }

    return Status::InvalidArgument(
        "rpc_server_port must be an integer or integer string" LOC_MARK);
}

PreservedTentConfigOverrides captureExplicitTransferEngineConfig(
    const Config& config) {
    PreservedTentConfigOverrides preserved;
    preserved.metadata_type =
        captureExplicitConfigValue(config, "metadata_type", std::string());
    preserved.metadata_servers =
        captureExplicitConfigValue(config, "metadata_servers", std::string());
    preserved.local_segment_name =
        captureExplicitConfigValue(config, "local_segment_name", std::string());
    preserved.rpc_server_hostname = captureExplicitConfigValue(
        config, "rpc_server_hostname", std::string());
    preserved.rpc_server_port =
        captureExplicitConfigValue(config, "rpc_server_port", json());
    return preserved;
}

template <typename T>
void restoreExplicitConfigValue(Config& config, const std::string& key,
                                const std::optional<T>& value) {
    if (value.has_value()) {
        config.set(key, *value);
    }
}

void restoreExplicitTransferEngineConfig(
    Config& config, const PreservedTentConfigOverrides& preserved) {
    restoreExplicitConfigValue(config, "metadata_type",
                               preserved.metadata_type);
    restoreExplicitConfigValue(config, "metadata_servers",
                               preserved.metadata_servers);
    restoreExplicitConfigValue(config, "local_segment_name",
                               preserved.local_segment_name);
    restoreExplicitConfigValue(config, "rpc_server_hostname",
                               preserved.rpc_server_hostname);
    restoreExplicitConfigValue(config, "rpc_server_port",
                               preserved.rpc_server_port);
}

TransferEngineImpl::TransferEngineImpl()
    : conf_(std::make_shared<Config>()),
      available_(false),
      port_(0),
      ipv6_(false),
      merge_requests_(true) {
    ConfigHelper().loadFromEnv(*conf_);
    auto status = construct();
    if (!status.ok()) {
        LOG(ERROR) << "Failed to construct Transfer Engine instance: "
                   << status.ToString();
    } else {
        available_ = true;
    }
}

TransferEngineImpl::TransferEngineImpl(std::shared_ptr<Config> conf)
    : conf_(conf),
      available_(false),
      port_(0),
      ipv6_(false),
      merge_requests_(true) {
    auto preserved = captureExplicitTransferEngineConfig(*conf_);
    // Allow MC_TENT_CONF to supply shared defaults while keeping the caller's
    // explicit metadata identity intact.
    ConfigHelper().loadFromEnv(*conf_);
    restoreExplicitTransferEngineConfig(*conf_, preserved);
    auto status = construct();
    if (!status.ok()) {
        LOG(ERROR) << "Failed to construct Transfer Engine instance: "
                   << status.ToString();
    } else {
        available_ = true;
    }
}

TransferEngineImpl::~TransferEngineImpl() { deconstruct(); }

std::string randomSegmentName() {
    std::string name = "segment_noname_";
    for (int i = 0; i < 8; ++i) name += 'a' + SimpleRandom::Get().next(26);
    return name;
}

void setLogLevel(const std::string level) {
    if (level == "info")
        FLAGS_minloglevel = google::INFO;
    else if (level == "warning")
        FLAGS_minloglevel = google::WARNING;
    else if (level == "error")
        FLAGS_minloglevel = google::ERROR;
}

static std::string readIdentityFile(const char* path) {
    std::ifstream file(path);
    if (!file) return "";
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    if (!content.empty() && content.back() == '\n') content.pop_back();
    return content;
}

std::string getMachineID() {
    const std::string boot_id =
        readIdentityFile("/proc/sys/kernel/random/boot_id");
    const std::string machine_id = readIdentityFile("/etc/machine-id");

    if (!boot_id.empty() && !machine_id.empty()) {
        return boot_id + ":" + machine_id;
    }

    if (!boot_id.empty()) return boot_id;
    if (!machine_id.empty()) return machine_id;

    std::string content = "undefined_machine_";
    for (int i = 0; i < 16; ++i) content += 'a' + SimpleRandom::Get().next(26);
    LOG(WARNING) << "TENT getMachineID source=fallback value=" << content;
    return content;
}

Status TransferEngineImpl::setupLocalSegment() {
    auto& manager = metadata_->segmentManager();
    auto segment = manager.getLocal();
    segment->name = local_segment_name_;
    segment->type = SegmentType::Memory;
    segment->machine_id = getMachineID();
    segment->rpc_server_addr = buildIpAddrWithPort(hostname_, port_, ipv6_);
    auto& detail = std::get<MemorySegmentDesc>(segment->detail);
    detail.topology = *(topology_.get());
    local_segment_tracker_ = std::make_unique<SegmentTracker>(segment);
    return manager.synchronizeLocal();
}

Status TransferEngineImpl::construct() {
    auto metadata_type = conf_->get("metadata_type", "p2p");
    auto metadata_servers = conf_->get("metadata_servers", "");

    setLogLevel(conf_->get("log_level", "info"));
    hostname_ = conf_->get("rpc_server_hostname", "");
    local_segment_name_ = conf_->get("local_segment_name", "");
    CHECK_STATUS(getRpcServerPortFromConfig(*conf_, 0, port_));
    merge_requests_ = conf_->get("merge_requests", true);
    max_failover_attempts_ = conf_->get("max_failover_attempts", 3);
    enable_auto_failover_on_poll_ =
        conf_->get("enable_auto_failover_on_poll", true);
    enable_progress_worker_ = conf_->get("enable_progress_worker", false);
    runtime_queue_config_.enabled = conf_->get("enable_runtime_queue", false);
    if (runtime_queue_config_.enabled) enable_progress_worker_ = true;
    const bool gds_enabled = conf_->get("transports/gds/enable", false);
    const size_t default_max_outstanding_owners = gds_enabled ? 64UL : 1024UL;
    const size_t default_max_outstanding_bytes =
        gds_enabled ? (256UL << 20) : (1UL << 30);
    runtime_queue_config_.limits.max_outstanding_owners =
        conf_->get("runtime_queue/max_outstanding_owners",
                   default_max_outstanding_owners);
    runtime_queue_config_.limits.max_outstanding_bytes =
        conf_->get("runtime_queue/max_outstanding_bytes",
                   default_max_outstanding_bytes);
    runtime_queue_config_.limits.staging_owner_reserve =
        conf_->get("runtime_queue/staging_owner_reserve", 0UL);
    runtime_queue_config_.limits.staging_byte_reserve =
        conf_->get("runtime_queue/staging_byte_reserve", 0UL);
    const size_t configured_gds_read_workers =
        conf_->get("transports/gds/read_worker_threads", 16UL);
    const size_t configured_gds_read_inflight =
        conf_->get("transports/gds/max_inflight_reads", 16UL);
    const size_t default_gds_read_inflight =
        std::min(configured_gds_read_workers, configured_gds_read_inflight);
    const size_t configured_gds_write_workers =
        conf_->get("transports/gds/write_worker_threads", 4UL);
    const size_t configured_gds_write_inflight =
        conf_->get("transports/gds/max_inflight_writes", 4UL);
    const size_t default_gds_write_inflight =
        std::min(configured_gds_write_workers, configured_gds_write_inflight);
    size_t default_max_dispatch_owners = 64;
    if (gds_enabled) {
        CHECK_STATUS(checkedDispatchOwnerSum(
            default_gds_read_inflight, default_gds_write_inflight,
            default_max_dispatch_owners));
    }
    runtime_queue_config_.max_dispatch_owners =
        conf_->get("runtime_queue/max_dispatch_owners",
                   default_max_dispatch_owners);
    runtime_queue_config_.max_dispatch_bytes =
        conf_->get("runtime_queue/max_dispatch_bytes", 64UL << 20);
    const size_t default_max_waiting_owners = std::max(
        kDefaultMinWaitingOwners,
        saturatingMultiply(
            runtime_queue_config_.limits.max_outstanding_owners, 8));
    const size_t default_max_waiting_bytes = std::max(
        kDefaultMinWaitingBytes,
        saturatingMultiply(
            runtime_queue_config_.limits.max_outstanding_bytes, 4));
    runtime_queue_config_.max_waiting_owners =
        conf_->get("runtime_queue/max_waiting_owners",
                   default_max_waiting_owners);
    runtime_queue_config_.max_waiting_bytes =
        conf_->get("runtime_queue/max_waiting_bytes",
                   default_max_waiting_bytes);
    runtime_queue_config_.max_dispatch_read_owners = conf_->get(
        "runtime_queue/max_dispatch_read_owners",
        gds_enabled ? std::min(default_gds_read_inflight,
                              runtime_queue_config_.max_dispatch_owners)
                    : runtime_queue_config_.max_dispatch_owners);
    runtime_queue_config_.max_dispatch_write_owners = conf_->get(
        "runtime_queue/max_dispatch_write_owners",
        gds_enabled ? std::min(default_gds_write_inflight,
                              runtime_queue_config_.max_dispatch_owners)
                    : runtime_queue_config_.max_dispatch_owners);
    const size_t default_progress_fallback_interval_us =
        gds_enabled ? 1000UL : 50000UL;
    runtime_queue_config_.progress_fallback_interval =
        std::chrono::microseconds(
            conf_->get("runtime_queue/progress_fallback_interval_us",
                       default_progress_fallback_interval_us));
    if (runtime_queue_config_.enabled &&
        (runtime_queue_config_.limits.max_outstanding_owners == 0 ||
         runtime_queue_config_.limits.max_outstanding_bytes == 0 ||
         runtime_queue_config_.max_dispatch_owners == 0 ||
         runtime_queue_config_.max_dispatch_bytes == 0 ||
         runtime_queue_config_.max_waiting_owners == 0 ||
         runtime_queue_config_.max_waiting_bytes == 0 ||
         runtime_queue_config_.max_dispatch_read_owners == 0 ||
         runtime_queue_config_.max_dispatch_write_owners == 0)) {
        return Status::InvalidArgument(
            "runtime queue limits, waiting backlog, and dispatch window must "
            "be non-zero"
            LOC_MARK);
    }
    if (runtime_queue_config_.enabled && gds_enabled) {
        size_t configured_directional_dispatch_owners = 0;
        CHECK_STATUS(checkedDispatchOwnerSum(
            runtime_queue_config_.max_dispatch_read_owners,
            runtime_queue_config_.max_dispatch_write_owners,
            configured_directional_dispatch_owners));
        if (runtime_queue_config_.max_dispatch_owners <
            configured_directional_dispatch_owners) {
            return Status::InvalidArgument(
                "runtime queue global dispatch owner window must cover the "
                "sum of GDS READ and WRITE direction windows" LOC_MARK);
        }
    }
    if (runtime_queue_config_.enabled &&
        (runtime_queue_config_.max_dispatch_owners >
             runtime_queue_config_.limits.max_outstanding_owners ||
         runtime_queue_config_.max_dispatch_bytes >
             runtime_queue_config_.limits.max_outstanding_bytes ||
         runtime_queue_config_.max_dispatch_read_owners >
             runtime_queue_config_.max_dispatch_owners ||
         runtime_queue_config_.max_dispatch_write_owners >
             runtime_queue_config_.max_dispatch_owners)) {
        return Status::InvalidArgument(
            "runtime queue dispatch window exceeds global/outstanding limits"
            LOC_MARK);
    }
    if (runtime_queue_config_.enabled &&
        (runtime_queue_config_.limits.staging_owner_reserve >
             runtime_queue_config_.limits.max_outstanding_owners ||
         runtime_queue_config_.limits.staging_byte_reserve >
             runtime_queue_config_.limits.max_outstanding_bytes)) {
        return Status::InvalidArgument(
            "runtime queue staging reserve exceeds outstanding limits"
            LOC_MARK);
    }
    if (runtime_queue_config_.enabled) {
        LOG(INFO) << "Runtime queue config: max_dispatch_owners="
                  << runtime_queue_config_.max_dispatch_owners
                  << ", max_dispatch_bytes="
                  << runtime_queue_config_.max_dispatch_bytes
                  << ", max_waiting_owners="
                  << runtime_queue_config_.max_waiting_owners
                  << ", max_waiting_bytes="
                  << runtime_queue_config_.max_waiting_bytes
                  << ", max_dispatch_read_owners="
                  << runtime_queue_config_.max_dispatch_read_owners
                  << ", max_dispatch_write_owners="
                  << runtime_queue_config_.max_dispatch_write_owners
                  << ", max_outstanding_owners="
                  << runtime_queue_config_.limits.max_outstanding_owners
                  << ", max_outstanding_bytes="
                  << runtime_queue_config_.limits.max_outstanding_bytes
                  << ", progress_fallback_interval_us="
                  << runtime_queue_config_.progress_fallback_interval.count()
                  << ", gds_enabled=" << gds_enabled;
    } else if (gds_enabled) {
        LOG(WARNING)
            << "GDS transport is enabled while the TENT runtime queue is "
               "disabled; large vLLM KV batches will be submitted to cuFile "
               "without an owner/byte admission window.";
    }
    runtime_queue_ = std::make_unique<LocalTransferAdmissionQueue>(
        runtime_queue_config_.limits);
    for (auto& summary : runtime_queue_summary_) {
        summary.queue_wait_us.reserve(
            kRuntimeQueueSummarySampleCapacity);
        summary.total_latency_us.reserve(
            kRuntimeQueueSummarySampleCapacity);
    }
    runtime_queue_summary_started_at_ =
        std::chrono::steady_clock::now();
    if (!hostname_.empty())
        CHECK_STATUS(checkLocalIpAddress(hostname_, ipv6_));
    else
        CHECK_STATUS(discoverLocalIpAddress(hostname_, ipv6_));

    topology_ = std::make_shared<Topology>();
    auto loader = &Platform::getLoader(conf_);
    CHECK_STATUS(topology_->discover({loader}));

    metadata_ =
        std::make_shared<ControlService>(metadata_type, metadata_servers, this);

    CHECK_STATUS(metadata_->start(port_, ipv6_));

    if (metadata_type == "p2p")
        local_segment_name_ = buildIpAddrWithPort(hostname_, port_, ipv6_);
    else if (local_segment_name_.empty())
        local_segment_name_ = randomSegmentName();

    CHECK_STATUS(setupLocalSegment());

    // Initialize transport selector
    transport_selector_ = std::make_unique<TransportSelector>(conf_);
    transport_selector_->setTopology(topology_);

    // Check if legacy mode is enabled (use original getTransportType logic)
    bool legacy_mode = conf_->get("use_legacy_transport_selection", false);
    transport_selector_->setLegacyMode(legacy_mode);
    if (legacy_mode) {
        LOG(INFO) << "Using legacy transport selection (original logic)";
    }

    CHECK_STATUS(loadTransports());

    std::string transport_string;
    for (auto& transport : transport_list_) {
        if (transport) {
            auto status = transport->install(local_segment_name_, metadata_,
                                             topology_, conf_);
            if (!status.ok()) {
                LOG(WARNING) << "Transport " << transport->getName()
                             << " skipped: " << status.ToString();
                transport = nullptr;
                continue;
            }
            transport_string += transport->getName();
            transport_string += " ";
        }
    }

    staging_proxy_ = std::make_unique<ProxyManager>(this);

    if (enable_progress_worker_) {
        progress_worker_ = std::make_unique<ProgressWorker>(
            this, runtime_queue_config_.enabled
                      ? runtime_queue_config_.progress_fallback_interval
                      : std::chrono::microseconds(0));
        progress_worker_->start();
    }

    // Initialize and start Metrics system
    auto metrics_config = MetricsConfigLoader::loadWithDefaults(conf_.get());
    if (metrics_config.enabled) {
        std::string validation_error;
        if (!MetricsConfigLoader::validateConfig(metrics_config,
                                                 &validation_error)) {
            LOG(WARNING) << "Invalid metrics configuration: "
                         << validation_error << ", Metrics system disabled";
        } else {
            // Initialize metrics
            auto status = TentMetrics::instance().initialize(metrics_config);
            if (!status.ok()) {
                LOG(WARNING) << "Failed to initialize TENT metrics: "
                             << status.ToString();
            } else {
                LOG(INFO) << "TENT Metrics system initialized";
            }
        }
    } else {
        LOG(INFO) << "Metrics system disabled by configuration";
    }

    if (conf_->get("verbose", false)) {
        LOG(INFO) << "========== Transfer Engine Parameters ==========";
        LOG(INFO) << " - Segment Name:       " << local_segment_name_;
        LOG(INFO) << " - RPC Server Address: "
                  << buildIpAddrWithPort(hostname_, port_, ipv6_);
        LOG(INFO) << " - Metadata Type:      " << metadata_type;
        LOG(INFO) << " - Metadata Servers:   " << metadata_servers;
        LOG(INFO) << " - Loaded Transports:  " << transport_string;
        LOG(INFO) << "================================================";
    } else {
        LOG(INFO) << "Transfer Engine " << local_segment_name_
                  << " started successfully";
    }

    return Status::OK();
}

Status TransferEngineImpl::deconstruct() {
    // Metrics cleanup is handled automatically by TentMetrics destructor

    // Stop the progress worker first so it cannot race with batch teardown
    // below (it dereferences BatchID into Batch* via progressBatch). Keep the
    // object alive until transports are destroyed: completion paths may still
    // issue a final no-op wake while their workers are joining.
    if (progress_worker_) {
        progress_worker_->stop();
    }

    // Destroy staging_proxy_ first: its destructor calls back into
    // unregisterLocalMemory/freeLocalMemory, which require
    // local_segment_tracker_ and metadata_ to be alive.
    staging_proxy_.reset();

    if (local_segment_tracker_) {
        local_segment_tracker_->forEach([&](BufferDesc& desc) -> Status {
            for (size_t type = 0; type < kSupportedTransportTypes; ++type) {
                if (transport_list_[type])
                    transport_list_[type]->removeMemoryBuffer(desc);
            }
            return Status::OK();
        });
    }

    // Free all batches BEFORE destroying transports, so that
    // freeSubBatch() can properly return SubBatch/Slice objects
    // to the global Slab/allocator instances used by the transports.
    //
    // Safety note: freeSubBatch() only performs Slab deallocation and
    // does not access transport-internal state (workers, connections).
    // Callers must ensure no transfers are in-flight before calling
    // deconstruct().
    {
        std::lock_guard<std::recursive_mutex> lk(progress_mutex_);
        std::unordered_set<Batch*> released_batches;
        auto release_batch = [&](Batch* batch) {
            if (!released_batches.insert(batch).second) return;
            for (size_t type = 0; type < kSupportedTransportTypes; ++type) {
                auto& transport = transport_list_[type];
                auto& sub_batch = batch->sub_batch[type];
                if (!transport || !sub_batch) continue;
                transport->freeSubBatch(sub_batch);
            }
            Slab<Batch>::Get().deallocate(batch);
        };
        for (auto& batch : batch_set_.active) release_batch(batch);
        for (auto& batch : batch_set_.freelist) release_batch(batch);
        batch_set_.active.clear();
        batch_set_.freelist.clear();
        alive_batches_.clear();
    }

    // Now safe to destroy transports (workers join here)
    for (auto& transport : transport_list_) transport.reset();
    progress_worker_.reset();
    local_segment_tracker_.reset();
    if (metadata_) {
        metadata_->segmentManager().deleteLocal();
        metadata_.reset();
    }
    return Status::OK();
}

const std::string TransferEngineImpl::getSegmentName() const {
    return local_segment_name_;
}

const std::string TransferEngineImpl::getRpcServerAddress() const {
    return hostname_;
}

uint16_t TransferEngineImpl::getRpcServerPort() const { return port_; }

Status TransferEngineImpl::exportLocalSegment(std::string& shared_handle) {
    return Status::NotImplemented(
        "exportLocalSegment not implemented" LOC_MARK);
}

Status TransferEngineImpl::importRemoteSegment(
    SegmentID& handle, const std::string& shared_handle) {
    return Status::NotImplemented(
        "importRemoteSegment not implemented" LOC_MARK);
}

Status TransferEngineImpl::openSegment(SegmentID& handle,
                                       const std::string& segment_name) {
    if (segment_name.empty() || segment_name == local_segment_name_) {
        handle = LOCAL_SEGMENT_ID;
        return Status::OK();
    }
    return metadata_->segmentManager().openRemote(handle, segment_name);
}

Status TransferEngineImpl::closeSegment(SegmentID handle) {
    if (handle == LOCAL_SEGMENT_ID) return Status::OK();
    return metadata_->segmentManager().closeRemote(handle);
}

Status TransferEngineImpl::getSegmentInfo(SegmentID handle, SegmentInfo& info) {
    SegmentDesc* desc = nullptr;
    if (handle == LOCAL_SEGMENT_ID) {
        desc = metadata_->segmentManager().getLocal().get();
    } else {
        CHECK_STATUS(metadata_->segmentManager().getRemoteCached(desc, handle));
    }
    if (desc->type == SegmentType::File) {
        info.type = SegmentInfo::File;
        auto& detail = std::get<FileSegmentDesc>(desc->detail);
        for (auto& entry : detail.buffers) {
            info.buffers.emplace_back(
                SegmentInfo::Buffer{.base = entry.offset,
                                    .length = entry.length,
                                    .location = kWildcardLocation});
        }
    } else if (desc->type == SegmentType::Block) {
        info.type = SegmentInfo::File;
        auto& detail = std::get<BlockSegmentDesc>(desc->detail);
        info.buffers.emplace_back(
            SegmentInfo::Buffer{.base = detail.offset,
                                .length = detail.length,
                                .location = kWildcardLocation});
    } else {
        info.type = SegmentInfo::Memory;
        auto& detail = std::get<MemorySegmentDesc>(desc->detail);
        for (auto& entry : detail.buffers) {
            if (entry.internal) continue;
            info.buffers.emplace_back(
                SegmentInfo::Buffer{.base = (uint64_t)entry.addr,
                                    .length = entry.length,
                                    .location = entry.location});
        }
    }
    return Status::OK();
}

Status TransferEngineImpl::allocateLocalMemory(void** addr, size_t size,
                                               Location location) {
    return allocateLocalMemory(addr, size, location, false);
}

Status TransferEngineImpl::allocateLocalMemory(void** addr, size_t size,
                                               Location location,
                                               bool internal) {
    // Decide transport type based on location
    MemoryOptions options;
    options.location = location;
    options.internal = internal;
    if (location == kWildcardLocation ||
        LocationParser(location).type() == "cpu") {
        if (transport_list_[SHM])
            options.type = SHM;
        else if (transport_list_[RDMA])
            options.type = RDMA;
        else
            options.type = TCP;
    } else {
        if (transport_list_[MNNVL])
            options.type = MNNVL;
        else if (transport_list_[RDMA])
            options.type = RDMA;
        else
            options.type = TCP;
    }
    return allocateLocalMemory(addr, size, options);
}

Status TransferEngineImpl::allocateLocalMemory(void** addr, size_t size,
                                               MemoryOptions& options) {
    if (options.type == UNSPEC) {
        if (transport_list_[RDMA])
            options.type = RDMA;
        else if (transport_list_[TCP])
            options.type = TCP;
        else
            return Status::InvalidArgument(
                "Not supported type in memory options" LOC_MARK);
    }
    auto& transport = transport_list_[options.type];
    if (!transport)
        return Status::InvalidArgument(
            "Not supported type in memory options" LOC_MARK);
    CHECK_STATUS(transport->allocateLocalMemory(addr, size, options));
    std::lock_guard<std::mutex> lock(mutex_);
    AllocatedMemory entry{.addr = *addr,
                          .size = size,
                          .transport = transport.get(),
                          .options = options};
    allocated_memory_.push_back(entry);
    return Status::OK();
}

Status TransferEngineImpl::freeLocalMemory(void* addr) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = allocated_memory_.begin(); it != allocated_memory_.end();
         ++it) {
        if (it->addr == addr) {
            auto status = it->transport->freeLocalMemory(addr, it->size);
            allocated_memory_.erase(it);
            return status;
        }
    }
    return Status::InvalidArgument("Address region not registered" LOC_MARK);
}

Status TransferEngineImpl::registerLocalMemory(void* addr, size_t size,
                                               Permission permission) {
    MemoryOptions options;
    options.perm = permission;
    return registerLocalMemory({addr}, {size}, options);
}

Status TransferEngineImpl::registerLocalMemory(std::vector<void*> addr_list,
                                               std::vector<size_t> size_list,
                                               Permission permission) {
    MemoryOptions options;
    options.perm = permission;
    return registerLocalMemory(addr_list, size_list, options);
}

std::vector<TransportType> TransferEngineImpl::getSupportedTransports(
    TransportType request_type) {
    std::vector<TransportType> result;
    if (request_type != UNSPEC) {
        if (request_type >= 0 && request_type < kSupportedTransportTypes &&
            transport_list_[request_type]) {
            result.push_back(request_type);
        }
        return result;
    }
    if (transport_list_[MNNVL]) result.push_back(MNNVL);
    if (transport_list_[NVLINK]) result.push_back(NVLINK);
    if (transport_list_[RDMA]) result.push_back(RDMA);
    if (transport_list_[SUNRISE_LINK]) result.push_back(SUNRISE_LINK);
    if (transport_list_[AscendDirect]) result.push_back(AscendDirect);
    if (transport_list_[SHM]) result.push_back(SHM);
    if (transport_list_[TCP]) result.push_back(TCP);
    if (transport_list_[GDS]) result.push_back(GDS);
    return result;
}

Status TransferEngineImpl::registerLocalMemory(std::vector<void*> addr_list,
                                               std::vector<size_t> size_list,
                                               MemoryOptions& options) {
    if (addr_list.size() != size_list.size()) {
        return Status::InvalidArgument(
            "Mismatched addresses and sizes in registerLocalMemory" LOC_MARK);
    }
    auto transports = getSupportedTransports(options.type);
    if (transports.empty()) {
        return Status::InvalidArgument(
            "No available transport for registerLocalMemory" LOC_MARK);
    }

    // Build BufferDescs: warm-up → NUMA probe → fill location
    std::vector<BufferDesc> desc_list;
    desc_list.reserve(addr_list.size());
    for (size_t i = 0; i < addr_list.size(); ++i) {
        BufferDesc desc;
        desc.addr = (uint64_t)addr_list[i];
        desc.length = size_list[i];

        // MR warm-up: pin pages via temp ibv_reg_mr, benefits both
        // subsequent RDMA registration and NUMA probing
        bool pages_pinned = false;
        for (auto type : transports) {
            if (transport_list_[type]->warmupMemory(addr_list[i],
                                                    size_list[i])) {
                pages_pinned = true;
                break;
            }
        }

        // NUMA probe: skip prefault if warm-up already pinned pages
        auto entries = Platform::getLoader().getLocation(
            addr_list[i], size_list[i], pages_pinned);
        if (entries.size() == 1) {
            desc.location = entries[0].location;
        } else {
            desc.location = entries[0].location;
            desc.regions = coalesceRegions(entries);
        }
        desc.ref_count = 1;
        if (options.location != kWildcardLocation)
            desc.location = options.location;
        if (options.internal) desc.internal = options.internal;
        desc_list.push_back(std::move(desc));
    }

    auto status = local_segment_tracker_->addInBatch(
        desc_list, [&](std::vector<BufferDesc>& descs) -> Status {
            for (auto type : transports) {
                auto s = transport_list_[type]->addMemoryBuffer(descs, options);
                if (!s.ok()) LOG(WARNING) << s.ToString();
            }
            return Status::OK();
        });
    if (!status.ok()) return status;
    // Synchronize local segment to metadata server so remote peers can see the
    // new buffers
    return metadata_->segmentManager().synchronizeLocal();
}

// WARNING: before exiting TE, make sure that all local memory are
// unregistered, otherwise the CUDA may halt!
Status TransferEngineImpl::unregisterLocalMemory(void* addr, size_t size) {
    bool removed = false;
    auto status = local_segment_tracker_->remove(
        (uint64_t)addr, size, [&](BufferDesc& desc) -> Status {
            removed = true;
            for (auto type : desc.transports) {
                auto status = transport_list_[type]->removeMemoryBuffer(desc);
                if (!status.ok()) LOG(WARNING) << status.ToString();
            }
            return Status::OK();
        });
    if (!status.ok()) return status;
    if (!removed) return Status::OK();
    return metadata_->segmentManager().synchronizeLocal();
}

Status TransferEngineImpl::unregisterLocalMemory(
    std::vector<void*> addr_list, std::vector<size_t> size_list) {
    if (!size_list.empty() && addr_list.size() != size_list.size()) {
        return Status::InvalidArgument(
            "Mismatched addresses and sizes in unregisterLocalMemory" LOC_MARK);
    }
    bool removed_any = false;
    for (size_t i = 0; i < addr_list.size(); ++i) {
        bool removed = false;
        auto status = local_segment_tracker_->remove(
            (uint64_t)addr_list[i], size_list.empty() ? 0 : size_list[i],
            [&](BufferDesc& desc) -> Status {
                removed = true;
                for (auto type : desc.transports) {
                    auto s = transport_list_[type]->removeMemoryBuffer(desc);
                    if (!s.ok()) LOG(WARNING) << s.ToString();
                }
                return Status::OK();
            });
        if (!status.ok()) return status;
        if (removed) removed_any = true;
    }
    if (!removed_any) return Status::OK();
    return metadata_->segmentManager().synchronizeLocal();
}

BatchID TransferEngineImpl::allocateBatch(size_t batch_size) {
    Batch* batch = Slab<Batch>::Get().allocate();
    if (!batch) return (BatchID)0;
    batch->max_size = batch_size;
    BatchID batch_id = (BatchID)batch;
    std::lock_guard<std::recursive_mutex> lk(progress_mutex_);
    batch_set_.active.insert(batch);
    alive_batches_.insert(batch_id);
    return batch_id;
}

Status TransferEngineImpl::freeBatch(BatchID batch_id) {
    if (!batch_id) return Status::InvalidArgument("Invalid batch ID" LOC_MARK);
    Batch* batch = (Batch*)(batch_id);
    std::lock_guard<std::recursive_mutex> lk(progress_mutex_);
    if (!alive_batches_.count(batch_id))
        return Status::InvalidArgument("Batch is not alive" LOC_MARK);
    if (runtime_queue_config_.enabled && batch->queue_token != 0) {
        auto retire_status = retireQueueForBatch(batch);
        if (!retire_status.ok() && !retire_status.IsInvalidEntry()) {
            return retire_status;
        }
    }
    if (batch->free_requested) {
        CHECK_STATUS(lazyFreeBatch());
        return Status::OK();
    }
    batch->free_requested = true;
    batch_set_.freelist.push_back(batch);
    lazyFreeBatch();
    return Status::OK();
}

Status TransferEngineImpl::lazyFreeBatch() {
    std::lock_guard<std::recursive_mutex> lk(progress_mutex_);
    for (auto it = batch_set_.freelist.begin();
         it != batch_set_.freelist.end();) {
        auto& batch = *it;
        if (batch->runtime_refs > 0) {
            it++;
            continue;
        }
        TransferStatus overall_status;
        CHECK_STATUS(getTransferStatus((BatchID)batch, overall_status));
        if (overall_status.s == PENDING) {
            it++;
            continue;
        }
        if (runtime_queue_config_.enabled && batch->queue_token != 0) {
            CHECK_STATUS(retireQueueForBatch(batch));
        }
        for (size_t type = 0; type < kSupportedTransportTypes; ++type) {
            auto& transport = transport_list_[type];
            auto& sub_batch = batch->sub_batch[type];
            if (transport && sub_batch) transport->freeSubBatch(sub_batch);
        }
        batch_set_.active.erase(batch);
        alive_batches_.erase((BatchID)batch);
        Slab<Batch>::Get().deallocate(batch);
        it = batch_set_.freelist.erase(it);
    }
    return Status::OK();
}

Status TransferEngineImpl::retainBatch(BatchID batch_id, Batch*& batch) {
    if (!batch_id) return Status::InvalidArgument("Invalid batch ID" LOC_MARK);
    std::lock_guard<std::recursive_mutex> lk(progress_mutex_);
    if (!alive_batches_.count(batch_id)) {
        return Status::InvalidArgument("Batch is not alive" LOC_MARK);
    }
    batch = (Batch*)batch_id;
    if (batch->free_requested) {
        return Status::InvalidArgument("Batch is being freed" LOC_MARK);
    }
    ++batch->runtime_refs;
    return Status::OK();
}

Status TransferEngineImpl::releaseBatch(Batch* batch) {
    if (!batch) return Status::InvalidArgument("Invalid batch" LOC_MARK);
    std::lock_guard<std::recursive_mutex> lk(progress_mutex_);
    if (batch->runtime_refs == 0) {
        return Status::InternalError("Batch runtime ref underflow" LOC_MARK);
    }
    --batch->runtime_refs;
    if (batch->runtime_refs == 0 && batch->free_requested) {
        CHECK_STATUS(lazyFreeBatch());
    }
    return Status::OK();
}

class TransferEngineImpl::BatchRef {
   public:
    BatchRef(TransferEngineImpl& engine, Batch* batch)
        : engine_(engine), batch_(batch) {}

    ~BatchRef() {
        if (!batch_) return;
        auto status = engine_.releaseBatch(batch_);
        if (!status.ok()) {
            LOG(WARNING) << "failed to release batch ref: "
                         << status.ToString();
        }
    }

    BatchRef(const BatchRef&) = delete;
    BatchRef& operator=(const BatchRef&) = delete;

    Batch* get() const { return batch_; }

    Status release() {
        if (!batch_) return Status::OK();
        auto status = engine_.releaseBatch(batch_);
        batch_ = nullptr;
        return status;
    }

   private:
    TransferEngineImpl& engine_;
    Batch* batch_{nullptr};
};

static bool isGpuType(MemoryType t) {
    return t == MTYPE_CUDA || t == MTYPE_ROCM;
}

static bool checkAvailability(const std::shared_ptr<Transport>& xport,
                              MemoryType local) {
    if (local == MTYPE_CPU) return xport && xport->capabilities().dram_to_file;
    if (isGpuType(local)) return xport && xport->capabilities().gpu_to_file;
    return false;
}

static bool checkAvailability(const std::shared_ptr<Transport>& xport,
                              MemoryType local, MemoryType remote) {
    if (local == MTYPE_CPU && remote == MTYPE_CPU)
        return xport && xport->capabilities().dram_to_dram;
    if (isGpuType(local) && isGpuType(remote))
        return xport && xport->capabilities().gpu_to_gpu;
    if (local == MTYPE_CPU && isGpuType(remote))
        return xport && xport->capabilities().dram_to_gpu;
    if (isGpuType(local) && remote == MTYPE_CPU)
        return xport && xport->capabilities().gpu_to_dram;
    return false;
}

static MemoryType getTypeEnum(const std::string& type) {
    if (type == "cpu" || type == "*") return MTYPE_CPU;
    if (type == "cuda") return MTYPE_CUDA;
    if (type == "npu") return MTYPE_CUDA;
    if (type == "rocm") return MTYPE_ROCM;
    return MTYPE_UNKNOWN;
}

Status TransferEngineImpl::validateTransportHint(const Request& req,
                                                 size_t request_index) {
    if (req.transport_hint == UNSPEC) return Status::OK();
    if ((int)req.transport_hint < 0 ||
        (int)req.transport_hint >= kSupportedTransportTypes) {
        return Status::InvalidArgument(
            "transport_hint out of range for request[" +
            std::to_string(request_index) + "]" LOC_MARK);
    }
    if (!transport_list_[req.transport_hint]) {
        return Status::InvalidArgument(
            "transport_hint=" +
            TransportSelector::transportTypeName(req.transport_hint) +
            " is not enabled in config (request[" +
            std::to_string(request_index) + "])" LOC_MARK);
    }
    return Status::OK();
}

SelectionResult TransferEngineImpl::getTransportType(const Request& request,
                                                     int transport_index) {
    SegmentDesc* desc;
    if (request.target_id == LOCAL_SEGMENT_ID) {
        desc = metadata_->segmentManager().getLocal().get();
    } else {
        auto status = metadata_->segmentManager().getRemoteCached(
            desc, request.target_id);
        if (!status.ok()) return SelectionResult{};
    }
    auto local_mtype = Platform::getLoader().getMemoryType(request.source);

    const TransportType hint = request.transport_hint;

    // Legacy mode: use original logic (before TransportSelector)
    if (transport_selector_ && transport_selector_->isLegacyMode()) {
        SelectionResult result;
        std::vector<TransportType> raw;
        if (desc->type == SegmentType::File ||
            desc->type == SegmentType::Block) {
            if (checkAvailability(transport_list_[GDS], local_mtype))
                raw.push_back(GDS);
            if (desc->type == SegmentType::File &&
                checkAvailability(transport_list_[IOURING], local_mtype))
                raw.push_back(IOURING);
        } else {
            auto entry =
                desc->findBuffer(request.target_offset, request.length);
            if (entry) {
                bool same_machine = (request.target_id == LOCAL_SEGMENT_ID);
                if (!same_machine) {
                    auto local_desc = metadata_->segmentManager().getLocal();
                    same_machine = local_desc && !desc->machine_id.empty() &&
                                   !local_desc->machine_id.empty() &&
                                   desc->machine_id == local_desc->machine_id;
                }
                auto remote_mtype =
                    getTypeEnum(LocationParser(entry->location).type());
                for (auto type : entry->transports) {
                    if ((type == NVLINK || type == SHM) && !same_machine)
                        continue;
                    if (checkAvailability(transport_list_[type], local_mtype,
                                          remote_mtype)) {
                        raw.push_back(type);
                    }
                }
            }
        }

        auto candidates = TransportSelector::reorderWithHint(raw, hint);
        if (!candidates) {
            return result;  // UNSPEC: hint not authorized for this req
        }
        if (transport_index >= 0 &&
            (size_t)transport_index < candidates->size()) {
            result.transport = (*candidates)[transport_index];
        }
        return result;
    }

    // Selector mode: build ctx, then defer everything to
    // TransportSelector::select().
    SelectionContext ctx;
    ctx.transfer_size = request.length;
    ctx.priority_level =
        request.priority;  // Use request priority for selection
    ctx.policy_name = request.policy_name;  // Optional: bind to specific policy

    if (desc->type == SegmentType::File || desc->type == SegmentType::Block) {
        // Storage segments use selector policies with empty buffer_transports.
        ctx.segment_type = desc->type;
        ctx.same_machine = true;
        ctx.local_memory_type = local_mtype;
        ctx.remote_memory_type = MTYPE_CPU;
        ctx.buffer_transports = nullptr;  // Empty - use policy priority
    } else {
        // Memory segment
        auto entry = desc->findBuffer(request.target_offset, request.length);
        if (!entry) return SelectionResult{};
        bool same_machine =
            (desc->machine_id ==
             metadata_->segmentManager().getLocal()->machine_id);
        auto remote_mtype = getTypeEnum(LocationParser(entry->location).type());

        ctx.segment_type = SegmentType::Memory;
        ctx.same_machine = same_machine;
        ctx.local_memory_type = local_mtype;
        ctx.remote_memory_type = remote_mtype;
        ctx.buffer_transports = &entry->transports;
    }

    return transport_selector_->select(ctx, transport_list_, transport_index,
                                       hint);
}

static const char* transportTypeName(TransportType type) {
    switch (type) {
        case UNSPEC:
            return "UNSPEC";
        case RDMA:
            return "RDMA";
        case MNNVL:
            return "MNNVL";
        case SHM:
            return "SHM";
        case NVLINK:
            return "NVLINK";
        case GDS:
            return "GDS";
        case IOURING:
            return "IOURING";
        case TCP:
            return "TCP";
        case AscendDirect:
            return "AscendDirect";
        case SUNRISE_LINK:
            return "SUNRISE_LINK";
    }
    return "UNKNOWN";
}

std::string printRequest(const Request& request) {
    std::stringstream ss;
    ss << "opcode " << request.opcode << " source " << request.source
       << " target_id " << request.target_id << " target_offset "
       << (void*)request.target_offset << " length " << request.length
       << " transport_hint " << transportTypeName(request.transport_hint);
    return ss.str();
}

struct BufferKey {
    uint64_t addr{0};
    uint64_t length{0};

    bool operator==(const BufferKey&) const = default;
};

struct RequestBoundaryInfo {
    std::optional<BufferKey> source_key;
    std::optional<BufferKey> target_key;
};

struct MergeResult {
    std::vector<Request> request_list;
    std::map<size_t, size_t> task_lookup;
};

struct TransferEngineImpl::PreparedSubmit {
    struct Task {
        size_t merged_task_index{0};
        size_t task_id{0};
    };

    struct Owner {
        size_t owner_task_id{0};
        bool has_owner_task_id{false};
        std::vector<size_t> derived_task_ids;
        Request request{};
        SelectionResult route{};
        bool staging{false};
        std::vector<std::string> staging_params;
    };

    std::chrono::steady_clock::time_point submit_time{};
    std::vector<Task> tasks;
    std::vector<Owner> owners;
};

namespace {

bool tryAddUint64(uint64_t lhs, uint64_t rhs, uint64_t& out) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) return false;
    out = lhs + rhs;
    return true;
}

MergeResult makePassThroughMergeResult(const std::vector<Request>& requests) {
    MergeResult result;
    result.request_list.reserve(requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        result.request_list.push_back(requests[i]);
        result.task_lookup[i] = i;
    }
    return result;
}

uint64_t requestSourceAddr(const Request& request) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(request.source));
}

}  // namespace

MergeResult mergeRequests(const std::vector<Request>& requests,
                          const std::vector<RequestBoundaryInfo>& boundaries,
                          bool do_merge) {
    if (requests.empty()) return {};
    if (!do_merge || boundaries.size() != requests.size()) {
        return makePassThroughMergeResult(requests);
    }

    struct Item {
        Request req;
        RequestBoundaryInfo boundary;
        size_t orig_idx;
    };

    std::vector<Item> items;
    items.reserve(requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        items.push_back({requests[i], boundaries[i], i});
    }

    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        if (a.req.opcode != b.req.opcode) return a.req.opcode < b.req.opcode;
        if (a.req.target_id != b.req.target_id)
            return a.req.target_id < b.req.target_id;
        if (a.req.transport_hint != b.req.transport_hint)
            return a.req.transport_hint < b.req.transport_hint;
        if (a.req.target_offset != b.req.target_offset)
            return a.req.target_offset < b.req.target_offset;
        return requestSourceAddr(a.req) < requestSourceAddr(b.req);
    });

    auto can_merge = [](const Item& last, const Item& curr) {
        if (last.req.opcode != curr.req.opcode ||
            last.req.target_id != curr.req.target_id) {
            return false;
        }
        // Mixed transport_hint inside one batch must not be merged.
        if (last.req.transport_hint != curr.req.transport_hint) {
            return false;
        }
        if (!last.boundary.source_key || !curr.boundary.source_key ||
            !last.boundary.target_key || !curr.boundary.target_key) {
            return false;
        }
        if (last.boundary.source_key != curr.boundary.source_key ||
            last.boundary.target_key != curr.boundary.target_key) {
            return false;
        }
        if (curr.req.length >
            std::numeric_limits<size_t>::max() - last.req.length) {
            return false;
        }

        uint64_t last_source_end = 0;
        uint64_t last_target_end = 0;
        if (!tryAddUint64(requestSourceAddr(last.req), last.req.length,
                          last_source_end) ||
            !tryAddUint64(last.req.target_offset, last.req.length,
                          last_target_end)) {
            return false;
        }
        return last_source_end == requestSourceAddr(curr.req) &&
               last_target_end == curr.req.target_offset;
    };

    std::vector<Item> merged_items;
    merged_items.reserve(items.size());

    MergeResult result;
    for (const auto& item : items) {
        if (merged_items.empty() || !can_merge(merged_items.back(), item)) {
            merged_items.push_back(item);
        } else {
            merged_items.back().req.length += item.req.length;
        }
        result.task_lookup[item.orig_idx] = merged_items.size() - 1;
    }

    result.request_list.reserve(merged_items.size());
    for (const auto& item : merged_items) {
        result.request_list.push_back(item.req);
    }
    return result;
}

std::optional<BufferKey> toBufferKey(BufferDesc* buffer) {
    if (!buffer) return std::nullopt;
    return BufferKey{buffer->addr, buffer->length};
}

std::vector<RequestBoundaryInfo> resolveRequestBoundaries(
    ControlService* metadata, const std::vector<Request>& requests) {
    // Group requests by target_id so withCachedSegment fires at most once per
    // peer.
    std::vector<RequestBoundaryInfo> boundaries(requests.size());
    auto* local_desc = metadata->segmentManager().getLocal().get();

    if (local_desc) {
        for (size_t i = 0; i < requests.size(); ++i) {
            auto source_addr = static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(requests[i].source));
            boundaries[i].source_key = toBufferKey(
                local_desc->findBuffer(source_addr, requests[i].length));
        }
    }

    std::unordered_map<SegmentID, std::vector<size_t>> by_target;
    for (size_t i = 0; i < requests.size(); ++i) {
        by_target[requests[i].target_id].push_back(i);
    }

    for (auto& [target_id, idxs] : by_target) {
        metadata->segmentManager().withCachedSegment(
            target_id, [&](SegmentDesc* target_desc) {
                if (target_desc->type != SegmentType::Memory) {
                    return Status::OK();
                }
                bool any_missing = false;
                for (size_t i : idxs) {
                    const auto& r = requests[i];
                    auto* buffer =
                        target_desc->findBuffer(r.target_offset, r.length);
                    if (!buffer) {
                        any_missing = true;
                        boundaries[i].target_key = std::nullopt;
                    } else {
                        boundaries[i].target_key = toBufferKey(buffer);
                    }
                }
                // Invariant: when this lambda returns NeedsRefreshCache, all
                // writes it made in this pass are wiped before it returns.
                // Reason: withCachedSegment will invalidate the cache and try
                // ONE refetch; if that refetch fails (e.g. peer RPC down) the
                // retry pass never runs, and any tentative writes from this
                // (stale) pass would leak downstream into mergeRequests. By
                // clearing here we leave a clean nullopt state for the group,
                // and the retry pass (if it does run) repopulates from the
                // fresh desc so the wipe is harmless.
                if (any_missing) {
                    for (size_t i : idxs) {
                        boundaries[i].target_key = std::nullopt;
                    }
                    return Status::NeedsRefreshCache(
                        "Requested address is not in registered "
                        "buffer" LOC_MARK);
                }
                return Status::OK();
            });
    }
    return boundaries;
}

void TransferEngineImpl::findStagingPolicy(const Request& request,
                                           std::vector<std::string>& policy) {
    if (request.target_id == LOCAL_SEGMENT_ID) return;

    SegmentDesc* desc = nullptr;
    BufferDesc* entry = nullptr;
    auto status = metadata_->segmentManager().withCachedSegment(
        request.target_id, [&](SegmentDesc* segment) {
            desc = segment;
            if (desc->type != SegmentType::Memory) {
                return Status::OK();
            }
            entry = desc->findBuffer(request.target_offset, request.length);
            if (!entry)
                return Status::NeedsRefreshCache(
                    "Requested address is not in registered buffer" LOC_MARK);
            return Status::OK();
        });

    if (!status.ok() || !entry) return;
    auto local =
        Platform::getLoader().getLocation(request.source, 1)[0].location;
    auto remote = entry->location;
    auto local_mtype = getTypeEnum(LocationParser(local).type());
    auto remote_mtype = getTypeEnum(LocationParser(remote).type());
    auto server_addr = desc->rpc_server_addr;
    policy.clear();
    // case 1: rdma without gpu direct
    if (transport_list_[RDMA] && transport_list_[NVLINK]) {
        auto& xport = transport_list_[RDMA];
        auto& caps = xport->capabilities();
        if (local_mtype == MTYPE_CUDA && remote_mtype == MTYPE_CUDA &&
            !caps.gpu_to_gpu) {
            policy.push_back(server_addr);
            policy.push_back(topology_->findNearMem(local));
            policy.push_back(desc->getMemory().topology.findNearMem(remote));
        } else if (local_mtype == MTYPE_CUDA && remote_mtype == MTYPE_CPU &&
                   !caps.gpu_to_dram) {
            policy.push_back(server_addr);
            policy.push_back(topology_->findNearMem(local));
            policy.push_back("");  // no remote stage
        } else if (local_mtype == MTYPE_CPU && remote_mtype == MTYPE_CUDA &&
                   !caps.dram_to_gpu) {
            policy.push_back(server_addr);
            policy.push_back("");  // no local stage
            policy.push_back(desc->getMemory().topology.findNearMem(remote));
        }
    }
    // case 2: pure mnnvl
    if (transport_list_[MNNVL] && transport_list_[NVLINK]) {
        auto& xport = transport_list_[RDMA];
        auto& caps = xport->capabilities();
        if (local_mtype == MTYPE_CPU && remote_mtype == MTYPE_CPU &&
            !caps.dram_to_dram) {
            policy.push_back(server_addr);
            policy.push_back(topology_->findNearMem(local, Topology::MEM_CUDA));
            policy.push_back("");  // remote stage
        } else if (local_mtype == MTYPE_CUDA && remote_mtype == MTYPE_CPU &&
                   !caps.gpu_to_dram) {
            policy.push_back(server_addr);
            policy.push_back("");  // no local stage
            policy.push_back(desc->getMemory().topology.findNearMem(
                remote, Topology::MEM_CUDA));
        }
    }
}

SelectionResult TransferEngineImpl::resolveTransport(const Request& req,
                                                     int transport_index,
                                                     bool invalidate_on_fail) {
    auto result = getTransportType(req, transport_index);
    if (result.transport == UNSPEC && invalidate_on_fail) {
        metadata_->segmentManager().invalidateRemote(req.target_id);
        result = getTransportType(req, transport_index);
    }
    return result;
}

Status TransferEngineImpl::prepareSubmit(
    Batch* batch, const std::vector<Request>& request_list,
    PreparedSubmit& prepared) {
    if (!batch) return Status::InvalidArgument("Invalid batch" LOC_MARK);
    for (size_t i = 0; i < request_list.size(); ++i) {
        auto st = validateTransportHint(request_list[i], i);
        if (!st.ok()) return st;
    }

    prepared = PreparedSubmit{};
    const size_t start_task_id = batch->task_list.size();
    prepared.submit_time = std::chrono::steady_clock::now();
    auto merge_boundaries =
        merge_requests_
            ? resolveRequestBoundaries(metadata_.get(), request_list)
            : std::vector<RequestBoundaryInfo>{};
    auto merged =
        mergeRequests(request_list, merge_boundaries, merge_requests_);

    prepared.owners.reserve(merged.request_list.size());
    for (const auto& request : merged.request_list) {
        PreparedSubmit::Owner owner;
        owner.request = request;
        owner.route = resolveTransport(owner.request, 0);
        if (owner.route.transport == TCP) {
            findStagingPolicy(owner.request, owner.staging_params);
            owner.staging = !owner.staging_params.empty() && staging_proxy_;
        }
        prepared.owners.push_back(std::move(owner));
    }

    prepared.tasks.reserve(merged.task_lookup.size());
    for (const auto& kv : merged.task_lookup) {
        const size_t public_task_index = kv.first;
        const size_t merged_task_index = kv.second;
        const size_t task_id = start_task_id + public_task_index;
        auto& owner = prepared.owners[merged_task_index];
        if (!owner.has_owner_task_id) {
            owner.owner_task_id = task_id;
            owner.has_owner_task_id = true;
        } else {
            owner.derived_task_ids.push_back(task_id);
        }
        prepared.tasks.push_back({merged_task_index, task_id});
    }

    size_t gds_input_requests = 0;
    size_t gds_merged_requests = 0;
    size_t gds_bytes = 0;
    for (const auto& owner : prepared.owners) {
        if (owner.route.transport != GDS) continue;
        ++gds_merged_requests;
        gds_input_requests += 1 + owner.derived_task_ids.size();
        gds_bytes += owner.request.length;
    }
    if (gds_input_requests > 0) {
        TentMetrics::instance().recordGdsCoalescing(
            gds_input_requests, gds_merged_requests, gds_bytes);
    }
    return Status::OK();
}

uint64_t TransferEngineImpl::nextBatchToken() { return next_batch_token_++; }

void TransferEngineImpl::attachProgressNotifier(
    Batch* batch, Transport::SubBatchRef sub_batch) {
    if (!batch || !sub_batch) return;
    sub_batch->progress_batch_id = (BatchID)batch;
    sub_batch->notify_progress = [this](BatchID batch_id) {
        notifyBatchMaybeReady(batch_id);
    };
}

Status TransferEngineImpl::commitPreparedSubmit(
    Batch* batch, const PreparedSubmit& prepared) {
    if (!batch) return Status::InvalidArgument("Invalid batch" LOC_MARK);

    std::vector<Request> classified_request_list[kSupportedTransportTypes];
    std::vector<size_t> task_id_list[kSupportedTransportTypes];
    std::unordered_map<size_t, TaskInfo> merged_task_id_map;

    batch->task_list.insert(batch->task_list.end(), prepared.tasks.size(),
                            TaskInfo{});

    std::unordered_map<TransportType, size_t> next_sub_task_id;
    for (const auto& task_plan : prepared.tasks) {
        size_t task_id = task_plan.task_id;
        size_t merged_task_id = task_plan.merged_task_index;
        auto& task = batch->task_list[task_id];
        const auto& owner = prepared.owners[merged_task_id];
        auto& merged_request = owner.request;
        if (merged_task_id_map.count(merged_task_id)) {
            task = merged_task_id_map[merged_task_id];
            task.derived = true;
            if (task.type != UNSPEC) task_id_list[task.type].push_back(task_id);
            continue;
        }

        task.failover_count = 0;
        task.xport_priority = 0;
        task.status = PENDING;
        task.request = merged_request;
        task.staging = false;
        task.start_time =
            prepared.submit_time;  // Record start time for latency tracking
        task.type = owner.route.transport;
        task.device_mask = owner.route.device_mask;
        if (task.type == UNSPEC) {
            LOG(WARNING) << "Unable to find registered buffer for request: "
                         << printRequest(merged_request);
            merged_task_id_map[merged_task_id] = task;
            continue;
        }

        if (owner.staging) {
            task.staging = true;
            staging_proxy_->submit(&task, (BatchID)batch, owner.staging_params);
            continue;
        }

        if (!batch->sub_batch[task.type]) {
            auto& transport = transport_list_[task.type];
            auto status = transport->allocateSubBatch(
                batch->sub_batch[task.type], batch->max_size);
            if (!status.ok()) {
                LOG(WARNING) << "Failed to allocate SubBatch " << task.type
                             << ":" << status.ToString();
                merged_task_id_map[merged_task_id] = task;
                continue;
            }
            attachProgressNotifier(batch, batch->sub_batch[task.type]);
        }

        if (!next_sub_task_id.count(task.type))
            next_sub_task_id[task.type] = batch->sub_batch[task.type]->size();
        size_t sub_task_id = next_sub_task_id[task.type];
        next_sub_task_id[task.type]++;

        classified_request_list[task.type].push_back(merged_request);
        task.sub_task_id = sub_task_id;
        task.derived = false;
        task_id_list[task.type].push_back(task_id);
        merged_task_id_map[merged_task_id] = task;
    }

    for (size_t type = 0; type < kSupportedTransportTypes; ++type) {
        if (classified_request_list[type].empty()) continue;
        auto& transport = transport_list_[type];
        auto& sub_batch = batch->sub_batch[type];

        // Set device_mask on SubBatch for RDMA transport
        if (type == RDMA && !task_id_list[type].empty()) {
            // Use the device_mask from the first task (we assume all tasks in
            // this batch should have the same policy)
            sub_batch->device_mask =
                batch->task_list[task_id_list[type][0]].device_mask;
        }

        auto status = transport->submitTransferTasks(
            sub_batch, classified_request_list[type]);
        if (!status.ok()) {
            // LOG(WARNING) << "Failed to submit SubBatch " << type << ":"
            //              << status.ToString();
            for (auto& task_id : task_id_list[type])
                batch->task_list[task_id].type = UNSPEC;
        }
    }

    return Status::OK();
}

Status TransferEngineImpl::enqueuePreparedSubmit(Batch* batch,
                                                 const PreparedSubmit& prepared,
                                                 QueueOwnerKind owner_kind) {
    std::lock_guard<std::recursive_mutex> lk(progress_mutex_);
    if (prepared.tasks.empty()) return Status::OK();
    if (prepared.tasks.size() > batch->max_size - batch->task_list.size()) {
        return Status::TooManyRequests(
            "batch public task capacity exceeded" LOC_MARK);
    }

    const size_t owner_capacity =
        owner_kind == QueueOwnerKind::User
            ? runtime_queue_config_.limits.max_outstanding_owners -
                  runtime_queue_config_.limits.staging_owner_reserve
            : runtime_queue_config_.limits.max_outstanding_owners;
    const size_t byte_capacity =
        owner_kind == QueueOwnerKind::User
            ? runtime_queue_config_.limits.max_outstanding_bytes -
                  runtime_queue_config_.limits.staging_byte_reserve
            : runtime_queue_config_.limits.max_outstanding_bytes;
    if (owner_capacity == 0 || byte_capacity == 0) {
        return Status::TooManyRequests(
            "runtime queue has no capacity for this owner class" LOC_MARK);
    }

    size_t submit_bytes = 0;
    const uint64_t batch_token =
        batch->queue_token != 0 ? batch->queue_token : nextBatchToken();
    for (const auto& owner : prepared.owners) {
        if (owner.request.length > runtime_queue_config_.max_dispatch_bytes) {
            return Status::TooManyRequests(
                "request exceeds runtime queue dispatch byte window" LOC_MARK);
        }
        if (owner.request.length > byte_capacity) {
            return Status::TooManyRequests(
                "request exceeds runtime queue admission byte capacity"
                LOC_MARK);
        }
        if (owner.request.length >
            std::numeric_limits<size_t>::max() - submit_bytes) {
            return Status::InvalidArgument(
                "runtime queue submit byte charge overflow" LOC_MARK);
        }
        submit_bytes += owner.request.length;
    }
    if (prepared.owners.size() > owner_capacity ||
        submit_bytes > byte_capacity) {
        LOG_EVERY_N(INFO, 64)
            << "Runtime queue streaming oversized submit: owners="
            << prepared.owners.size() << ", bytes=" << submit_bytes
            << ", admission_owner_capacity=" << owner_capacity
            << ", admission_byte_capacity=" << byte_capacity;
    }
    if (prepared.owners.size() >
            std::numeric_limits<size_t>::max() -
                admission_waiting_owners_ ||
        submit_bytes > std::numeric_limits<size_t>::max() -
                           admission_waiting_bytes_) {
        return Status::InvalidArgument(
            "runtime queue waiting backlog overflow" LOC_MARK);
    }
    if (admission_waiting_owners_ >
            runtime_queue_config_.max_waiting_owners ||
        admission_waiting_bytes_ >
            runtime_queue_config_.max_waiting_bytes ||
        prepared.owners.size() >
            runtime_queue_config_.max_waiting_owners -
                admission_waiting_owners_ ||
        submit_bytes > runtime_queue_config_.max_waiting_bytes -
                           admission_waiting_bytes_) {
        return Status::TooManyRequests(
            "runtime queue pre-admission waiting high watermark exceeded"
            LOC_MARK);
    }

    batch->queue_token = batch_token;
    batch->task_list.insert(batch->task_list.end(), prepared.tasks.size(),
                            TaskInfo{});
    for (const auto& task_plan : prepared.tasks) {
        auto& task = batch->task_list[task_plan.task_id];
        const auto& owner = prepared.owners[task_plan.merged_task_index];
        task.failover_count = 0;
        task.xport_priority = 0;
        task.status = PENDING;
        task.request = owner.request;
        task.staging = false;
        task.start_time = prepared.submit_time;
        task.type = UNSPEC;
        task.sub_task_id = -1;
        task.device_mask = owner.route.device_mask;
        task.derived = task_plan.task_id != owner.owner_task_id;
        task.runtime_admission_waiting = true;
    }

    admission_waiting_owners_ += prepared.owners.size();
    admission_waiting_bytes_ += submit_bytes;
    for (const auto& owner : prepared.owners) {
        AdmissionWaitingOwner waiting;
        waiting.batch = batch;
        waiting.batch_token = batch_token;
        waiting.owner_task_id = owner.owner_task_id;
        waiting.derived_task_ids = owner.derived_task_ids;
        waiting.byte_charge = owner.request.length;
        waiting.enqueue_time = prepared.submit_time;
        waiting.initial_transport = owner.route.transport;
        waiting.kind = owner_kind;
        admission_waiting_queue_.push_back(std::move(waiting));

        if (owner.route.transport != GDS) continue;
        if (owner.request.opcode == Request::READ) {
            ++admission_waiting_gds_reads_;
        } else {
            ++admission_waiting_gds_writes_;
        }
    }
    updateRuntimeQueueMetrics();
    return Status::OK();
}

Status TransferEngineImpl::admitWaitingOwners() {
    while (!admission_waiting_queue_.empty()) {
        const auto find_gds_opcode = [&](Request::OpCode opcode) {
            return std::find_if(
                admission_waiting_queue_.begin(),
                admission_waiting_queue_.end(),
                [&](const AdmissionWaitingOwner& waiting) {
                    return waiting.initial_transport == GDS &&
                           waiting.batch &&
                           waiting.owner_task_id <
                               waiting.batch->task_list.size() &&
                           waiting.batch->task_list[waiting.owner_task_id]
                                   .request.opcode == opcode;
                });
        };
        const auto non_gds_it = std::find_if(
            admission_waiting_queue_.begin(),
            admission_waiting_queue_.end(),
            [](const AdmissionWaitingOwner& waiting) {
                return waiting.initial_transport != GDS;
            });
        auto read_it = find_gds_opcode(Request::READ);
        auto write_it = find_gds_opcode(Request::WRITE);
        const bool has_read = read_it != admission_waiting_queue_.end();
        const bool has_write = write_it != admission_waiting_queue_.end();
        const bool choose_read =
            has_read &&
            (!has_write || consecutive_admission_read_owners_ <
                               kMaxConsecutiveAdmissionReads);
        auto selected_it = choose_read ? read_it : write_it;
        if (selected_it == admission_waiting_queue_.end() &&
            non_gds_it == admission_waiting_queue_.end()) {
            return Status::InternalError(
                "runtime admission backlog contains no dispatchable owner"
                LOC_MARK);
        }
        if (non_gds_it != admission_waiting_queue_.end() &&
            (selected_it == admission_waiting_queue_.end() ||
             std::distance(admission_waiting_queue_.begin(), non_gds_it) <
                 std::distance(admission_waiting_queue_.begin(),
                               selected_it))) {
            selected_it = non_gds_it;
        }

        const auto waiting = *selected_it;
        if (!waiting.batch ||
            waiting.owner_task_id >= waiting.batch->task_list.size()) {
            return Status::InternalError(
                "runtime admission owner metadata is stale" LOC_MARK);
        }
        const auto& request =
            waiting.batch->task_list[waiting.owner_task_id].request;
        auto capacity = runtime_queue_->availableCapacity(waiting.kind);
        const bool admitting_gds_read =
            waiting.initial_transport == GDS &&
            request.opcode == Request::READ;
        const bool admitting_gds_write =
            waiting.initial_transport == GDS &&
            request.opcode == Request::WRITE;

        if (capacity.owners == 0 || request.length > capacity.bytes) break;

        QueueSubmit submit;
        submit.batch_token = waiting.batch_token;
        submit.batch_slots_left = 1 + waiting.derived_task_ids.size();
        QueueOwnerInput input;
        input.owner_task_id = waiting.owner_task_id;
        input.derived_task_ids = waiting.derived_task_ids;
        input.request = request;
        input.transport = waiting.initial_transport;
        input.kind = waiting.kind;
        submit.owners.push_back(std::move(input));

        std::vector<QueueOwnerId> admitted_owner_ids;
        auto admit_status =
            runtime_queue_->tryAdmit(submit, admitted_owner_ids);
        if (admit_status.IsTooManyRequests()) break;
        CHECK_STATUS(admit_status);
        if (admitted_owner_ids.size() != 1) {
            return Status::InternalError(
                "runtime admission did not publish exactly one owner"
                LOC_MARK);
        }

        QueuedOwnerState queued;
        queued.batch = waiting.batch;
        queued.owner_task_id = waiting.owner_task_id;
        queued.byte_charge = waiting.byte_charge;
        queued.enqueue_time = waiting.enqueue_time;
        queued.initial_transport = waiting.initial_transport;
        queued.public_task_ids.push_back(waiting.owner_task_id);
        queued.public_task_ids.insert(queued.public_task_ids.end(),
                                      waiting.derived_task_ids.begin(),
                                      waiting.derived_task_ids.end());
        queued_owners_.emplace(admitted_owner_ids.front(),
                               std::move(queued));
        waiting.batch->task_list[waiting.owner_task_id]
            .runtime_admission_waiting = false;
        for (const auto derived_task_id : waiting.derived_task_ids) {
            waiting.batch->task_list[derived_task_id]
                .runtime_admission_waiting = false;
        }

        if (admission_waiting_owners_ == 0 ||
            admission_waiting_bytes_ < waiting.byte_charge) {
            return Status::InternalError(
                "runtime admission waiting accounting underflow" LOC_MARK);
        }
        --admission_waiting_owners_;
        admission_waiting_bytes_ -= waiting.byte_charge;
        if (admitting_gds_read) {
            if (admission_waiting_gds_reads_ == 0) {
                return Status::InternalError(
                    "runtime READ admission accounting underflow" LOC_MARK);
            }
            --admission_waiting_gds_reads_;
            if (consecutive_admission_read_owners_ <
                kMaxConsecutiveAdmissionReads) {
                ++consecutive_admission_read_owners_;
            }
        } else if (admitting_gds_write) {
            if (admission_waiting_gds_writes_ == 0) {
                return Status::InternalError(
                    "runtime WRITE admission accounting underflow" LOC_MARK);
            }
            --admission_waiting_gds_writes_;
            consecutive_admission_read_owners_ = 0;
        }
        admission_waiting_queue_.erase(selected_it);
    }
    if (admission_waiting_queue_.empty()) {
        consecutive_admission_read_owners_ = 0;
    }
    updateRuntimeQueueMetrics();
    return Status::OK();
}

void TransferEngineImpl::updateRuntimeQueueMetrics() {
    size_t queued_gds_reads = admission_waiting_gds_reads_;
    size_t queued_gds_writes = admission_waiting_gds_writes_;
    for (const auto& entry : queued_owners_) {
        const auto& queued = entry.second;
        if (queued.in_dispatch_window ||
            queued.initial_transport != GDS || !queued.batch ||
            queued.owner_task_id >= queued.batch->task_list.size()) {
            continue;
        }
        const auto opcode =
            queued.batch->task_list[queued.owner_task_id].request.opcode;
        if (opcode == Request::READ) {
            ++queued_gds_reads;
        } else if (opcode == Request::WRITE) {
            ++queued_gds_writes;
        }
    }
    const auto& gds_transport = transport_list_[GDS];
    if (gds_transport) {
        gds_transport->updateRuntimeQueueDepth(
            queued_gds_reads, queued_gds_writes);
    }

    if (!runtime_queue_config_.enabled || !runtime_queue_) {
        TentMetrics::instance().updateRuntimeQueue(0, 0, 0, 0);
        return;
    }
    const size_t outstanding_owners = runtime_queue_->outstandingOwners();
    const size_t outstanding_bytes = runtime_queue_->outstandingBytes();
    const size_t admitted_queued_owners =
        outstanding_owners >= dispatch_inflight_owners_
            ? outstanding_owners - dispatch_inflight_owners_
            : 0;
    const size_t admitted_queued_bytes =
        outstanding_bytes >= dispatch_inflight_bytes_
            ? outstanding_bytes - dispatch_inflight_bytes_
            : 0;
    const size_t queued_owners =
        admitted_queued_owners + admission_waiting_owners_;
    const size_t queued_bytes =
        admitted_queued_bytes + admission_waiting_bytes_;
    TentMetrics::instance().updateRuntimeQueue(
        queued_owners, queued_bytes, dispatch_inflight_owners_,
        dispatch_inflight_bytes_);
}

void TransferEngineImpl::recordRuntimeQueueWaitSummary(
    bool read, double queue_wait_seconds) {
    auto& summary = runtime_queue_summary_[read ? 0 : 1];
    ++summary.dispatches;
    const size_t sample_index =
        summary.queue_wait_samples_seen %
        kRuntimeQueueSummarySampleCapacity;
    if (summary.queue_wait_us.size() <
        kRuntimeQueueSummarySampleCapacity) {
        summary.queue_wait_us.push_back(queue_wait_seconds * 1e6);
    } else {
        summary.queue_wait_us[sample_index] =
            queue_wait_seconds * 1e6;
    }
    ++summary.queue_wait_samples_seen;
    maybeLogRuntimeQueueSummary(std::chrono::steady_clock::now());
}

void TransferEngineImpl::recordRuntimeQueueCompletionSummary(
    bool read, size_t bytes, TransferStatusEnum terminal_status,
    double total_latency_seconds) {
    auto& summary = runtime_queue_summary_[read ? 0 : 1];
    ++summary.completions;
    if (terminal_status != COMPLETED) {
        ++summary.failures;
    } else if (bytes >
               std::numeric_limits<size_t>::max() - summary.bytes) {
        summary.bytes = std::numeric_limits<size_t>::max();
    } else {
        summary.bytes += bytes;
    }
    const size_t sample_index =
        summary.total_latency_samples_seen %
        kRuntimeQueueSummarySampleCapacity;
    if (summary.total_latency_us.size() <
        kRuntimeQueueSummarySampleCapacity) {
        summary.total_latency_us.push_back(
            total_latency_seconds * 1e6);
    } else {
        summary.total_latency_us[sample_index] =
            total_latency_seconds * 1e6;
    }
    ++summary.total_latency_samples_seen;
    maybeLogRuntimeQueueSummary(std::chrono::steady_clock::now());
}

void TransferEngineImpl::maybeLogRuntimeQueueSummary(
    std::chrono::steady_clock::time_point now) {
    if (runtime_queue_summary_started_at_.time_since_epoch().count() == 0) {
        runtime_queue_summary_started_at_ = now;
        return;
    }
    const auto elapsed = now - runtime_queue_summary_started_at_;
    if (elapsed < kRuntimeQueueSummaryInterval) return;

    size_t queued_owners = admission_waiting_owners_;
    size_t queued_bytes = admission_waiting_bytes_;
    if (runtime_queue_) {
        const size_t outstanding_owners =
            runtime_queue_->outstandingOwners();
        const size_t outstanding_bytes =
            runtime_queue_->outstandingBytes();
        queued_owners = saturatingAdd(
            queued_owners,
            outstanding_owners >= dispatch_inflight_owners_
                ? outstanding_owners - dispatch_inflight_owners_
                : 0);
        queued_bytes = saturatingAdd(
            queued_bytes,
            outstanding_bytes >= dispatch_inflight_bytes_
                ? outstanding_bytes - dispatch_inflight_bytes_
                : 0);
    }

    const auto& read = runtime_queue_summary_[0];
    const auto& write = runtime_queue_summary_[1];
    const double elapsed_seconds =
        std::chrono::duration<double>(elapsed).count();
    LOG(INFO)
        << "Runtime queue 1s summary: window_ms="
        << elapsed_seconds * 1000.0
        << ", queued_owners=" << queued_owners
        << ", queued_bytes=" << queued_bytes
        << ", dispatch_inflight_owners=" << dispatch_inflight_owners_
        << ", dispatch_inflight_bytes=" << dispatch_inflight_bytes_
        << ", READ{dispatches=" << read.dispatches
        << ", completions=" << read.completions
        << ", failures=" << read.failures
        << ", bytes=" << read.bytes
        << ", throughput_mib_s="
        << static_cast<double>(read.bytes) /
               (1024.0 * 1024.0 * elapsed_seconds)
        << ", queue_wait_p99_us="
        << runtimeQueueNearestRankP99(read.queue_wait_us)
        << ", total_p99_us="
        << runtimeQueueNearestRankP99(read.total_latency_us)
        << "}, WRITE{dispatches=" << write.dispatches
        << ", completions=" << write.completions
        << ", failures=" << write.failures
        << ", bytes=" << write.bytes
        << ", throughput_mib_s="
        << static_cast<double>(write.bytes) /
               (1024.0 * 1024.0 * elapsed_seconds)
        << ", queue_wait_p99_us="
        << runtimeQueueNearestRankP99(write.queue_wait_us)
        << ", total_p99_us="
        << runtimeQueueNearestRankP99(write.total_latency_us)
        << "}}";

    for (auto& summary : runtime_queue_summary_) {
        summary.dispatches = 0;
        summary.completions = 0;
        summary.failures = 0;
        summary.bytes = 0;
        summary.queue_wait_samples_seen = 0;
        summary.total_latency_samples_seen = 0;
        summary.queue_wait_us.clear();
        summary.total_latency_us.clear();
    }
    runtime_queue_summary_started_at_ = now;
}

Status TransferEngineImpl::finishQueuedOwner(
    QueueOwnerId owner_id, TransferStatusEnum terminal_status) {
    auto queued_it = queued_owners_.find(owner_id);
    if (queued_it == queued_owners_.end()) {
        return Status::InvalidEntry("queued owner not found" LOC_MARK);
    }
    auto& queued = queued_it->second;
    if (queued.in_dispatch_window) {
        if (dispatch_inflight_owners_ == 0 ||
            dispatch_inflight_bytes_ < queued.byte_charge ||
            (queued.gds_read_in_dispatch &&
             dispatch_inflight_read_owners_ == 0) ||
            (queued.gds_write_in_dispatch &&
             dispatch_inflight_write_owners_ == 0)) {
            return Status::InternalError(
                "runtime dispatch window accounting underflow" LOC_MARK);
        }
    }
    CHECK_STATUS(runtime_queue_->complete(owner_id, terminal_status));
    const bool is_read =
        queued.batch->task_list[queued.owner_task_id].request.opcode ==
        Request::READ;
    const double total_latency_seconds = std::chrono::duration<double>(
                                             std::chrono::steady_clock::now() -
                                             queued.enqueue_time)
                                             .count();
    TentMetrics::instance().recordRuntimeQueueTotal(
        is_read, total_latency_seconds);
    recordRuntimeQueueCompletionSummary(
        is_read, queued.byte_charge, terminal_status,
        total_latency_seconds);
    if (queued.in_dispatch_window) {
        --dispatch_inflight_owners_;
        dispatch_inflight_bytes_ -= queued.byte_charge;
        if (queued.gds_read_in_dispatch) {
            --dispatch_inflight_read_owners_;
            queued.gds_read_in_dispatch = false;
        }
        if (queued.gds_write_in_dispatch) {
            --dispatch_inflight_write_owners_;
            queued.gds_write_in_dispatch = false;
        }
        queued.in_dispatch_window = false;
    }
    for (const auto task_id : queued.public_task_ids) {
        queued.batch->task_list[task_id].status = terminal_status;
    }
    queued_owners_.erase(queued_it);
    updateRuntimeQueueMetrics();
    return Status::OK();
}

Status TransferEngineImpl::retireQueueForBatch(Batch* batch) {
    if (!batch || batch->queue_token == 0) return Status::OK();
    if (std::any_of(batch->task_list.begin(), batch->task_list.end(),
                    [](const TaskInfo& task) {
                        return task.runtime_admission_waiting;
                    })) {
        return Status::InvalidEntry(
            "batch still has owners waiting for runtime admission" LOC_MARK);
    }
    auto status = runtime_queue_->retireBatch(batch->queue_token);
    if (!status.ok()) return status;
    batch->queue_token = 0;
    return Status::OK();
}

Status TransferEngineImpl::markQueuedOwnerSubmitted(QueueOwnerId owner_id) {
    auto queued_it = queued_owners_.find(owner_id);
    if (queued_it == queued_owners_.end()) {
        return Status::InternalError("queued owner metadata missing" LOC_MARK);
    }
    auto& queued = queued_it->second;
    if (!queued.in_dispatch_window) {
        const auto& task = queued.batch->task_list[queued.owner_task_id];
        queued.in_dispatch_window = true;
        ++dispatch_inflight_owners_;
        dispatch_inflight_bytes_ += queued.byte_charge;
        if (task.type == GDS && task.request.opcode == Request::READ) {
            queued.gds_read_in_dispatch = true;
            ++dispatch_inflight_read_owners_;
        } else if (task.type == GDS) {
            queued.gds_write_in_dispatch = true;
            ++dispatch_inflight_write_owners_;
        }
        updateRuntimeQueueMetrics();
    }
    return Status::OK();
}

Status TransferEngineImpl::dispatchQueuedOwners(
    const std::vector<QueueOwnerId>& owner_ids) {
    struct DispatchGroup {
        Batch* batch{nullptr};
        TransportType type{UNSPEC};
        uint64_t device_mask{~0ULL};
        std::vector<QueueOwnerId> owner_ids;
        std::vector<Request> requests;
        size_t bytes{0};
    };

    Status first_error = Status::OK();
    const auto remember_error = [&](const Status& status) {
        if (first_error.ok() && !status.ok()) first_error = status;
    };
    std::vector<DispatchGroup> groups;
    groups.reserve(owner_ids.size());
    for (const auto owner_id : owner_ids) {
        auto queued_it = queued_owners_.find(owner_id);
        if (queued_it == queued_owners_.end()) {
            remember_error(Status::InternalError(
                "queued owner metadata missing" LOC_MARK));
            // pickForDispatch() already moved the owner to Dispatching. Do
            // not strand it there even if the runtime-side metadata is
            // unexpectedly inconsistent.
            auto complete_status = runtime_queue_->complete(owner_id, FAILED);
            remember_error(complete_status);
            if (complete_status.ok()) updateRuntimeQueueMetrics();
            continue;
        }
        const auto queued = queued_it->second;
        auto* batch = queued.batch;
        auto& task = batch->task_list[queued.owner_task_id];
        const double queue_wait_seconds = std::chrono::duration<double>(
                                              std::chrono::steady_clock::now() -
                                              queued.enqueue_time)
                                              .count();
        const bool is_read = task.request.opcode == Request::READ;
        TentMetrics::instance().recordRuntimeQueueWait(
            is_read, queue_wait_seconds);
        recordRuntimeQueueWaitSummary(is_read, queue_wait_seconds);
        auto route = resolveTransport(task.request, 0);
        task.type = route.transport;
        task.device_mask = route.device_mask;
        if (task.type == UNSPEC) {
            remember_error(finishQueuedOwner(owner_id, FAILED));
            continue;
        }

        if (task.type == TCP) {
            std::vector<std::string> staging_params;
            findStagingPolicy(task.request, staging_params);
            if (!staging_params.empty() && staging_proxy_) {
                task.staging = true;
                auto status = staging_proxy_->submit(
                    &task, (BatchID)batch, staging_params);
                if (!status.ok()) {
                    remember_error(finishQueuedOwner(owner_id, FAILED));
                } else {
                    remember_error(markQueuedOwnerSubmitted(owner_id));
                }
                continue;
            }
        }

        auto group_it = groups.end();
        // GDS owns separate READ/WRITE single-IO worker pools. Keep each GDS
        // owner as its own transport submission so the runtime queue cannot
        // accidentally recreate a multi-entry cuFile batch. Other transports
        // retain their existing per-batch grouping behavior.
        if (task.type != GDS) {
            group_it = std::find_if(
                groups.begin(), groups.end(), [&](const DispatchGroup& group) {
                    return group.batch == batch && group.type == task.type &&
                           group.device_mask == task.device_mask;
                });
        }
        if (group_it == groups.end()) {
            groups.push_back(
                DispatchGroup{batch, task.type, task.device_mask, {}, {}, 0});
            group_it = std::prev(groups.end());
        }
        group_it->owner_ids.push_back(owner_id);
        group_it->requests.push_back(task.request);
        group_it->bytes += task.request.length;
    }

    for (auto& group : groups) {
        auto& transport = transport_list_[group.type];
        const auto fail_group = [&]() {
            for (const auto owner_id : group.owner_ids) {
                auto queued_it = queued_owners_.find(owner_id);
                if (queued_it == queued_owners_.end()) {
                    remember_error(Status::InternalError(
                        "queued owner metadata disappeared" LOC_MARK));
                    auto complete_status =
                        runtime_queue_->complete(owner_id, FAILED);
                    remember_error(complete_status);
                    if (complete_status.ok()) updateRuntimeQueueMetrics();
                    continue;
                }
                auto& task = queued_it->second.batch->task_list[
                    queued_it->second.owner_task_id];
                task.type = UNSPEC;
                remember_error(finishQueuedOwner(owner_id, FAILED));
            }
        };
        if (!transport) {
            fail_group();
            continue;
        }

        auto& sub_batch = group.batch->sub_batch[group.type];
        if (!sub_batch) {
            auto status =
                transport->allocateSubBatch(sub_batch, group.batch->max_size);
            if (!status.ok()) {
                fail_group();
                continue;
            }
            attachProgressNotifier(group.batch, sub_batch);
        }
        if (group.type == RDMA) sub_batch->device_mask = group.device_mask;

        const size_t first_sub_task_id = sub_batch->size();
        bool group_metadata_valid = true;
        for (size_t index = 0; index < group.owner_ids.size(); ++index) {
            auto queued_it = queued_owners_.find(group.owner_ids[index]);
            if (queued_it == queued_owners_.end()) {
                remember_error(Status::InternalError(
                    "queued owner metadata disappeared" LOC_MARK));
                group_metadata_valid = false;
                break;
            }
            auto& task = group.batch->task_list[queued_it->second.owner_task_id];
            task.sub_task_id = first_sub_task_id + index;
        }
        if (!group_metadata_valid) {
            fail_group();
            continue;
        }

        auto status = transport->submitTransferTasks(sub_batch, group.requests);
        if (!status.ok()) {
            fail_group();
            continue;
        }
        LOG_EVERY_N(INFO, 256)
            << "Runtime queue transport dispatch: transport="
            << transportTypeName(group.type)
            << ", requests=" << group.requests.size()
            << ", bytes=" << group.bytes;
        for (const auto owner_id : group.owner_ids) {
            remember_error(markQueuedOwnerSubmitted(owner_id));
        }
    }
    return first_error;
}

Status TransferEngineImpl::refillDispatchWindow() {
    std::lock_guard<std::recursive_mutex> lk(progress_mutex_);
    if (!runtime_queue_config_.enabled) return Status::OK();
    CHECK_STATUS(admitWaitingOwners());
    // Publish external GDS backlog immediately before selecting owners. The
    // post-dispatch update below then removes every owner that actually
    // entered the transport window.
    updateRuntimeQueueMetrics();
    if (dispatch_inflight_owners_ >=
            runtime_queue_config_.max_dispatch_owners ||
        dispatch_inflight_bytes_ >= runtime_queue_config_.max_dispatch_bytes) {
        return Status::OK();
    }

    const size_t owner_budget =
        runtime_queue_config_.max_dispatch_owners - dispatch_inflight_owners_;
    const size_t byte_budget =
        runtime_queue_config_.max_dispatch_bytes - dispatch_inflight_bytes_;
    size_t read_owner_limit =
        runtime_queue_config_.max_dispatch_read_owners;
    size_t write_owner_limit =
        runtime_queue_config_.max_dispatch_write_owners;
    // GDS owns a second, latency-driven concurrency controller. Clamp the
    // runtime-side owner window to its live limit so a P99-triggered
    // reduction drains all the way back to the runtime queue instead of
    // merely moving excess owners into GDS's internal pending deques.
    const auto& gds_transport = transport_list_[GDS];
    if (gds_transport) {
        read_owner_limit = std::min(
            read_owner_limit,
            gds_transport->runtimeQueueDispatchLimit(Request::READ));
        write_owner_limit = std::min(
            write_owner_limit,
            gds_transport->runtimeQueueDispatchLimit(Request::WRITE));
    }
    const size_t read_budget =
        read_owner_limit > dispatch_inflight_read_owners_
            ? read_owner_limit - dispatch_inflight_read_owners_
            : 0;
    const size_t write_budget =
        write_owner_limit > dispatch_inflight_write_owners_
            ? write_owner_limit - dispatch_inflight_write_owners_
            : 0;
    auto picked = runtime_queue_->pickForDispatch(
        owner_budget, byte_budget, read_budget, write_budget);
    auto dispatch_status = dispatchQueuedOwners(picked);
    updateRuntimeQueueMetrics();
    return dispatch_status;
}

Status TransferEngineImpl::progressRuntimeQueue() {
    std::lock_guard<std::recursive_mutex> lk(progress_mutex_);
    if (!runtime_queue_config_.enabled) return Status::OK();

    CHECK_STATUS(refillDispatchWindow());

    std::vector<QueueOwnerId> owner_ids;
    owner_ids.reserve(queued_owners_.size());
    for (const auto& entry : queued_owners_) {
        if (entry.second.in_dispatch_window) owner_ids.push_back(entry.first);
    }

    bool released_window = false;
    for (const auto owner_id : owner_ids) {
        auto queued_it = queued_owners_.find(owner_id);
        if (queued_it == queued_owners_.end()) continue;

        auto& queued = queued_it->second;
        if (!queued.in_dispatch_window) continue;
        auto* batch = queued.batch;
        if (!batch || !alive_batches_.count((BatchID)batch)) continue;
        if (queued.owner_task_id >= batch->task_list.size()) {
            return Status::InternalError(
                "queued owner task id out of range" LOC_MARK);
        }

        auto& task = batch->task_list[queued.owner_task_id];
        auto prev_status = task.status;
        TransferStatus task_status;
        CHECK_STATUS(pollTaskStatus(batch, queued.owner_task_id, task_status));
        updateTaskStatusAfterPoll(batch, queued.owner_task_id, task_status,
                                  true);
        recordTaskCompletionMetrics(task, prev_status, task_status.s);

        if (task_status.s == PENDING) continue;

        CHECK_STATUS(finishQueuedOwner(owner_id, task_status.s));
        if (task_status.s == COMPLETED)
            CHECK_STATUS(maybeFireSubmitHooks(batch));
        released_window = true;
    }

    if (released_window) CHECK_STATUS(refillDispatchWindow());
    return Status::OK();
}

bool TransferEngineImpl::hasActiveRuntimeQueue() {
    std::lock_guard<std::recursive_mutex> lk(progress_mutex_);
    return runtime_queue_config_.enabled &&
           (!queued_owners_.empty() || !admission_waiting_queue_.empty());
}

bool TransferEngineImpl::shouldQueueSubmit(const PreparedSubmit& prepared,
                                           QueueOwnerKind owner_kind) const {
    if (!runtime_queue_config_.enabled) return false;
    if (owner_kind == QueueOwnerKind::StagingInternal) return true;
    return std::none_of(
        prepared.owners.begin(), prepared.owners.end(),
        [](const PreparedSubmit::Owner& owner) { return owner.staging; });
}

Status TransferEngineImpl::submitTransfer(
    BatchID batch_id, const std::vector<Request>& request_list,
    const Notification* notifi, QueueOwnerKind owner_kind) {
    Batch* batch = nullptr;
    CHECK_STATUS(retainBatch(batch_id, batch));
    BatchRef batch_ref(*this, batch);
    const size_t start_task_id = batch_ref.get()->task_list.size();
    PreparedSubmit prepared;
    CHECK_STATUS(prepareSubmit(batch_ref.get(), request_list, prepared));

    if (shouldQueueSubmit(prepared, owner_kind)) {
        CHECK_STATUS(
            enqueuePreparedSubmit(batch_ref.get(), prepared, owner_kind));
        auto dispatch_status = refillDispatchWindow();
        if (!dispatch_status.ok()) {
            LOG(WARNING) << "runtime queue dispatch failed after admission: "
                         << dispatch_status.ToString();
        }
        notifyRuntimeQueueReady();
    } else {
        CHECK_STATUS(commitPreparedSubmit(batch_ref.get(), prepared));
    }

    if (notifi) {
        addSubmitHook(batch_ref.get(), start_task_id, request_list, *notifi);
    }
    return batch_ref.release();
}

Status TransferEngineImpl::submitTransfer(
    BatchID batch_id, const std::vector<Request>& request_list) {
    return submitTransfer(batch_id, request_list, nullptr,
                          QueueOwnerKind::User);
}

Status TransferEngineImpl::submitStagingTransfer(
    BatchID batch_id, const std::vector<Request>& request_list) {
    return submitTransfer(batch_id, request_list, nullptr,
                          QueueOwnerKind::StagingInternal);
}

void TransferEngineImpl::addSubmitHook(Batch* batch, size_t start_task_id,
                                       const std::vector<Request>& request_list,
                                       const Notification& notifi) {
    Batch::SubmitHook hook;
    hook.start_task_id = start_task_id;
    hook.end_task_id = start_task_id + request_list.size();
    hook.notifi = notifi;
    hook.fired = false;
    for (const auto& request : request_list)
        hook.targets.insert(request.target_id);
    batch->submit_hooks.emplace_back(std::move(hook));
}

Status TransferEngineImpl::maybeFireSubmitHooks(Batch* batch, bool check) {
    for (auto& hook : batch->submit_hooks) {
        if (hook.fired) continue;
        bool all_completed = true;
        if (check) {
            for (size_t tid = hook.start_task_id; tid < hook.end_task_id;
                 ++tid) {
                auto& t = batch->task_list[tid];
                if (t.status == PENDING) {
                    all_completed = false;
                    break;
                }
                if (t.status != COMPLETED) {
                    all_completed = false;
                    break;
                }
            }
        }
        if (!all_completed) continue;
        Status last = Status::OK();
        for (auto target_id : hook.targets) {
            last = sendNotification(target_id, hook.notifi);
            if (!last.ok()) {
                LOG(WARNING) << "sendNotification failed: " << last.ToString();
                break;
            }
        }
        if (last.ok()) hook.fired = true;
    }
    return Status::OK();
}

Status TransferEngineImpl::submitTransfer(
    BatchID batch_id, const std::vector<Request>& request_list,
    const Notification& notifi) {
    return submitTransfer(batch_id, request_list, &notifi,
                          QueueOwnerKind::User);
}

Status TransferEngineImpl::resubmitTransferTask(Batch* batch, size_t task_id) {
    auto& task = batch->task_list[task_id];
    auto prev_type = task.type;

    if (++task.failover_count > max_failover_attempts_) {
        LOG(WARNING) << "Task failover limit reached ("
                     << max_failover_attempts_
                     << "), last transport=" << transportTypeName(prev_type);
        return Status::InvalidEntry(
            "Failover limit exceeded, all transports exhausted");
    }

    if (task.staging)
        task.staging = false;
    else
        task.xport_priority = task.failover_count;

    auto result = resolveTransport(task.request, task.xport_priority);
    auto type = result.transport;
    if (type == UNSPEC) {
        // One failed physical GDS batch can contain many logical tasks. Avoid
        // emitting the same failover warning once per task while preserving a
        // periodic signal for sustained failures.
        LOG_EVERY_N(WARNING, 64)
            << "No more transports available after "
            << transportTypeName(prev_type) << " failed";
        return Status::InvalidEntry("All available transports are failed");
    }

    LOG(INFO) << "Transport failover: " << transportTypeName(prev_type)
              << " -> " << transportTypeName(type) << " (attempt "
              << task.failover_count << "/" << max_failover_attempts_ << ")";
    TENT_RECORD_TRANSPORT_FAILOVER();

    auto& transport = transport_list_[type];
    if (!batch->sub_batch[type]) {
        CHECK_STATUS(transport->allocateSubBatch(batch->sub_batch[type],
                                                 batch->max_size));
        attachProgressNotifier(batch, batch->sub_batch[type]);
    }
    auto& sub_batch = batch->sub_batch[type];
    task.sub_task_id = sub_batch->size();
    task.type = type;
    return transport->submitTransferTasks(sub_batch, {task.request});
}

Status TransferEngineImpl::pollTaskStatus(Batch* batch, size_t task_id,
                                          TransferStatus& task_status) {
    auto& task = batch->task_list[task_id];
    if (task.staging) {
        return staging_proxy_->getStatus(&task, task_status);
    }

    if (task.type == UNSPEC) {
        task_status.s = FAILED;
        task_status.transferred_bytes = 0;
        return Status::OK();
    }

    auto& transport = transport_list_[task.type];
    auto& sub_batch = batch->sub_batch[task.type];
    if (!transport || !sub_batch) {
        return Status::InvalidArgument("Transport not available" LOC_MARK);
    }
    return transport->getTransferStatus(sub_batch, task.sub_task_id,
                                        task_status);
}

void TransferEngineImpl::updateTaskStatusAfterPoll(Batch* batch, size_t task_id,
                                                   TransferStatus& task_status,
                                                   bool allow_failover) {
    auto& task = batch->task_list[task_id];
    task.status = task_status.s;
    if (!allow_failover || task_status.s != FAILED || task.type == UNSPEC)
        return;

    if (resubmitTransferTask(batch, task_id).ok()) {
        task_status.s = PENDING;
        task.status = PENDING;
    }
}

Status TransferEngineImpl::sendNotification(SegmentID target_id,
                                            const Notification& notifi) {
    for (size_t type = 0; type < kSupportedTransportTypes; ++type) {
        auto& transport = transport_list_[type];
        if (!transport || !transport->supportNotification()) continue;
        return transport->sendNotification(target_id, notifi);
    }
    return Status::InvalidArgument("Notification not supported" LOC_MARK);
}

Status TransferEngineImpl::probePeerAliveByID(SegmentID target_id) {
    return metadata_->segmentManager().withCachedSegment(
        target_id, [&](SegmentDesc* segment) {
            auto rpc_server_addr = segment->rpc_server_addr;
            if (rpc_server_addr.empty()) {
                return Status::NeedsRefreshCache(
                    "Empty RPC server addr" LOC_MARK);
            }
            auto status = ControlClient::probe(rpc_server_addr);
            if (status.IsRpcServiceError()) {
                // Perhaps rpc_server_addr can be updated in the future
                return Status::NeedsRefreshCache(
                    "RPC service error: " + std::string{status.message()} +
                    LOC_MARK);
            }
            return status;
        });
}

Status TransferEngineImpl::receiveNotification(
    std::vector<Notification>& notifi_list) {
    for (size_t type = 0; type < kSupportedTransportTypes; ++type) {
        auto& transport = transport_list_[type];
        if (!transport || !transport->supportNotification()) continue;
        return transport->receiveNotification(notifi_list);
    }
    return Status::InvalidArgument("Notification not supported" LOC_MARK);
}

Status TransferEngineImpl::getTransferStatus(BatchID batch_id, size_t task_id,
                                             TransferStatus& task_status) {
    if (!batch_id) return Status::InvalidArgument("Invalid batch ID" LOC_MARK);
    std::lock_guard<std::recursive_mutex> lk(progress_mutex_);
    if (!alive_batches_.count(batch_id))
        return Status::InvalidArgument("Batch is not alive" LOC_MARK);
    Batch* batch = (Batch*)(batch_id);
    if (task_id >= batch->task_list.size())
        return Status::InvalidArgument("Invalid task ID" LOC_MARK);
    if (batch->task_list[task_id].runtime_admission_waiting) {
        task_status.s = PENDING;
        task_status.transferred_bytes = 0;
        return Status::OK();
    }
    const size_t public_task_id = task_id;
    size_t poll_task_id = task_id;
    CHECK_STATUS(refillDispatchWindow());
    if (runtime_queue_config_.enabled && batch->queue_token != 0) {
        QueueOwnerId owner_id = 0;
        auto resolve_status = runtime_queue_->resolveOwner(
            batch->queue_token, public_task_id, owner_id);
        if (resolve_status.ok()) {
            TransferStatusEnum public_status = PENDING;
            CHECK_STATUS(runtime_queue_->getPublicStatus(
                batch->queue_token, public_task_id, public_status));
            auto queued_it = queued_owners_.find(owner_id);
            if (public_status != PENDING ||
                (queued_it != queued_owners_.end() &&
                 !queued_it->second.in_dispatch_window)) {
                task_status.s = public_status;
                task_status.transferred_bytes =
                    public_status == COMPLETED
                        ? batch->task_list[public_task_id].request.length
                        : 0;
                return Status::OK();
            }
            if (batch->task_list[public_task_id].derived &&
                queued_it != queued_owners_.end()) {
                poll_task_id = queued_it->second.owner_task_id;
            }
        }
    }
    auto& task = batch->task_list[poll_task_id];
    auto prev_status = task.status;
    CHECK_STATUS(pollTaskStatus(batch, poll_task_id, task_status));
    updateTaskStatusAfterPoll(batch, poll_task_id, task_status,
                              enable_auto_failover_on_poll_);
    if (runtime_queue_config_.enabled && batch->queue_token != 0 &&
        task_status.s != PENDING) {
        QueueOwnerId owner_id = 0;
        auto resolve_status = runtime_queue_->resolveOwner(
            batch->queue_token, public_task_id, owner_id);
        if (resolve_status.ok()) {
            CHECK_STATUS(finishQueuedOwner(owner_id, task_status.s));
            CHECK_STATUS(refillDispatchWindow());
        }
    }

    // Record metrics when task transitions to terminal state
    recordTaskCompletionMetrics(batch->task_list[poll_task_id], prev_status,
                                task_status.s);

    if (task_status.s == COMPLETED) CHECK_STATUS(maybeFireSubmitHooks(batch));
    return Status::OK();
}

Status TransferEngineImpl::getTransferStatus(
    BatchID batch_id, std::vector<TransferStatus>& status_list) {
    if (!batch_id) return Status::InvalidArgument("Invalid batch ID" LOC_MARK);
    std::lock_guard<std::recursive_mutex> lk(progress_mutex_);
    if (!alive_batches_.count(batch_id))
        return Status::InvalidArgument("Batch is not alive" LOC_MARK);
    Batch* batch = (Batch*)(batch_id);
    status_list.clear();
    for (size_t task_id = 0; task_id < batch->task_list.size(); ++task_id) {
        TransferStatus task_status;
        CHECK_STATUS(getTransferStatus(batch_id, task_id, task_status));
        status_list.push_back(task_status);
    }
    return Status::OK();
}

Status TransferEngineImpl::getBatchStatus(BatchID batch_id,
                                          TransferStatus& overall_status,
                                          bool allow_failover) {
    if (!batch_id) return Status::InvalidArgument("Invalid batch ID" LOC_MARK);
    std::lock_guard<std::recursive_mutex> lk(progress_mutex_);
    if (!alive_batches_.count(batch_id))
        return Status::InvalidArgument("Batch is not alive" LOC_MARK);
    CHECK_STATUS(refillDispatchWindow());
    Batch* batch = (Batch*)(batch_id);
    overall_status.s = PENDING;
    overall_status.transferred_bytes = 0;
    size_t success_tasks = 0;
    size_t failed_tasks = 0;
    size_t total_tasks = 0;
    TransferStatusEnum worst_failure = PENDING;
    auto isWorse = [](TransferStatusEnum cur, TransferStatusEnum best) {
        static const std::unordered_map<TransferStatusEnum, int> severity = {
            {INITIAL, 0},  {PENDING, 0}, {COMPLETED, 0}, {INVALID, 1},
            {CANCELED, 2}, {TIMEOUT, 3}, {FAILED, 4},
        };
        return severity.at(cur) > severity.at(best);
    };
    for (size_t task_id = 0; task_id < batch->task_list.size(); ++task_id) {
        auto& task = batch->task_list[task_id];
        if (task.derived) continue;  // This task is performed by other tasks
        total_tasks++;
        if (task.runtime_admission_waiting) continue;
        if (runtime_queue_config_.enabled && batch->queue_token != 0) {
            QueueOwnerId owner_id = 0;
            auto resolve_status = runtime_queue_->resolveOwner(
                batch->queue_token, task_id, owner_id);
            if (resolve_status.ok()) {
                TransferStatusEnum public_status = PENDING;
                CHECK_STATUS(runtime_queue_->getPublicStatus(
                    batch->queue_token, task_id, public_status));
                auto queued_it = queued_owners_.find(owner_id);
                if (public_status == PENDING) {
                    if (queued_it != queued_owners_.end() &&
                        !queued_it->second.in_dispatch_window) {
                        continue;
                    }
                }
                if (public_status == COMPLETED) {
                    success_tasks++;
                    overall_status.transferred_bytes += task.request.length;
                    continue;
                }
                if (public_status != PENDING) {
                    failed_tasks++;
                    if (isWorse(public_status, worst_failure))
                        worst_failure = public_status;
                    continue;
                }
            }
        }
        TransferStatus task_status;
        if (task.status != PENDING) {
            if (task.status == COMPLETED) {
                success_tasks++;
                overall_status.transferred_bytes += task.request.length;
            } else {
                failed_tasks++;
                if (isWorse(task.status, worst_failure))
                    worst_failure = task.status;
            }
            continue;
        }
        auto prev_status = task.status;
        CHECK_STATUS(pollTaskStatus(batch, task_id, task_status));
        updateTaskStatusAfterPoll(batch, task_id, task_status, allow_failover);
        if (runtime_queue_config_.enabled && batch->queue_token != 0 &&
            task_status.s != PENDING) {
            QueueOwnerId owner_id = 0;
            auto resolve_status = runtime_queue_->resolveOwner(
                batch->queue_token, task_id, owner_id);
            if (resolve_status.ok()) {
                CHECK_STATUS(finishQueuedOwner(owner_id, task_status.s));
                CHECK_STATUS(refillDispatchWindow());
            }
        }

        if (task_status.s == COMPLETED) {
            success_tasks++;
            overall_status.transferred_bytes += task_status.transferred_bytes;
        } else if (task_status.s != PENDING) {
            failed_tasks++;
            if (isWorse(task_status.s, worst_failure))
                worst_failure = task_status.s;
        }

        // Record metrics when task transitions to terminal state
        recordTaskCompletionMetrics(batch->task_list[task_id], prev_status,
                                    task_status.s);
    }
    // Determine overall status: COMPLETED only when all succeed; FAILED only
    // when all tasks are terminal (no in-flight work) and at least one failed;
    // otherwise PENDING (some tasks still running).
    if (success_tasks == total_tasks) {
        overall_status.s = COMPLETED;
    } else if (success_tasks + failed_tasks == total_tasks) {
        overall_status.s = worst_failure;
    }
    // else: some tasks still PENDING → overall_status.s stays PENDING
    CHECK_STATUS(maybeFireSubmitHooks(batch, overall_status.s == COMPLETED));
    return Status::OK();
}

Status TransferEngineImpl::getTransferStatus(BatchID batch_id,
                                             TransferStatus& overall_status) {
    return getBatchStatus(batch_id, overall_status,
                          enable_auto_failover_on_poll_);
}

Status TransferEngineImpl::progressBatch(BatchID batch_id,
                                         TransferStatus& overall_status) {
    return getBatchStatus(batch_id, overall_status, true);
}

void TransferEngineImpl::notifyBatchMaybeReady(BatchID batch_id) {
    if (progress_worker_) progress_worker_->notifyBatchMaybeReady(batch_id);
}

void TransferEngineImpl::notifyRuntimeQueueReady() {
    if (progress_worker_) progress_worker_->notifyRuntimeQueueReady();
}

Status TransferEngineImpl::waitTransferCompletion(BatchID batch_id) {
    TransferStatus xfer_status;
    while (true) {
        CHECK_STATUS(progressBatch(batch_id, xfer_status));
        if (xfer_status.s != PENDING) {
            freeBatch(batch_id);
            return xfer_status.s == COMPLETED
                       ? Status::OK()
                       : Status::InternalError(
                             "Transfer failed: " +
                             std::to_string((int)xfer_status.s));
        }
    }
}

Status TransferEngineImpl::transferSync(
    const std::vector<Request>& request_list) {
    auto batch_id = allocateBatch(request_list.size());
    CHECK_STATUS(submitTransfer(batch_id, request_list));
    while (true) {
        TransferStatus xfer_status;
        CHECK_STATUS(progressBatch(batch_id, xfer_status));
        if (xfer_status.s == COMPLETED) break;
        if (xfer_status.s != PENDING) {
            CHECK_STATUS(freeBatch(batch_id));
            return Status::InternalError(
                "Transfer via stage buffer failed" LOC_MARK);
        }
    }
    CHECK_STATUS(freeBatch(batch_id));
    return Status::OK();
}

uint64_t TransferEngineImpl::lockStageBuffer(const std::string& location) {
    uint64_t addr = 0;
    auto status = staging_proxy_->pinStageBuffer(location, addr);
    if (!status.ok()) LOG(ERROR) << status.ToString();
    return addr;
}

Status TransferEngineImpl::unlockStageBuffer(uint64_t addr) {
    return staging_proxy_->unpinStageBuffer(addr);
}

void TransferEngineImpl::recordTaskCompletionMetrics(
    TaskInfo& task, TransferStatusEnum prev_status,
    TransferStatusEnum new_status) {
    if (prev_status == PENDING && new_status != PENDING && !task.derived) {
        auto start_time = task.start_time;
        if (start_time.time_since_epoch().count() > 0) {
            auto end_time = std::chrono::steady_clock::now();
            double latency_seconds =
                std::chrono::duration<double>(end_time - start_time).count();
            if (new_status == COMPLETED) {
                if (task.request.opcode == Request::READ) {
                    TentMetrics::instance().recordReadCompleted(
                        task.request.length, latency_seconds);
                } else {
                    TentMetrics::instance().recordWriteCompleted(
                        task.request.length, latency_seconds);
                }
                // Observability only (RFC #2519): if this transfer carried a
                // deadline, emit the post-hoc feasibility ratio MLU =
                // actual_transfer_time / available_window, where the window is
                // (deadline - submit_time). MLU < 1 met the deadline; >= 1
                // missed it. This does not drive any admission/scheduling yet.
                if (task.request.deadline_ns != 0) {
                    uint64_t start_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            start_time.time_since_epoch())
                            .count());
                    if (task.request.deadline_ns > start_ns) {
                        double window_seconds =
                            (task.request.deadline_ns - start_ns) / 1e9;
                        TentMetrics::instance().recordDeadlineMLU(
                            latency_seconds / window_seconds);
                    } else {
                        // Deadline already in the past at submit: infeasible.
                        TentMetrics::instance().recordDeadlineMLU(5.0);
                    }
                }
            } else if (new_status == FAILED) {
                if (task.request.opcode == Request::READ) {
                    TentMetrics::instance().recordReadFailed(
                        task.request.length);
                } else {
                    TentMetrics::instance().recordWriteFailed(
                        task.request.length);
                }
            }
            // Reset start_time to prevent duplicate recording
            task.start_time = std::chrono::steady_clock::time_point{};
        }
    }
}

}  // namespace tent
}  // namespace mooncake
