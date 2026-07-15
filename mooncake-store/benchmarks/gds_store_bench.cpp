#include <cuda_runtime_api.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
#include <sys/resource.h>

#include "real_client.h"
#include "replica.h"
#include "utils.h"

DEFINE_string(operation, "read",
              "Workload: prepare, write, read, mixed, verify, check, or cleanup");
DEFINE_string(local_hostname, "",
              "Required non-loopback host:port; host must match the GDS accessor client_host");
DEFINE_string(metadata_server, "http://127.0.0.1:8080/metadata",
              "Transfer-engine metadata connection string");
DEFINE_string(master_server, "127.0.0.1:50051",
              "Mooncake master address");
DEFINE_string(protocol, "tcp", "Store protocol (TENT selects GDS internally)");
DEFINE_string(device_name, "", "Device list passed to Store setup");
DEFINE_string(tenant_id, "default", "Tenant ID");
DEFINE_uint64(global_segment_size, 1024,
              "Small identity segment used to register reader host_id with Master");
DEFINE_uint64(local_buffer_size, 0,
              "Store staging buffer size; the benchmark registers its GPU buffers");

DEFINE_int32(gpu_id, 0, "CUDA device ID");
DEFINE_uint64(value_size, 4ULL * 1024 * 1024, "Object size in bytes");
DEFINE_uint64(num_objects, 1024,
              "Number of persistent objects used by prepare/read/verify/cleanup");
DEFINE_uint64(key_start, 0, "First persistent object ID");
DEFINE_string(key_prefix, "gds-store-bench", "Object key prefix");
DEFINE_int32(threads, 1, "Number of synchronous worker threads");
DEFINE_int32(batch_size, 1, "Objects per Store batch call");
DEFINE_int32(duration_sec, 30, "Measured duration for write/read/mixed");
DEFINE_int32(warmup_sec, 3, "Warmup duration for write/read/mixed");
DEFINE_int32(read_ratio, 70, "Read request percentage in mixed mode");
DEFINE_uint64(seed, 1, "Random seed");
DEFINE_uint64(latency_sample_every, 1,
              "Record one request latency for every N requests");
DEFINE_uint64(max_latency_samples_per_thread, 1000000,
              "Maximum retained request latency samples per worker");

DEFINE_int32(memory_replica_num, 0, "Number of memory replicas for writes");
DEFINE_int32(nof_replica_num, 0, "Number of NoF replicas for writes");
DEFINE_int32(gds_replica_num, 1, "Number of GDS SSD replicas for writes");
DEFINE_string(preferred_gds_segments, "",
              "Comma-separated preferred GDS SSD segment names");
DEFINE_bool(prepare_before_run, false,
            "Prepare num_objects patterned objects before read or mixed");
DEFINE_bool(verify_reads, false,
            "Copy successful reads to host and verify the per-key byte pattern");
DEFINE_bool(cleanup_after_run, false,
            "Remove objects created or consumed by this run");
DEFINE_bool(cleanup_force, false, "Use force=true for object removal");
DEFINE_int32(cleanup_batch_size, 128, "Keys per BatchRemove call");
DEFINE_string(allow_destructive_gds, "",
              "Must be YES for any workload that writes GDS replicas");
DEFINE_string(csv_path, "", "Append the measured phase result to this CSV file");

namespace {

using Clock = std::chrono::steady_clock;
using mooncake::RealClient;
using mooncake::ReplicateConfig;

constexpr uint64_t kGdsAlignment = 4096;

enum class Operation {
    kPrepare,
    kWrite,
    kRead,
    kMixed,
    kVerify,
    kCheck,
    kCleanup
};

struct PhaseSpec {
    std::string name;
    Operation operation;
    std::string read_prefix;
    std::string write_prefix;
    uint64_t read_start{0};
    uint64_t read_count{0};
    uint64_t write_start{0};
    uint64_t finite_count{0};
    int duration_sec{0};
    bool patterned_writes{false};
    bool verify_reads{false};
};

struct ThreadStats {
    uint64_t requests{0};
    uint64_t successful_kvs{0};
    uint64_t failed_kvs{0};
    uint64_t bytes{0};
    std::vector<double> latencies_us;
    std::map<int64_t, uint64_t> errors;
};

struct PhaseResult : ThreadStats {
    std::string name;
    double elapsed_sec{0};
    double cpu_user_sec{0};
    double cpu_system_sec{0};
    uint64_t write_begin{0};
    uint64_t write_end{0};
};

void checkCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(status));
    }
}

Operation parseOperation(const std::string& value) {
    if (value == "prepare") return Operation::kPrepare;
    if (value == "write") return Operation::kWrite;
    if (value == "read") return Operation::kRead;
    if (value == "mixed") return Operation::kMixed;
    if (value == "verify") return Operation::kVerify;
    if (value == "check") return Operation::kCheck;
    if (value == "cleanup") return Operation::kCleanup;
    throw std::invalid_argument("invalid --operation=" + value);
}

bool operationWrites(Operation op) {
    return op == Operation::kPrepare || op == Operation::kWrite ||
           op == Operation::kMixed || op == Operation::kVerify;
}

bool operationIsTimed(Operation op) {
    return op == Operation::kWrite || op == Operation::kRead ||
           op == Operation::kMixed;
}

std::vector<std::string> splitCsv(const std::string& value) {
    std::vector<std::string> result;
    std::unordered_set<std::string> seen;
    std::stringstream input(value);
    std::string item;
    while (std::getline(input, item, ',')) {
        const auto first = item.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        const auto last = item.find_last_not_of(" \t\r\n");
        item = item.substr(first, last - first + 1);
        if (seen.insert(item).second) result.push_back(item);
    }
    return result;
}

std::string makeKey(const std::string& prefix, uint64_t id) {
    std::ostringstream out;
    out << prefix << '-' << std::setw(20) << std::setfill('0') << id;
    return out.str();
}

uint8_t patternForId(uint64_t id) {
    return static_cast<uint8_t>((id * 1315423911ULL + 0x5a) & 0xff);
}

class RegisteredGpuBuffer {
   public:
    RegisteredGpuBuffer(std::shared_ptr<RealClient> client, int gpu_id,
                        size_t bytes)
        : client_(std::move(client)), gpu_id_(gpu_id), bytes_(bytes) {
        checkCuda(cudaSetDevice(gpu_id_), "cudaSetDevice");
        checkCuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
        checkCuda(cudaMemset(data_, 0x5a, bytes_), "cudaMemset");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        if (reinterpret_cast<uintptr_t>(data_) % kGdsAlignment != 0) {
            cudaFree(data_);
            data_ = nullptr;
            throw std::runtime_error("cudaMalloc returned a non-4096-aligned pointer");
        }
        const int rc = client_->register_buffer(data_, bytes_);
        if (rc != 0) {
            cudaFree(data_);
            data_ = nullptr;
            throw std::runtime_error("Store register_buffer failed: " +
                                     std::to_string(rc));
        }
        registered_ = true;
    }

    RegisteredGpuBuffer(const RegisteredGpuBuffer&) = delete;
    RegisteredGpuBuffer& operator=(const RegisteredGpuBuffer&) = delete;

    ~RegisteredGpuBuffer() {
        if (data_ == nullptr) return;
        cudaSetDevice(gpu_id_);
        if (registered_) {
            const int rc = client_->unregister_buffer(data_);
            if (rc != 0) {
                LOG(ERROR) << "unregister_buffer failed: " << rc;
            }
        }
        const auto rc = cudaFree(data_);
        if (rc != cudaSuccess) {
            LOG(ERROR) << "cudaFree failed: " << cudaGetErrorString(rc);
        }
    }

    void* slot(size_t index, size_t value_size) const {
        return static_cast<char*>(data_) + index * value_size;
    }

   private:
    std::shared_ptr<RealClient> client_;
    int gpu_id_;
    size_t bytes_;
    void* data_{nullptr};
    bool registered_{false};
};

ReplicateConfig makeReplicateConfig() {
    ReplicateConfig config;
    config.replica_num = static_cast<size_t>(FLAGS_memory_replica_num);
    config.nof_replica_num = static_cast<size_t>(FLAGS_nof_replica_num);
    config.gds_replica_num = static_cast<size_t>(FLAGS_gds_replica_num);
    config.preferred_gds_segments = splitCsv(FLAGS_preferred_gds_segments);
    return config;
}

void mergeStats(PhaseResult& dst, ThreadStats&& src) {
    dst.requests += src.requests;
    dst.successful_kvs += src.successful_kvs;
    dst.failed_kvs += src.failed_kvs;
    dst.bytes += src.bytes;
    dst.latencies_us.insert(dst.latencies_us.end(), src.latencies_us.begin(),
                            src.latencies_us.end());
    for (const auto& [code, count] : src.errors) dst.errors[code] += count;
}

double percentile(const std::vector<double>& sorted, double fraction) {
    if (sorted.empty()) return 0;
    const auto rank = static_cast<size_t>(
        std::ceil(fraction * static_cast<double>(sorted.size())));
    return sorted[std::min(std::max<size_t>(rank, 1), sorted.size()) - 1];
}

double timevalSeconds(const timeval& value) {
    return static_cast<double>(value.tv_sec) +
           static_cast<double>(value.tv_usec) / 1000000.0;
}

std::string formatErrors(const std::map<int64_t, uint64_t>& errors) {
    if (errors.empty()) return "none";
    std::ostringstream out;
    bool first = true;
    for (const auto& [code, count] : errors) {
        if (!first) out << ';';
        first = false;
        out << code << ':' << count;
    }
    return out.str();
}

void fillPatterns(RegisteredGpuBuffer& buffer,
                  const std::vector<uint64_t>& ids, size_t value_size) {
    for (size_t i = 0; i < ids.size(); ++i) {
        checkCuda(cudaMemset(buffer.slot(i, value_size), patternForId(ids[i]),
                             value_size),
                  "cudaMemset(pattern)");
    }
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(pattern)");
}

bool verifyPattern(void* device_ptr, uint64_t id,
                   std::vector<uint8_t>& host_buffer) {
    checkCuda(cudaMemcpy(host_buffer.data(), device_ptr, host_buffer.size(),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(D2H verify)");
    const uint8_t expected = patternForId(id);
    return std::all_of(host_buffer.begin(), host_buffer.end(),
                       [expected](uint8_t value) { return value == expected; });
}

PhaseResult runPhase(const PhaseSpec& spec,
                     const std::shared_ptr<RealClient>& client,
                     const ReplicateConfig& replicate_config,
                     const std::vector<std::unique_ptr<RegisteredGpuBuffer>>& buffers) {
    PhaseResult result;
    result.name = spec.name;
    result.write_begin = spec.write_start;

    std::atomic<uint64_t> finite_next{0};
    std::atomic<uint64_t> write_next{spec.write_start};
    std::atomic<bool> abort_phase{false};
    std::mutex merge_mutex;
    std::mutex exception_mutex;
    std::exception_ptr worker_exception;
    std::barrier<> start_barrier(FLAGS_threads + 1);
    Clock::time_point start_time;
    Clock::time_point end_time;
    rusage usage_before{};
    rusage usage_after{};
    std::vector<std::thread> workers;
    workers.reserve(FLAGS_threads);

    for (int worker_id = 0; worker_id < FLAGS_threads; ++worker_id) {
        workers.emplace_back([&, worker_id]() {
            ThreadStats local;
            bool joined_barrier = false;
            try {
                checkCuda(cudaSetDevice(FLAGS_gpu_id), "worker cudaSetDevice");
                auto& gpu_buffer = *buffers[worker_id];
                std::mt19937_64 rng(FLAGS_seed +
                                    static_cast<uint64_t>(worker_id));
                std::uniform_int_distribution<int> ratio_distribution(0, 99);
                std::uniform_int_distribution<uint64_t> read_distribution(
                    spec.read_start,
                    spec.read_count == 0
                        ? spec.read_start
                        : spec.read_start + spec.read_count - 1);
                std::vector<uint8_t> verify_host;
                if (spec.verify_reads) verify_host.resize(FLAGS_value_size);

                start_barrier.arrive_and_wait();
                joined_barrier = true;
                while (!abort_phase.load(std::memory_order_relaxed)) {
                    if (spec.finite_count == 0 && Clock::now() >= end_time) {
                        break;
                    }

                    Operation request_op = spec.operation;
                    if (request_op == Operation::kMixed) {
                        request_op = ratio_distribution(rng) < FLAGS_read_ratio
                                         ? Operation::kRead
                                         : Operation::kWrite;
                    }

                    size_t count = static_cast<size_t>(FLAGS_batch_size);
                    std::vector<uint64_t> ids;
                    ids.reserve(count);
                    if (spec.finite_count != 0) {
                        const uint64_t begin = finite_next.fetch_add(count);
                        if (begin >= spec.finite_count) break;
                        count = static_cast<size_t>(std::min<uint64_t>(
                            count, spec.finite_count - begin));
                        for (size_t i = 0; i < count; ++i) {
                            ids.push_back(spec.read_start + begin + i);
                        }
                    } else if (request_op == Operation::kWrite) {
                        const uint64_t begin = write_next.fetch_add(count);
                        for (size_t i = 0; i < count; ++i) {
                            ids.push_back(begin + i);
                        }
                    } else {
                        std::unordered_set<uint64_t> selected;
                        while (ids.size() < count) {
                            const uint64_t id = read_distribution(rng);
                            if (selected.insert(id).second) ids.push_back(id);
                        }
                    }

                    const bool is_write = request_op == Operation::kWrite ||
                                          request_op == Operation::kPrepare;
                    if (is_write && spec.patterned_writes) {
                        fillPatterns(gpu_buffer, ids, FLAGS_value_size);
                    }

                    std::vector<std::string> keys;
                    std::vector<void*> pointers;
                    std::vector<size_t> sizes(count, FLAGS_value_size);
                    keys.reserve(count);
                    pointers.reserve(count);
                    const std::string& prefix =
                        is_write ? spec.write_prefix : spec.read_prefix;
                    for (size_t i = 0; i < count; ++i) {
                        keys.push_back(makeKey(prefix, ids[i]));
                        pointers.push_back(
                            gpu_buffer.slot(i, FLAGS_value_size));
                    }

                    const auto request_start = Clock::now();
                    std::vector<int64_t> codes;
                    codes.reserve(count);
                    if (count == 1 && is_write) {
                        codes.push_back(client->put_from(
                            keys[0], pointers[0], sizes[0], replicate_config));
                    } else if (count == 1) {
                        codes.push_back(
                            client->get_into(keys[0], pointers[0], sizes[0]));
                    } else if (is_write) {
                        const auto write_results = client->batch_put_from(
                            keys, pointers, sizes, replicate_config);
                        for (int code : write_results) {
                            codes.push_back(code);
                        }
                    } else {
                        codes = client->batch_get_into(keys, pointers, sizes);
                    }
                    const auto request_end = Clock::now();

                    ++local.requests;
                    if ((local.requests - 1) % FLAGS_latency_sample_every == 0 &&
                        local.latencies_us.size() <
                            FLAGS_max_latency_samples_per_thread) {
                        local.latencies_us.push_back(
                            std::chrono::duration<double, std::micro>(
                                request_end - request_start)
                                .count());
                    }

                    if (codes.size() != count) {
                        local.failed_kvs += count;
                        local.errors[-1] += count;
                        continue;
                    }
                    for (size_t i = 0; i < count; ++i) {
                        bool success =
                            is_write
                                ? codes[i] == 0
                                : codes[i] == static_cast<int64_t>(
                                                  FLAGS_value_size);
                        if (success && !is_write && spec.verify_reads) {
                            success = verifyPattern(pointers[i], ids[i],
                                                    verify_host);
                            if (!success) codes[i] = -2;
                        }
                        if (success) {
                            ++local.successful_kvs;
                            local.bytes += FLAGS_value_size;
                        } else {
                            ++local.failed_kvs;
                            local.errors[codes[i]]++;
                        }
                    }
                }
            } catch (...) {
                abort_phase.store(true, std::memory_order_relaxed);
                {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    if (!worker_exception) {
                        worker_exception = std::current_exception();
                    }
                }
                if (!joined_barrier) start_barrier.arrive_and_drop();
            }

            std::lock_guard<std::mutex> lock(merge_mutex);
            mergeStats(result, std::move(local));
        });
    }

    getrusage(RUSAGE_SELF, &usage_before);
    start_time = Clock::now();
    end_time = start_time + std::chrono::seconds(spec.duration_sec);
    start_barrier.arrive_and_wait();
    for (auto& worker : workers) worker.join();
    if (worker_exception) std::rethrow_exception(worker_exception);
    result.elapsed_sec =
        std::chrono::duration<double>(Clock::now() - start_time).count();
    getrusage(RUSAGE_SELF, &usage_after);
    result.cpu_user_sec = timevalSeconds(usage_after.ru_utime) -
                          timevalSeconds(usage_before.ru_utime);
    result.cpu_system_sec = timevalSeconds(usage_after.ru_stime) -
                            timevalSeconds(usage_before.ru_stime);
    result.write_end = write_next.load();
    std::sort(result.latencies_us.begin(), result.latencies_us.end());
    return result;
}

void printResult(const PhaseResult& result) {
    const double seconds = std::max(result.elapsed_sec, 1e-9);
    const double gib_per_sec =
        static_cast<double>(result.bytes) / seconds / (1024.0 * 1024 * 1024);
    const double cpu_percent =
        (result.cpu_user_sec + result.cpu_system_sec) / seconds * 100.0;
    std::cout << "\n[" << result.name << "]\n"
              << "elapsed_sec=" << std::fixed << std::setprecision(3)
              << result.elapsed_sec << " requests=" << result.requests
              << " success_kvs=" << result.successful_kvs
              << " failed_kvs=" << result.failed_kvs << '\n'
              << "throughput_gib_s=" << gib_per_sec
              << " kv_iops=" << result.successful_kvs / seconds
              << " request_qps=" << result.requests / seconds << '\n'
              << "cpu_user_sec=" << result.cpu_user_sec
              << " cpu_system_sec=" << result.cpu_system_sec
              << " cpu_percent_one_core=" << cpu_percent << '\n'
              << "latency_us(request): p50="
              << percentile(result.latencies_us, 0.50)
              << " p95=" << percentile(result.latencies_us, 0.95)
              << " p99=" << percentile(result.latencies_us, 0.99)
              << " p999=" << percentile(result.latencies_us, 0.999)
              << " max="
              << (result.latencies_us.empty() ? 0 : result.latencies_us.back())
              << " samples=" << result.latencies_us.size() << '\n'
              << "errors=" << formatErrors(result.errors) << std::endl;
}

std::string csvEscape(const std::string& value) {
    if (value.find_first_of(",\"") == std::string::npos) return value;
    std::string escaped = "\"";
    for (char ch : value) escaped += ch == '\"' ? "\"\"" : std::string(1, ch);
    return escaped + "\"";
}

void appendCsv(const PhaseResult& result) {
    if (FLAGS_csv_path.empty()) return;
    const bool write_header = !std::filesystem::exists(FLAGS_csv_path) ||
                              std::filesystem::file_size(FLAGS_csv_path) == 0;
    std::ofstream output(FLAGS_csv_path, std::ios::app);
    if (!output) {
        throw std::runtime_error("cannot open --csv_path=" + FLAGS_csv_path);
    }
    if (write_header) {
        output << "phase,operation,value_size,threads,batch_size,elapsed_sec,"
                  "requests,success_kvs,failed_kvs,bytes,gib_per_sec,kv_iops,"
                  "request_qps,cpu_user_sec,cpu_system_sec,cpu_percent_one_core,"
                  "p50_us,p95_us,p99_us,p999_us,max_us,errors\n";
    }
    const double seconds = std::max(result.elapsed_sec, 1e-9);
    output << csvEscape(result.name) << ',' << FLAGS_operation << ','
           << FLAGS_value_size << ',' << FLAGS_threads << ',' << FLAGS_batch_size
           << ',' << result.elapsed_sec << ',' << result.requests << ','
           << result.successful_kvs << ',' << result.failed_kvs << ','
           << result.bytes << ','
           << static_cast<double>(result.bytes) / seconds /
                  (1024.0 * 1024 * 1024)
           << ',' << result.successful_kvs / seconds << ','
           << result.requests / seconds << ',' << result.cpu_user_sec << ','
           << result.cpu_system_sec << ','
           << (result.cpu_user_sec + result.cpu_system_sec) / seconds * 100.0
           << ','
           << percentile(result.latencies_us, 0.50) << ','
           << percentile(result.latencies_us, 0.95) << ','
           << percentile(result.latencies_us, 0.99) << ','
           << percentile(result.latencies_us, 0.999) << ','
           << (result.latencies_us.empty() ? 0 : result.latencies_us.back())
           << ',' << csvEscape(formatErrors(result.errors)) << '\n';
}

bool cleanupRange(const std::shared_ptr<RealClient>& client,
                  const std::string& prefix, uint64_t begin, uint64_t end) {
    uint64_t removed = 0;
    uint64_t failed = 0;
    for (uint64_t cursor = begin; cursor < end;) {
        const size_t count = static_cast<size_t>(std::min<uint64_t>(
            FLAGS_cleanup_batch_size, end - cursor));
        std::vector<std::string> keys;
        keys.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            keys.push_back(makeKey(prefix, cursor + i));
        }
        const auto results = client->batchRemove(keys, FLAGS_cleanup_force);
        for (int code : results) {
            code == 0 ? ++removed : ++failed;
        }
        if (results.size() != count) failed += count - results.size();
        cursor += count;
    }
    std::cout << "cleanup prefix=" << prefix << " range=[" << begin << ','
              << end << ") removed=" << removed << " failed=" << failed
              << std::endl;
    return failed == 0;
}

void validateFlags(Operation operation) {
    if (FLAGS_local_hostname.empty() ||
        mooncake::ResolveMooncakeHostId(FLAGS_local_hostname).empty()) {
        throw std::invalid_argument(
            "--local_hostname must use a non-loopback host or IP plus a port");
    }
    if (FLAGS_threads <= 0 || FLAGS_batch_size <= 0) {
        throw std::invalid_argument("--threads and --batch_size must be positive");
    }
    if (FLAGS_cleanup_batch_size <= 0 || FLAGS_latency_sample_every == 0 ||
        FLAGS_max_latency_samples_per_thread == 0) {
        throw std::invalid_argument(
            "cleanup and latency sampling limits must be positive");
    }
    if (FLAGS_value_size == 0 || FLAGS_value_size % kGdsAlignment != 0) {
        throw std::invalid_argument("--value_size must be a positive multiple of 4096");
    }
    if (FLAGS_value_size >
        std::numeric_limits<size_t>::max() /
            static_cast<uint64_t>(FLAGS_batch_size)) {
        throw std::invalid_argument("value_size * batch_size overflows size_t");
    }
    if (FLAGS_key_prefix.empty()) {
        throw std::invalid_argument("--key_prefix must not be empty");
    }
    if (FLAGS_num_objects == 0 && operation != Operation::kWrite) {
        throw std::invalid_argument("--num_objects must be positive");
    }
    if (FLAGS_num_objects > std::numeric_limits<uint64_t>::max() -
                                FLAGS_key_start) {
        throw std::invalid_argument("key_start + num_objects overflows uint64");
    }
    if ((operation == Operation::kRead || operation == Operation::kMixed) &&
        FLAGS_num_objects < static_cast<uint64_t>(FLAGS_batch_size)) {
        throw std::invalid_argument(
            "read and mixed workloads require num_objects >= batch_size");
    }
    if (FLAGS_prepare_before_run && operation != Operation::kRead &&
        operation != Operation::kMixed) {
        throw std::invalid_argument(
            "--prepare_before_run is only valid for read or mixed");
    }
    if (operationIsTimed(operation) && FLAGS_duration_sec <= 0) {
        throw std::invalid_argument("--duration_sec must be positive");
    }
    if (FLAGS_warmup_sec < 0 || FLAGS_read_ratio < 0 ||
        FLAGS_read_ratio > 100) {
        throw std::invalid_argument("invalid warmup duration or read ratio");
    }
    if (FLAGS_memory_replica_num < 0 || FLAGS_nof_replica_num < 0 ||
        FLAGS_gds_replica_num < 0) {
        throw std::invalid_argument("replica counts must not be negative");
    }
    const bool writes = operationWrites(operation) || FLAGS_prepare_before_run;
    if (writes && FLAGS_memory_replica_num + FLAGS_nof_replica_num +
                          FLAGS_gds_replica_num ==
                      0) {
        throw std::invalid_argument("a write requires at least one replica");
    }
    if (writes && FLAGS_gds_replica_num > 0 &&
        FLAGS_allow_destructive_gds != "YES") {
        throw std::invalid_argument(
            "GDS writes require --allow_destructive_gds=YES and a dedicated test namespace");
    }
    if (FLAGS_gds_replica_num > 0 && std::getenv("MC_USE_TENT") == nullptr &&
        std::getenv("MC_USE_TEV1") == nullptr) {
        throw std::invalid_argument(
            "GDS requires MC_USE_TENT=1 (or the legacy MC_USE_TEV1)");
    }
}

PhaseSpec prepareSpec(const std::string& name) {
    return PhaseSpec{.name = name,
                     .operation = Operation::kPrepare,
                     .read_prefix = FLAGS_key_prefix,
                     .write_prefix = FLAGS_key_prefix,
                     .read_start = FLAGS_key_start,
                     .read_count = FLAGS_num_objects,
                     .write_start = FLAGS_key_start,
                     .finite_count = FLAGS_num_objects,
                     .patterned_writes = true};
}

}  // namespace

int main(int argc, char** argv) {
    gflags::SetUsageMessage(
        "End-to-end Mooncake Store GDS SSD benchmark using registered CUDA buffers");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    std::shared_ptr<RealClient> client;
    int exit_code = 0;
    try {
        const Operation operation = parseOperation(FLAGS_operation);
        validateFlags(operation);
        checkCuda(cudaSetDevice(FLAGS_gpu_id), "cudaSetDevice");

        client = RealClient::create();
        const int setup_rc = client->setup_real(
            FLAGS_local_hostname, FLAGS_metadata_server,
            FLAGS_global_segment_size, FLAGS_local_buffer_size, FLAGS_protocol,
            FLAGS_device_name, FLAGS_master_server, nullptr, "", false, "",
            FLAGS_tenant_id);
        if (setup_rc != 0) {
            throw std::runtime_error("Store setup_real failed: " +
                                     std::to_string(setup_rc));
        }

        std::cout << "operation=" << FLAGS_operation
                  << " local_hostname=" << FLAGS_local_hostname
                  << " gpu=" << FLAGS_gpu_id
                  << " value_size=" << FLAGS_value_size
                  << " threads=" << FLAGS_threads
                  << " batch_size=" << FLAGS_batch_size
                  << " replicas(memory/nof/gds)=" << FLAGS_memory_replica_num
                  << '/' << FLAGS_nof_replica_num << '/'
                  << FLAGS_gds_replica_num << " preferred_gds_segments="
                  << (FLAGS_preferred_gds_segments.empty()
                          ? "<allocator-selected>"
                          : FLAGS_preferred_gds_segments)
                  << std::endl;

        if (operation == Operation::kCleanup) {
            exit_code = cleanupRange(client, FLAGS_key_prefix, FLAGS_key_start,
                                     FLAGS_key_start + FLAGS_num_objects)
                            ? 0
                            : 2;
        } else {
            const size_t gpu_bytes = static_cast<size_t>(
                FLAGS_value_size * static_cast<uint64_t>(FLAGS_batch_size));
            std::vector<std::unique_ptr<RegisteredGpuBuffer>> buffers;
            buffers.reserve(FLAGS_threads);
            for (int i = 0; i < FLAGS_threads; ++i) {
                buffers.push_back(std::make_unique<RegisteredGpuBuffer>(
                    client, FLAGS_gpu_id, gpu_bytes));
            }
            const ReplicateConfig replicate_config = makeReplicateConfig();

            auto run_and_report = [&](const PhaseSpec& spec, bool write_csv) {
                auto result = runPhase(spec, client, replicate_config, buffers);
                printResult(result);
                if (write_csv) appendCsv(result);
                if (result.failed_kvs != 0) exit_code = 2;
                return result;
            };
            auto cleanup_and_record = [&](const std::string& prefix,
                                          uint64_t begin, uint64_t end) {
                if (!cleanupRange(client, prefix, begin, end)) exit_code = 2;
            };

            if (operation == Operation::kPrepare) {
                run_and_report(prepareSpec("prepare"), true);
                if (FLAGS_cleanup_after_run) {
                    cleanup_and_record(FLAGS_key_prefix, FLAGS_key_start,
                                       FLAGS_key_start + FLAGS_num_objects);
                }
            } else if (operation == Operation::kVerify) {
                run_and_report(prepareSpec("verify-prepare"), false);
                PhaseSpec verify{.name = "verify-read",
                                 .operation = Operation::kRead,
                                 .read_prefix = FLAGS_key_prefix,
                                 .write_prefix = FLAGS_key_prefix,
                                 .read_start = FLAGS_key_start,
                                 .read_count = FLAGS_num_objects,
                                 .write_start = FLAGS_key_start,
                                 .finite_count = FLAGS_num_objects,
                                 .verify_reads = true};
                run_and_report(verify, true);
                if (FLAGS_cleanup_after_run) {
                    cleanup_and_record(FLAGS_key_prefix, FLAGS_key_start,
                                       FLAGS_key_start + FLAGS_num_objects);
                }
            } else if (operation == Operation::kCheck) {
                PhaseSpec check{.name = "check-existing",
                                .operation = Operation::kRead,
                                .read_prefix = FLAGS_key_prefix,
                                .write_prefix = FLAGS_key_prefix,
                                .read_start = FLAGS_key_start,
                                .read_count = FLAGS_num_objects,
                                .write_start = FLAGS_key_start,
                                .finite_count = FLAGS_num_objects,
                                .verify_reads = true};
                run_and_report(check, true);
                if (FLAGS_cleanup_after_run) {
                    cleanup_and_record(FLAGS_key_prefix, FLAGS_key_start,
                                       FLAGS_key_start + FLAGS_num_objects);
                }
            } else {
                if (FLAGS_prepare_before_run) {
                    run_and_report(prepareSpec("prepare-before-run"), false);
                }

                if (FLAGS_warmup_sec > 0) {
                    PhaseSpec warmup{.name = "warmup",
                                     .operation = operation,
                                     .read_prefix = FLAGS_key_prefix,
                                     .write_prefix = FLAGS_key_prefix + ".warmup",
                                     .read_start = FLAGS_key_start,
                                     .read_count = FLAGS_num_objects,
                                     .write_start = 0,
                                     .duration_sec = FLAGS_warmup_sec,
                                     .verify_reads = false};
                    auto warmup_result = run_and_report(warmup, false);
                    if (operation == Operation::kWrite ||
                        operation == Operation::kMixed) {
                        cleanup_and_record(warmup.write_prefix,
                                           warmup_result.write_begin,
                                           warmup_result.write_end);
                    }
                }

                const uint64_t timed_write_start =
                    operation == Operation::kMixed
                        ? FLAGS_key_start + FLAGS_num_objects
                        : FLAGS_key_start;
                PhaseSpec measured{.name = "measured",
                                   .operation = operation,
                                   .read_prefix = FLAGS_key_prefix,
                                   .write_prefix = FLAGS_key_prefix,
                                   .read_start = FLAGS_key_start,
                                   .read_count = FLAGS_num_objects,
                                   .write_start = timed_write_start,
                                   .duration_sec = FLAGS_duration_sec,
                                   .verify_reads = FLAGS_verify_reads};
                auto measured_result = run_and_report(measured, true);

                if (FLAGS_cleanup_after_run) {
                    if (operation == Operation::kWrite ||
                        operation == Operation::kMixed) {
                        cleanup_and_record(measured.write_prefix,
                                           measured_result.write_begin,
                                           measured_result.write_end);
                    }
                    if (operation == Operation::kRead ||
                        operation == Operation::kMixed) {
                        cleanup_and_record(
                            FLAGS_key_prefix, FLAGS_key_start,
                            FLAGS_key_start + FLAGS_num_objects);
                    }
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "gds_store_bench: " << error.what() << std::endl;
        exit_code = 1;
    }

    if (client) {
        const int rc = client->tearDownAll();
        if (rc != 0) {
            std::cerr << "tearDownAll failed: " << rc << std::endl;
            if (exit_code == 0) exit_code = 2;
        }
    }
    google::ShutdownGoogleLogging();
    return exit_code;
}
