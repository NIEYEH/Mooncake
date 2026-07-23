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

#include "tent/metrics/tent_metrics.h"

#include <glog/logging.h>
#include <tent/thirdparty/nlohmann/json.h>
#include <sstream>
#include <iomanip>

namespace mooncake::tent {

TentMetrics& TentMetrics::instance() {
    static TentMetrics instance;
    return instance;
}

TentMetrics::~TentMetrics() { shutdown(); }

#if TENT_METRICS_ENABLED

Status TentMetrics::initialize(const MetricsConfig& config) {
    // Use compare_exchange to prevent race condition during initialization
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true)) {
        return Status::OK();  // Already initialized by another thread
    }

    config_ = config;

    // Set runtime enabled state from config
    runtime_enabled_.store(config_.enabled, std::memory_order_relaxed);

    // Configure histogram buckets if provided (recreate histograms)
    // Note: config latency_buckets are in seconds, convert to microseconds for
    // histogram
    if (!config_.latency_buckets.empty()) {
        // Convert seconds to microseconds for histogram buckets
        std::vector<double> latency_buckets_us;
        latency_buckets_us.reserve(config_.latency_buckets.size());
        for (double bucket_sec : config_.latency_buckets) {
            latency_buckets_us.push_back(bucket_sec *
                                         1000000.0);  // seconds -> microseconds
        }
        read_latency_ = ylt::metric::histogram_t(
            "tent_read_latency_us", "Read latency distribution in microseconds",
            latency_buckets_us);
        write_latency_ = ylt::metric::histogram_t(
            "tent_write_latency_us",
            "Write latency distribution in microseconds", latency_buckets_us);
    }

    // Configure size histogram buckets if provided
    if (!config_.size_buckets.empty()) {
        read_size_ = ylt::metric::histogram_t(
            "tent_read_size_bytes", "Read request size distribution in bytes",
            config_.size_buckets);
        write_size_ = ylt::metric::histogram_t(
            "tent_write_size_bytes", "Write request size distribution in bytes",
            config_.size_buckets);
    }

    // Register all metrics to vectors for unified serialization
    registerMetrics();

    // Initialize and start HTTP server
    initHttpServer();

    // Start periodic metric reporting thread if interval > 0
    if (config_.report_interval_seconds > 0) {
        metric_report_running_ = true;
        metric_report_thread_ = std::thread([this]() {
            while (metric_report_running_) {
                std::string summary = getSummaryString();
                LOG(INFO) << "TENT Metrics: " << summary;

                // Use condition variable for interruptible sleep
                std::unique_lock<std::mutex> lock(metric_report_mutex_);
                metric_report_cv_.wait_for(
                    lock, std::chrono::seconds(config_.report_interval_seconds),
                    [this]() { return !metric_report_running_.load(); });
            }
        });
    }

    LOG(INFO)
        << "TENT metrics initialized successfully, HTTP server listening on "
        << config_.http_host << ":" << config_.http_port
        << ", runtime_enabled=" << (runtime_enabled_.load() ? "true" : "false");
    return Status::OK();
}

void TentMetrics::initHttpServer() {
    using namespace coro_http;

    // Create HTTP server with configurable threads
    http_server_ = std::make_unique<coro_http_server>(
        config_.http_server_threads, config_.http_port);

    // Register /metrics endpoint for Prometheus
    http_server_->set_http_handler<GET>(
        "/metrics", [this](coro_http_request& req, coro_http_response& resp) {
            std::string metrics = getPrometheusMetrics();
            resp.add_header("Content-Type", "text/plain; version=0.0.4");
            resp.set_status_and_content(status_type::ok, std::move(metrics));
        });

    // Register /metrics/summary endpoint for human-readable summary
    http_server_->set_http_handler<GET>(
        "/metrics/summary",
        [this](coro_http_request& req, coro_http_response& resp) {
            std::string summary = getSummaryString();
            resp.add_header("Content-Type", "text/plain");
            resp.set_status_and_content(status_type::ok, std::move(summary));
        });

    // Register /metrics/json endpoint for JSON format
    http_server_->set_http_handler<GET>(
        "/metrics/json",
        [this](coro_http_request& req, coro_http_response& resp) {
            std::string json = getJsonMetrics();
            resp.add_header("Content-Type", "application/json");
            resp.set_status_and_content(status_type::ok, std::move(json));
        });

    // Register /health endpoint for health check
    http_server_->set_http_handler<GET>(
        "/health", [](coro_http_request& req, coro_http_response& resp) {
            resp.add_header("Content-Type", "text/plain");
            resp.set_status_and_content(status_type::ok, "OK");
        });

    // Start the HTTP server asynchronously
    http_server_->async_start();
}

void TentMetrics::shutdown() {
    if (!initialized_) return;

    // Stop metric reporting thread
    metric_report_running_ = false;
    metric_report_cv_.notify_all();  // Wake up the sleeping thread immediately
    if (metric_report_thread_.joinable()) {
        metric_report_thread_.join();
    }

    // Stop HTTP server
    if (http_server_) {
        http_server_->stop();
        http_server_.reset();
    }

    // Clear metric vectors
    counters_.clear();
    gauges_.clear();
    histograms_.clear();
    histogram_boundaries_.clear();

    initialized_ = false;
    LOG(INFO) << "TENT metrics shutdown complete";
}

void TentMetrics::registerMetrics() {
    // Pre-allocate vectors to avoid reallocation
    counters_.reserve(28);
    gauges_.reserve(16);
    histograms_.reserve(23);
    histogram_boundaries_.reserve(23);

    // Register all counters - add new counters here
    counters_ = {
        &read_bytes_total_,     &write_bytes_total_,   &read_requests_total_,
        &write_requests_total_, &read_failures_total_, &write_failures_total_,
        &failover_total_, &gds_handle_registration_failures_total_,
        &gds_buffer_registration_failures_total_,
        &gds_batch_submit_failures_total_,
        &gds_input_requests_total_, &gds_merged_requests_total_,
        &gds_logical_bytes_total_, &gds_physical_ios_total_,
        &gds_physical_bytes_total_, &gds_physical_batches_total_,
        &gds_transport_submit_calls_total_,
        &gds_transport_logical_requests_total_,
        &gds_transport_physical_ios_created_total_,
        &gds_small_requests_total_,
        &gds_underfilled_batches_total_, &gds_dispatch_window_full_total_,
        &gds_read_io_failures_total_, &gds_write_io_failures_total_,
        &gds_read_concurrency_reductions_total_,
        &gds_write_concurrency_reductions_total_,
        &gds_read_concurrency_recoveries_total_,
        &gds_write_concurrency_recoveries_total_,
    };

    gauges_ = {
        &runtime_queue_owners_, &runtime_queue_bytes_,
        &runtime_inflight_owners_, &runtime_inflight_bytes_,
        &gds_read_queue_ios_, &gds_read_queue_bytes_,
        &gds_write_queue_ios_, &gds_write_queue_bytes_,
        &gds_active_read_workers_, &gds_active_write_workers_,
        &gds_read_inflight_limit_, &gds_write_inflight_limit_,
        &gds_read_window_p99_latency_us_,
        &gds_write_window_p99_latency_us_,
        &gds_read_adaptive_queued_ios_,
        &gds_write_adaptive_queued_ios_,
    };
    for (auto* gauge : gauges_) gauge->update(0);

    // Register all histograms - add new histograms here
    // Note: histogram_boundaries_ must match the order of histograms_
    histograms_ = {
        &read_latency_, &write_latency_, &read_size_,
        &write_size_, &deadline_mlu_, &gds_batch_submit_latency_,
        &gds_requests_per_submit_, &gds_physical_ios_per_batch_,
        &gds_physical_batches_per_submit_, &gds_dispatch_queued_batches_,
        &gds_dispatch_inflight_batches_,
        &runtime_queue_read_wait_latency_,
        &runtime_queue_write_wait_latency_,
        &runtime_queue_read_total_latency_,
        &runtime_queue_write_total_latency_,
        &gds_read_queue_wait_latency_, &gds_write_queue_wait_latency_,
        &gds_read_cufile_io_latency_, &gds_write_cufile_io_latency_,
        &gds_read_total_latency_, &gds_write_total_latency_,
        &gds_read_adaptive_p99_latency_,
        &gds_write_adaptive_p99_latency_,
    };
    histogram_boundaries_ = {
        kLatencyBuckets, kLatencyBuckets,     kSizeBuckets,
        kSizeBuckets, kMluPerMilleBuckets, kLatencyBuckets,
        kBatchCountBuckets, kBatchCountBuckets, kBatchCountBuckets,
        kBatchCountBuckets,
        kBatchCountBuckets,
        kLatencyBuckets, kLatencyBuckets, kLatencyBuckets, kLatencyBuckets,
        kLatencyBuckets, kLatencyBuckets, kLatencyBuckets, kLatencyBuckets,
        kLatencyBuckets, kLatencyBuckets, kLatencyBuckets, kLatencyBuckets,
    };
}

void TentMetrics::recordReadCompleted(size_t bytes, double latency_seconds) {
    // Fast path: check runtime switch first
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;

    read_bytes_total_.inc(static_cast<double>(bytes));
    read_requests_total_.inc();
    read_size_.observe(static_cast<int64_t>(bytes));
    if (latency_seconds > 0.0) {
        // Convert seconds to microseconds for histogram (int64_t internally)
        int64_t latency_us = static_cast<int64_t>(latency_seconds * 1000000.0);
        read_latency_.observe(latency_us);
    }
}

void TentMetrics::recordWriteCompleted(size_t bytes, double latency_seconds) {
    // Fast path: check runtime switch first
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;

    write_bytes_total_.inc(static_cast<double>(bytes));
    write_requests_total_.inc();
    write_size_.observe(static_cast<int64_t>(bytes));
    if (latency_seconds > 0.0) {
        // Convert seconds to microseconds for histogram (int64_t internally)
        int64_t latency_us = static_cast<int64_t>(latency_seconds * 1000000.0);
        write_latency_.observe(latency_us);
    }
}

void TentMetrics::recordDeadlineMLU(double mlu) {
    // Fast path: check runtime switch first
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;
    if (mlu < 0.0) return;  // defensive: ignore invalid (e.g. window <= 0)
    // Store in per-mille so the integer histogram can bucket fractional ratios.
    deadline_mlu_.observe(static_cast<int64_t>(mlu * 1000.0));
}

void TentMetrics::recordReadFailed(size_t bytes) {
    // Fast path: check runtime switch first
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;

    read_failures_total_.inc();
    read_requests_total_.inc();  // Count failed requests too
}

void TentMetrics::recordWriteFailed(size_t bytes) {
    // Fast path: check runtime switch first
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;

    write_failures_total_.inc();
    write_requests_total_.inc();  // Count failed requests too
}

void TentMetrics::recordTransportFailover() {
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;

    failover_total_.inc();
}

void TentMetrics::recordGdsHandleRegistrationFailed() {
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;
    gds_handle_registration_failures_total_.inc();
}

void TentMetrics::recordGdsBufferRegistrationFailed() {
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;
    gds_buffer_registration_failures_total_.inc();
}

void TentMetrics::recordGdsBatchSubmitFailed() {
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;
    gds_batch_submit_failures_total_.inc();
}

void TentMetrics::recordGdsCoalescing(size_t input_requests,
                                      size_t merged_requests, size_t bytes) {
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;
    gds_input_requests_total_.inc(static_cast<int64_t>(input_requests));
    gds_merged_requests_total_.inc(static_cast<int64_t>(merged_requests));
    gds_logical_bytes_total_.inc(static_cast<int64_t>(bytes));
}

void TentMetrics::recordGdsTransportSubmission(size_t logical_requests,
                                               size_t physical_ios,
                                               size_t physical_batches,
                                               size_t small_requests) {
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;
    gds_transport_submit_calls_total_.inc();
    gds_transport_logical_requests_total_.inc(
        static_cast<int64_t>(logical_requests));
    gds_transport_physical_ios_created_total_.inc(
        static_cast<int64_t>(physical_ios));
    gds_small_requests_total_.inc(static_cast<int64_t>(small_requests));
    gds_requests_per_submit_.observe(static_cast<int64_t>(logical_requests));
    gds_physical_batches_per_submit_.observe(
        static_cast<int64_t>(physical_batches));
}

void TentMetrics::recordGdsDispatchWindowFull(size_t queued_batches,
                                              size_t inflight_batches) {
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;
    gds_dispatch_window_full_total_.inc();
    gds_dispatch_queued_batches_.observe(
        static_cast<int64_t>(queued_batches));
    gds_dispatch_inflight_batches_.observe(
        static_cast<int64_t>(inflight_batches));
}

void TentMetrics::recordGdsPhysicalBatch(size_t io_count, size_t bytes,
                                         double submit_latency_seconds,
                                         bool underfilled) {
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;
    gds_physical_ios_total_.inc(static_cast<int64_t>(io_count));
    gds_physical_bytes_total_.inc(static_cast<int64_t>(bytes));
    gds_physical_batches_total_.inc();
    if (underfilled) gds_underfilled_batches_total_.inc();
    gds_physical_ios_per_batch_.observe(static_cast<int64_t>(io_count));
    if (submit_latency_seconds > 0.0) {
        gds_batch_submit_latency_.observe(static_cast<int64_t>(
            submit_latency_seconds * 1000000.0));
    }
}

void TentMetrics::recordGdsDirectIo(bool is_read, size_t bytes, bool success,
                                    double queue_wait_seconds,
                                    double io_latency_seconds,
                                    double total_latency_seconds) {
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;

    gds_physical_ios_total_.inc();
    gds_physical_bytes_total_.inc(static_cast<int64_t>(bytes));
    // Preserve the legacy counter: with Batch disabled, every physical
    // submission contains exactly one IO.
    gds_physical_batches_total_.inc();
    gds_physical_ios_per_batch_.observe(1);
    const auto observe_nonnegative_seconds = [](
        ylt::metric::histogram_t& histogram, double seconds) {
        if (seconds >= 0.0) {
            histogram.observe(
                static_cast<int64_t>(seconds * 1000000.0));
        }
    };
    const auto observe_positive_seconds = [](
        ylt::metric::histogram_t& histogram, double seconds) {
        if (seconds > 0.0) {
            histogram.observe(
                static_cast<int64_t>(seconds * 1000000.0));
        }
    };
    if (is_read) {
        if (!success) gds_read_io_failures_total_.inc();
        observe_nonnegative_seconds(gds_read_queue_wait_latency_,
                                    queue_wait_seconds);
        observe_positive_seconds(gds_read_cufile_io_latency_,
                                 io_latency_seconds);
        observe_nonnegative_seconds(gds_read_total_latency_,
                                    total_latency_seconds);
    } else {
        if (!success) gds_write_io_failures_total_.inc();
        observe_nonnegative_seconds(gds_write_queue_wait_latency_,
                                    queue_wait_seconds);
        observe_positive_seconds(gds_write_cufile_io_latency_,
                                 io_latency_seconds);
        observe_nonnegative_seconds(gds_write_total_latency_,
                                    total_latency_seconds);
    }
    observe_positive_seconds(gds_batch_submit_latency_, io_latency_seconds);
}

void TentMetrics::updateGdsIoState(size_t queued_read_ios,
                                   size_t queued_read_bytes,
                                   size_t queued_write_ios,
                                   size_t queued_write_bytes,
                                   size_t active_read_workers,
                                   size_t active_write_workers,
                                   size_t read_inflight_limit,
                                   size_t write_inflight_limit) {
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;
    gds_read_queue_ios_.update(static_cast<int64_t>(queued_read_ios));
    gds_read_queue_bytes_.update(static_cast<int64_t>(queued_read_bytes));
    gds_write_queue_ios_.update(static_cast<int64_t>(queued_write_ios));
    gds_write_queue_bytes_.update(static_cast<int64_t>(queued_write_bytes));
    gds_active_read_workers_.update(
        static_cast<int64_t>(active_read_workers));
    gds_active_write_workers_.update(
        static_cast<int64_t>(active_write_workers));
    gds_read_inflight_limit_.update(
        static_cast<int64_t>(read_inflight_limit));
    gds_write_inflight_limit_.update(
        static_cast<int64_t>(write_inflight_limit));
}

void TentMetrics::observeGdsAdaptiveWindow(bool is_read,
                                           double p99_latency_seconds,
                                           size_t queued_ios) {
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;
    const int64_t p99_us =
        static_cast<int64_t>(p99_latency_seconds * 1000000.0);
    if (is_read) {
        gds_read_window_p99_latency_us_.update(p99_us);
        gds_read_adaptive_queued_ios_.update(
            static_cast<int64_t>(queued_ios));
        gds_read_adaptive_p99_latency_.observe(p99_us);
    } else {
        gds_write_window_p99_latency_us_.update(p99_us);
        gds_write_adaptive_queued_ios_.update(
            static_cast<int64_t>(queued_ios));
        gds_write_adaptive_p99_latency_.observe(p99_us);
    }
}

void TentMetrics::recordGdsAdaptiveConcurrency(bool is_read, bool reduced,
                                               size_t new_limit) {
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;
    if (is_read) {
        if (reduced) {
            gds_read_concurrency_reductions_total_.inc();
        } else {
            gds_read_concurrency_recoveries_total_.inc();
        }
        gds_read_inflight_limit_.update(static_cast<int64_t>(new_limit));
    } else {
        if (reduced) {
            gds_write_concurrency_reductions_total_.inc();
        } else {
            gds_write_concurrency_recoveries_total_.inc();
        }
        gds_write_inflight_limit_.update(static_cast<int64_t>(new_limit));
    }
}

void TentMetrics::updateRuntimeQueue(size_t queued_owners,
                                     size_t queued_bytes,
                                     size_t inflight_owners,
                                     size_t inflight_bytes) {
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed))
        return;
    runtime_queue_owners_.update(static_cast<int64_t>(queued_owners));
    runtime_queue_bytes_.update(static_cast<int64_t>(queued_bytes));
    runtime_inflight_owners_.update(static_cast<int64_t>(inflight_owners));
    runtime_inflight_bytes_.update(static_cast<int64_t>(inflight_bytes));
}

void TentMetrics::recordRuntimeQueueWait(bool is_read,
                                         double latency_seconds) {
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed) ||
        latency_seconds < 0.0) {
        return;
    }
    const int64_t latency_us =
        static_cast<int64_t>(latency_seconds * 1000000.0);
    if (is_read) {
        runtime_queue_read_wait_latency_.observe(latency_us);
    } else {
        runtime_queue_write_wait_latency_.observe(latency_us);
    }
}

void TentMetrics::recordRuntimeQueueTotal(bool is_read,
                                          double latency_seconds) {
    if (!initialized_ || !runtime_enabled_.load(std::memory_order_relaxed) ||
        latency_seconds < 0.0) {
        return;
    }
    const int64_t latency_us =
        static_cast<int64_t>(latency_seconds * 1000000.0);
    if (is_read) {
        runtime_queue_read_total_latency_.observe(latency_us);
    } else {
        runtime_queue_write_total_latency_.observe(latency_us);
    }
}

std::string TentMetrics::getPrometheusMetrics() {
    if (!initialized_) return "";

    try {
        std::string result;
        // Pre-allocate buffer to avoid reallocation during serialization
        result.reserve(kPrometheusBufferSize);

        // Serialize all counters
        for (auto* counter : counters_) {
            counter->serialize(result);
        }

        for (auto* gauge : gauges_) {
            gauge->serialize(result);
        }

        // Serialize all histograms
        for (auto* histogram : histograms_) {
            histogram->serialize(result);
        }

        return result;
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to serialize Prometheus metrics: " << e.what();
        return "";
    }
}

std::string TentMetrics::getJsonMetrics() {
    if (!initialized_) return "{}";

    try {
        nlohmann::json root;

        // Serialize all counters
        for (auto* counter : counters_) {
            root[counter->str_name()] = counter->value();
        }

        for (auto* gauge : gauges_) {
            root[gauge->str_name()] = gauge->value();
        }

        // Serialize all histograms
        for (size_t h = 0; h < histograms_.size(); ++h) {
            auto* histogram = histograms_[h];
            const auto& boundaries = histogram_boundaries_[h];

            auto bucket_counts = histogram->get_bucket_counts();

            // Calculate total count
            int64_t total_count = 0;
            for (auto& bucket : bucket_counts) {
                total_count += bucket->value();
            }

            nlohmann::json hist_obj;
            hist_obj["count"] = total_count;

            nlohmann::json buckets_obj;
            for (size_t i = 0;
                 i < boundaries.size() && i < bucket_counts.size(); ++i) {
                buckets_obj[std::to_string(static_cast<int64_t>(
                    boundaries[i]))] = bucket_counts[i]->value();
            }
            hist_obj["buckets"] = buckets_obj;

            root[histogram->str_name()] = hist_obj;
        }

        return root.dump(2);  // Pretty print with 2-space indent
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to serialize JSON metrics: " << e.what();
        return R"({"error": "Failed to serialize metrics"})";
    }
}

std::string TentMetrics::getSummaryString() {
    if (!initialized_) return "Metrics not initialized";

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    double read_bytes = read_bytes_total_.value();
    double write_bytes = write_bytes_total_.value();
    double read_reqs = read_requests_total_.value();
    double write_reqs = write_requests_total_.value();
    double read_fails = read_failures_total_.value();
    double write_fails = write_failures_total_.value();
    double failovers = failover_total_.value();

    // Format bytes in human-readable form
    auto formatBytes = [](double bytes) -> std::string {
        std::ostringstream s;
        s << std::fixed << std::setprecision(2);
        if (bytes >= 1e12)
            s << bytes / 1e12 << " TB";
        else if (bytes >= 1e9)
            s << bytes / 1e9 << " GB";
        else if (bytes >= 1e6)
            s << bytes / 1e6 << " MB";
        else if (bytes >= 1e3)
            s << bytes / 1e3 << " KB";
        else
            s << bytes << " B";
        return s.str();
    };

    oss << "Read: " << formatBytes(read_bytes) << " ("
        << static_cast<uint64_t>(read_reqs) << " reqs, "
        << static_cast<uint64_t>(read_fails) << " fails) | "
        << "Write: " << formatBytes(write_bytes) << " ("
        << static_cast<uint64_t>(write_reqs) << " reqs, "
        << static_cast<uint64_t>(write_fails) << " fails) | "
        << "Failovers: " << static_cast<uint64_t>(failovers);

    return oss.str();
}

#else  // !TENT_METRICS_ENABLED

// Stub implementations when metrics are disabled at compile time
Status TentMetrics::initialize(const MetricsConfig& config) {
    config_ = config;
    initialized_ = true;
    LOG(INFO)
        << "TENT metrics disabled at compile time (TENT_METRICS_ENABLED=0)";
    return Status::OK();
}

void TentMetrics::shutdown() { initialized_ = false; }

void TentMetrics::recordReadCompleted(size_t, double) {}
void TentMetrics::recordWriteCompleted(size_t, double) {}
void TentMetrics::recordReadFailed(size_t) {}
void TentMetrics::recordWriteFailed(size_t) {}
void TentMetrics::recordTransportFailover() {}
void TentMetrics::recordGdsHandleRegistrationFailed() {}
void TentMetrics::recordGdsBufferRegistrationFailed() {}
void TentMetrics::recordGdsBatchSubmitFailed() {}
void TentMetrics::recordGdsCoalescing(size_t, size_t, size_t) {}
void TentMetrics::recordGdsTransportSubmission(size_t, size_t, size_t,
                                               size_t) {}
void TentMetrics::recordGdsDispatchWindowFull(size_t, size_t) {}
void TentMetrics::recordGdsPhysicalBatch(size_t, size_t, double, bool) {}
void TentMetrics::recordGdsDirectIo(bool, size_t, bool, double, double,
                                    double) {}
void TentMetrics::updateGdsIoState(size_t, size_t, size_t, size_t, size_t,
                                   size_t, size_t, size_t) {}
void TentMetrics::observeGdsAdaptiveWindow(bool, double, size_t) {}
void TentMetrics::recordGdsAdaptiveConcurrency(bool, bool, size_t) {}
void TentMetrics::updateRuntimeQueue(size_t, size_t, size_t, size_t) {}
void TentMetrics::recordRuntimeQueueWait(bool, double) {}
void TentMetrics::recordRuntimeQueueTotal(bool, double) {}
void TentMetrics::recordDeadlineMLU(double) {}

std::string TentMetrics::getPrometheusMetrics() {
    return "# TENT metrics disabled at compile time\n";
}

std::string TentMetrics::getJsonMetrics() {
    return R"({"status": "disabled", "message": "TENT metrics disabled at compile time"})";
}

std::string TentMetrics::getSummaryString() {
    return "TENT metrics disabled at compile time";
}

#endif  // TENT_METRICS_ENABLED

}  // namespace mooncake::tent
