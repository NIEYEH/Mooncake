# Add allocator benchmark executable
add_executable(allocator_bench allocator_bench.cpp)
target_link_libraries(allocator_bench PRIVATE cachelib_memory_allocator
                                              mooncake_store)

# Add master benchmark executable
add_executable(master_bench master_bench.cpp)
target_link_libraries(master_bench PRIVATE cachelib_memory_allocator
                                           mooncake_store)

# Add BatchGetReplicaList latency benchmark executable
add_executable(batch_get_replica_bench batch_get_replica_bench.cpp)
target_link_libraries(batch_get_replica_bench PRIVATE cachelib_memory_allocator
                                                       mooncake_store)

# Add file interface benchmark executable
add_executable(file_interface_bench file_interface_bench.cpp)
target_link_libraries(file_interface_bench PRIVATE mooncake_store glog::glog)

# Add storage backend benchmark executable
# This benchmark tests all storage backends (OffsetAllocator, Bucket, FilePerKey)
# with realistic KV cache workloads for LLM inference scenarios
add_executable(storage_backend_bench storage_backend_bench.cpp)
target_link_libraries(
  storage_backend_bench PRIVATE mooncake_store cachelib_memory_allocator
                                gflags::gflags glog::glog pthread)

# Add allocation strategy benchmark executable
# This benchmark tests AllocationStrategy performance with configurable
# segment counts, allocation sizes, replica counts, and workload patterns
add_executable(allocation_strategy_bench allocation_strategy_bench.cpp)
target_link_libraries(
  allocation_strategy_bench PRIVATE mooncake_store cachelib_memory_allocator
                                    gflags::gflags glog::glog pthread)

# Add NoF worker pool benchmark executable
if(USE_NOF)
  add_executable(nof_worker_pool_bench nof_worker_pool_bench.cpp)
  target_link_libraries(nof_worker_pool_bench PRIVATE mooncake_store glog::glog
                                                       gflags::gflags)
endif()

# End-to-end Store benchmark for GPU buffers backed by GDS SSD replicas.
if(USE_TENT AND USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(gds_store_bench gds_store_bench.cpp)
  target_link_libraries(
    gds_store_bench PRIVATE mooncake_store gflags::gflags glog::glog
                            CUDA::cudart pthread)
endif()
