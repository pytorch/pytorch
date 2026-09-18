/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cupti_activity.h>
#include <libkineto.h>
#include <unistd.h>

#include <folly/init/Init.h>

#include <ApproximateClock.h>
#include "kineto/libkineto/stress_test/stress_test_input.h"
#include "kineto/libkineto/stress_test/utils.h"
#include "mpi.h"

using namespace kineto_stress_test;

void trace_collection_thread(
    uint32_t trace_delay_us,
    uint32_t trace_length_us,
    uint32_t cupti_buffer_mb) {
  if (cupti_buffer_mb > 0) {
    // Configure CUPTI buffer sizes
    size_t attrValue = 0;
    size_t attrValueSize = sizeof(size_t);
    attrValue = (size_t)(cupti_buffer_mb * 1024 * 1024);
    cuptiActivitySetAttribute(
        CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue);
  }

  // Config Kineto
  std::set<libkineto::ActivityType> types = {
      libkineto::ActivityType::CONCURRENT_KERNEL,
      libkineto::ActivityType::GPU_MEMCPY,
      libkineto::ActivityType::GPU_MEMSET,
      libkineto::ActivityType::CUDA_RUNTIME,
      libkineto::ActivityType::EXTERNAL_CORRELATION,
      libkineto::ActivityType::OVERHEAD,
      libkineto::ActivityType::COLLECTIVE_COMM};
  auto& profiler = libkineto::api().activityProfiler();
  libkineto::api().initProfilerIfRegistered();
  profiler.prepareTrace(types);

  // Wait a bit before collecting the trace
  usleep(trace_delay_us);

  // Collect the trace
  profiler.startTrace();
  usleep(trace_length_us);
  auto trace = profiler.stopTrace();

  // Save the trace
  std::string kTraceFile = "/tmp/kineto_stress_test_trace_";
  kTraceFile.append(std::to_string(getpid()));
  kTraceFile.append(".json");
  trace->save(kTraceFile);
}

void uvm_allocation_thread(stress_test_args* test_args) {
  uint64_t alloc_size = test_args->uvm_len * sizeof(float);

  std::cout << "UVM is used. Allocation size (MB) = "
            << 2 * alloc_size / (1024 * 1024) << std::endl;
  int currentDevice = 0;
  checkCudaStatus(cudaGetDevice(&currentDevice), __LINE__);
  checkCudaStatus(
      cudaMallocManaged(
          (void**)&test_args->uvm_a, alloc_size, cudaMemAttachGlobal),
      __LINE__);
  checkCudaStatus(
      cudaMallocManaged(
          (void**)&test_args->uvm_b, alloc_size, cudaMemAttachGlobal),
      __LINE__);
  checkCudaStatus(
      cudaMemAdvise(
          (void*)test_args->uvm_a,
          alloc_size,
          cudaMemAdviseSetPreferredLocation,
          cudaCpuDeviceId),
      __LINE__);
  checkCudaStatus(
      cudaMemAdvise(
          (void*)test_args->uvm_b,
          alloc_size,
          cudaMemAdviseSetPreferredLocation,
          cudaCpuDeviceId),
      __LINE__);
  checkCudaStatus(
      cudaMemAdvise(
          (void*)test_args->uvm_a,
          alloc_size,
          cudaMemAdviseSetAccessedBy,
          currentDevice),
      __LINE__);
  checkCudaStatus(
      cudaMemAdvise(
          (void*)test_args->uvm_b,
          alloc_size,
          cudaMemAdviseSetAccessedBy,
          currentDevice),
      __LINE__);
  std::cout << "UVM buffers allocated. Initializing them with values."
            << std::endl;

  // Put a bunch of non-zero values into the UVM buffers
  srand(time(nullptr));
  for (uint64_t i = 0; i < 32 * 128 * 1024; ++i) {
    uint64_t idx_a = rand() % test_args->uvm_len;
    uint64_t idx_b = rand() % test_args->uvm_len;
    test_args->uvm_a[idx_a] =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    test_args->uvm_b[idx_b] =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  std::cout << "UVM buffers initialized." << std::endl;
}

void run_parallel_stress_test(stress_test_args test_args) {
  std::vector<std::thread> v_workers;
  if (test_args.num_workers > 1) {
    v_workers.reserve(test_args.num_workers);
    for (int i = 0; std::cmp_less(i, test_args.num_workers); ++i) {
      v_workers.emplace_back(
          run_stress_test, i, test_args.num_workers, test_args);
    }
    for (auto& t : v_workers) {
      t.join();
    }
    v_workers.clear();
  } else {
    run_stress_test(0, 1, test_args);
  }
}

void create_cuda_streams(stress_test_args& test_args) {
  test_args.compute_streams = (cudaStream_t*)malloc(
      test_args.num_cuda_streams * test_args.num_workers *
      sizeof(cudaStream_t));
  for (uint32_t i = 0; i < test_args.num_cuda_streams * test_args.num_workers;
       ++i) {
    checkCudaStatus(
        cudaStreamCreateWithFlags(
            test_args.compute_streams + i, cudaStreamNonBlocking),
        __LINE__);
  }

  if (test_args.use_memcpy_stream) {
    test_args.memcpy_streams =
        (cudaStream_t*)malloc(test_args.num_workers * sizeof(cudaStream_t));
    for (uint32_t i = 0; i < test_args.num_workers; ++i) {
      // We want to test the effect of CUDA stream priorities on the order of
      // memory transfers.
      if (i % 2 != 0) {
        checkCudaStatus(
            cudaStreamCreateWithFlags(
                test_args.memcpy_streams + i, cudaStreamNonBlocking),
            __LINE__);
      } else {
        int leastPriority = 0;
        int greatestPriority = 0;
        checkCudaStatus(
            cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority),
            __LINE__);
        checkCudaStatus(
            cudaStreamCreateWithPriority(
                test_args.memcpy_streams + i,
                cudaStreamNonBlocking,
                leastPriority),
            __LINE__);
      }
    }
  }

  if (test_args.use_uvm_stream) {
    test_args.uvm_streams =
        (cudaStream_t*)malloc(test_args.num_workers * sizeof(cudaStream_t));
    for (uint32_t i = 0; i < test_args.num_workers; ++i) {
      checkCudaStatus(
          cudaStreamCreateWithFlags(
              test_args.uvm_streams + i, cudaStreamNonBlocking),
          __LINE__);
    }
  }
}

void cleanup_cuda_streams(stress_test_args& test_args) {
  for (int i = 0; i < test_args.num_cuda_streams * test_args.num_workers; ++i) {
    checkCudaStatus(
        cudaStreamSynchronize(test_args.compute_streams[i]), __LINE__);
    checkCudaStatus(cudaStreamDestroy(test_args.compute_streams[i]), __LINE__);
  }

  if (test_args.compute_streams) {
    free(test_args.compute_streams);
  }

  if (test_args.use_memcpy_stream) {
    for (int i = 0; std::cmp_less(i, test_args.num_workers); ++i) {
      checkCudaStatus(
          cudaStreamSynchronize(test_args.memcpy_streams[i]), __LINE__);
      checkCudaStatus(cudaStreamDestroy(test_args.memcpy_streams[i]), __LINE__);
    }

    if (test_args.memcpy_streams) {
      free(test_args.memcpy_streams);
    }
  }

  if (test_args.use_uvm_stream) {
    for (int i = 0; std::cmp_less(i, test_args.num_workers); ++i) {
      checkCudaStatus(
          cudaStreamSynchronize(test_args.uvm_streams[i]), __LINE__);
      checkCudaStatus(cudaStreamDestroy(test_args.uvm_streams[i]), __LINE__);
    }

    if (test_args.uvm_streams) {
      free(test_args.uvm_streams);
    }
  }
}

int main(int argc, char* argv[]) {
  /////////////////////////////////////////////////////////////////////////////
  // Read test configuration
  /////////////////////////////////////////////////////////////////////////////

  int rank = 0;
  int num_ranks = 0;

  folly::init(&argc, &argv);

  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_ranks));

  tensor_cache_args cache_args;
  stress_test_args test_args;

  if (argc < 2) {
    std::cout << "Please specify input JSON file." << std::endl;
    std::cout << "Usage: " << argv[0] << " <json_file>" << std::endl;
    return -1;
  }

  // Parse input json file
  read_inputs_from_json(argv[1], &test_args, &cache_args);
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (test_args.is_multi_rank) {
    test_args.num_workers = 1;
    test_args.num_ranks = num_ranks;
    test_args.rank = rank;
    std::cout
        << "When running in multi-rank mode, only a single worker can be used!"
        << std::endl;
  } else {
    test_args.rank = 0;
    test_args.num_ranks = 1;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Start test
  /////////////////////////////////////////////////////////////////////////////

  // Initialize multi-processing
  if (test_args.is_multi_rank) {
    checkCudaStatus(cudaSetDevice(test_args.rank));

    // Broadcast ID to other ranks
    if (test_args.rank == 0) {
      NCCLCHECK(ncclGetUniqueId(&nccl_id));
    }
    MPICHECK(MPI_Bcast(
        (void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Create the communicator
    NCCLCHECK(ncclCommInitRank(
        &nccl_communicator, test_args.num_ranks, nccl_id, test_args.rank));

    // Allocate memory for the buffers
    checkCudaStatus(
        cudaMalloc(&pBuffNCCLSend, test_args.sz_nccl_buff_KB * 1024), __LINE__);
    checkCudaStatus(
        cudaMalloc(&pBuffNCCLRecv, test_args.sz_nccl_buff_KB * 1024), __LINE__);
    checkCudaStatus(
        cudaMemset(pBuffNCCLSend, 0, test_args.sz_nccl_buff_KB * 1024),
        __LINE__);
    checkCudaStatus(
        cudaMemset(pBuffNCCLRecv, 0, test_args.sz_nccl_buff_KB * 1024),
        __LINE__);

    // Make sure all the processes have allocated this memory
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  // Pre-allocate CUDA streams
  if (test_args.pre_alloc_streams) {
    create_cuda_streams(test_args);
  }

  std::thread uvm_init_thread;
  if (test_args.use_uvm_buffers) {
    if (test_args.parallel_uvm_alloc) {
      uvm_init_thread = std::thread(uvm_allocation_thread, &test_args);
    } else {
      uvm_allocation_thread(&test_args);
      std::cout << "Rank " << test_args.rank << " finished UVM init."
                << std::endl;
    }
  }

  generate_tensor_cache(cache_args);
  std::cout << "Rank " << test_args.rank
            << " generating tensor cache completed." << std::endl;

  if (test_args.use_uvm_buffers) {
    if (test_args.parallel_uvm_alloc) {
      uvm_init_thread.join();
      std::cout << "Rank " << test_args.rank << " finished UVM init."
                << std::endl;
    }
  }

  if (test_args.is_multi_rank) {
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  // Warmup run
  uint32_t num_test_operations = test_args.num_operations;
  test_args.num_operations = 200;
  run_stress_test(0, 1, test_args);
  test_args.num_operations = num_test_operations;
  std::cout << "Rank " << test_args.rank << " warmup completed." << std::endl;

  // Re-generate tensor cache values
  re_initialize_buffer_values();
  if (test_args.is_multi_rank) {
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  double t_no_trace = 0.0;
  clock_t t_start;
  clock_t t_stop;

  if (test_args.do_warmup) {
    // Run without kineto tracing
    t_start = clock();
    run_parallel_stress_test(test_args);
    t_stop = clock();
    t_no_trace = (double)(t_stop - t_start) / 1e+3;
    std::cout << "Rank " << test_args.rank
              << " before kineto tracing. Duration (ms) = " << t_no_trace
              << std::endl;

    // Re-generate tensor cache values
    re_initialize_buffer_values();
    if (test_args.is_multi_rank) {
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
  }

  // If configured we are collecting a few traces
  if (test_args.trace_length_us > 0) {
    // Tracing thread
    std::thread kineto_thread;

    // We are gradually increasing the GPU memory usage so that we have GPU
    // traces being collected while we are almost out-of-memory. This is an
    // attempt to expose errors that we often see in our fleet like: illegal
    // instruction, uncorrectable NVLink error, etc.

    for (uint32_t idx = 0; idx < cache_args.num_increments; ++idx) {
      // Run with kineto tracing
      t_start = clock();
      kineto_thread = std::thread(
          trace_collection_thread,
          test_args.trace_delay_us,
          test_args.trace_length_us,
          test_args.cupti_buffer_mb);
      run_parallel_stress_test(test_args);
      kineto_thread.join();
      t_stop = clock();
      double t_with_trace = (double)(t_stop - t_start) / 1e+3;

      std::cout << "Rank " << test_args.rank << " kineto run " << idx
                << " completed. Used GPU memory (MB) = "
                << sz_memory_pool_KB / 1024
                << "; Duration (ms) = " << t_with_trace << std::endl;

      // The first run is the default run
      add_pairs_to_tensor_cache(cache_args, cache_args.num_pairs_per_increment);
    }

    // Re-generate tensor cache values
    re_initialize_buffer_values();
    if (test_args.is_multi_rank) {
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
  } else {
    std::cout << "Rank " << test_args.rank
              << " has tracing disabled (trace_length = 0)!" << std::endl;
  }

  // Run again after kineto tracing
  t_start = clock();
  run_parallel_stress_test(test_args);
  t_stop = clock();
  double t_after_trace = (double)(t_stop - t_start) / 1e+3;
  std::cout << "Rank " << test_args.rank
            << " after kineto tracing. Duration (ms) = " << t_after_trace
            << "; Kernel Launch Throughput = "
            << (double)test_args.num_operations / (t_after_trace / 1000)
            << " kernels/second" << std::endl;

  // Final barrier before destroying everything
  if (test_args.is_multi_rank) {
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  // Destroy CUDA streams
  if (test_args.pre_alloc_streams) {
    cleanup_cuda_streams(test_args);
  }

  // Free the tensor cache on the GPU
  free_tensor_cache();

  // Free UVM
  if (test_args.use_uvm_buffers) {
    checkCudaStatus(cudaFree(test_args.uvm_a), __LINE__);
    checkCudaStatus(cudaFree(test_args.uvm_b), __LINE__);
  }

  if (test_args.is_multi_rank) {
    checkCudaStatus(cudaFree(pBuffNCCLRecv));
    checkCudaStatus(cudaFree(pBuffNCCLSend));
    NCCLCHECK(ncclCommDestroy(nccl_communicator));
    MPICHECK(MPI_Finalize());
  }

  return 0;
}
