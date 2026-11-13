/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "utils.h"

#pragma once

namespace kineto_stress_test {

// The unique identifier used for NCCL communication
extern ncclUniqueId nccl_id;

// The communicator object used for communication
extern ncclComm_t nccl_communicator;

// NCCL buffers on the device
extern float* pBuffNCCLSend;
extern float* pBuffNCCLRecv;

struct stress_test_args {
  // Number of threads that run the stress test
  uint32_t num_workers{1};

  // Number of operations per stress test
  uint32_t num_operations{10000};

  // Can improve event density. Cuda streams per worker
  uint32_t num_cuda_streams{1};

  // Simulates cuda mallocs happening in the PT Cuda Cache Allocator
  double prob_cuda_malloc{0.001};

  // The min number of compute iterations in the cuda kernel. If high
  // this reduces event density.
  uint32_t min_iters_kernel{1};

  // The max number of compute iterations in the cuda kernel. If high
  // this reduces event density.
  uint32_t max_iters_kernel{5};

  // Controls the number of threads per block for the LCG kernel
  uint32_t kernel_block_size{256};

  // Controls the number of CUDA thread blocks for the LCG kernel
  uint32_t thread_blocks_per_kernel{256};

  // The probability that instead of a kernel call we do a memset on the
  // input buffers, using a magic value
  double memset_prob{0.05};

  // The min idle time between kernel launches in microseconds
  uint32_t min_idle_us{1};

  // The max idle time between kernel launches in microseconds
  uint32_t max_idle_us{2};

  // If true, a warmup cycle is ran before a profiled run
  bool do_warmup{false};

  // If true, we randomly sleep a number of microseconds between kernel
  // launches.
  bool simulate_host_time{false};

  // If non-zero, we allocate UVM memory and use it
  bool use_uvm_buffers{false};

  // Size of a single buffer in FP32 elements in UVM
  uint64_t uvm_len{0};

  // If set true, the UVM allocation and initialization will be done in parallel
  // with cache allocation (e.g. cudaHostAlloc)
  bool parallel_uvm_alloc{false};

  // The probability of running a kernel that uses UVM
  double uvm_kernel_prob{0.001};

  // If true we need to run the binary using MPI on multiple ranks
  bool is_multi_rank{false};

  // Number of parallel processes to be spawned via MPI
  int32_t num_ranks{1};

  // Use this variable to pin a process to a specific GPU index.
  // Do not modify!
  int32_t rank{0};

  // Size of the NCCL buffers which needs to be at least the size
  // of the largest tensor
  uint32_t sz_nccl_buff_KB{1024};

  // Number of iterations between NCCL sync calls
  uint32_t num_iters_nccl_sync{100};

  // If true, we pre-allocate CUDA streams and reuse them throughout
  // the experiment
  bool pre_alloc_streams{false};

  // If true, h2d and d2h transfers would be scheduled on their own
  // CUDA stream
  bool use_memcpy_stream{false};

  // If true, kernels using UVM would be scheduled on their own
  // CUDA stream
  bool use_uvm_stream{false};

  // If true, we use cudaGetMemInfo throughout the stress test to
  // measure peak memory usage
  bool monitor_mem_usage{false};

  // Number of microseconds to wait before the trace collection starts
  uint32_t trace_delay_us{3000000};

  // Number of microseconds for trace collection. If 0 the trace is
  // not collected
  uint32_t trace_length_us{1000000};

  // Size of the CUPTI activity buffer in MB. If it's 0, we don't
  // explicitly set a value
  uint32_t cupti_buffer_mb{0};

  /* VARIABLES */

  // The CUDA streams vector
  cudaStream_t* compute_streams{nullptr};

  // The explicit memcpy stream
  cudaStream_t* memcpy_streams{nullptr};

  // The explicit UVM stream
  cudaStream_t* uvm_streams{nullptr};

  // UVM buffers
  float* uvm_a{nullptr};
  float* uvm_b{nullptr};
};

// We are using this to reduce the number of code lines
struct lcg_kernel_input {
  float const* __restrict__ d_a;
  float const* __restrict__ d_b;
  float* __restrict__ d_c;
  int len;
  float const* __restrict__ uvm_a;
  float const* __restrict__ uvm_b;
  uint64_t uvm_len;
  int iters;
};

// Use this function to vary the kernel name at runtime
void call_compute_kernel(
    uint32_t thread_blocks,
    uint32_t threads_per_block,
    uint32_t shmem_sz,
    cudaStream_t stream,
    lcg_kernel_input kernel_args,
    uint32_t op_id);

void run_stress_test(
    uint32_t thread_id,
    uint32_t num_workers,
    stress_test_args test_args);

} // namespace kineto_stress_test
