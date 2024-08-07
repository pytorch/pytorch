// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <sys/types.h>
#include <cstdint>

namespace facebook::gpu_power_damper {

class cuda_power_damper {
  // Confguration of the power damper
  uint32_t _gpu_idx;
  uint32_t _tdp_pct_start;
  uint32_t _tdp_pct_stop;
  uint32_t _drain_milliseconds;
  uint32_t _steps;

  // Internal parameters for power generation
  uint32_t _num_multiprocessors;
  uint32_t _block_size;
  uint32_t _buffer_elements;
  uint32_t _power_gen_kernel_us;

  // The power damper will only run CUDA kernels on a low priority stream
  cudaStream_t _damper_cuda_stream;

  // Device buffer that we want persistent to avoid frequent mallocs outside
  // to what the caching allocator can do.
  float *_dev_buffer;

  // Heuristic parameter designed to avoid tanking workflow performance.
  // The idea is to measure power virus execution time while the GPU is
  // idle and then rememsure as we drain the power. If the kernel is
  // running for much longer, we can stop the power drainage because
  // we can assume that the GPU is executing something else.
  float _tolerated_slowdown = 2.5;
  uint32_t _strike_slowdowns = 2;

 public:
  cuda_power_damper(
      uint32_t gpu_idx,
      uint32_t tdp_pct_start,
      uint32_t tdp_pct_stop,
      uint32_t drain_milliseconds,
      uint32_t steps)
      : _gpu_idx(gpu_idx),
        _tdp_pct_start(tdp_pct_start),
        _tdp_pct_stop(tdp_pct_stop),
        _drain_milliseconds(drain_milliseconds),
        _steps(steps) {
    this->_num_multiprocessors = 132;
    this->_block_size = 1024;
    uint32_t numWarps = this->_block_size / 32 * this->_num_multiprocessors;
    this->_buffer_elements = numWarps * 16 * 16;
    this->_power_gen_kernel_us = 1000;
    this->_dev_buffer = nullptr;
  }

  ~cuda_power_damper() {}

  void initialize_power_gen_params();

  int gen_and_drain_power();

  uint32_t get_gpu_idx() {
    return _gpu_idx;
  };
};

} // namespace facebook::gpu_power_damper
