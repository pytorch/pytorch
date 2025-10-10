#pragma once
#include <c10/macros/Macros.h>

// Marks a lambda as executable on both the host and device. The __host__
// attribute is important so that we can access static type information from
// the host, even if the function is typically only executed on the device.
#ifndef GPU_LAMBDA
#define GPU_LAMBDA __host__ __device__
#endif

#if defined(USE_ROCM)
constexpr int num_threads() {
  return 256;
}

constexpr int thread_work_size() { return 4; }
#else
constexpr uint32_t num_threads() {
  return C10_WARP_SIZE * 4;
}

constexpr int thread_work_size() { return 8; }
#endif

constexpr int block_work_size() { return thread_work_size() * num_threads(); }
