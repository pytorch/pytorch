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

// Random number generation constants. They should not be modified, otherwise
// it's likely that most values will converge to 0.
#define LCG_A 8121
#define LCG_C 28411
#define LCG_M 134456

// Size of the memory pool in kilobytes
extern uint32_t sz_memory_pool_KB;

// Number of tensor pairs in the memory pool
extern uint32_t num_tensor_pairs;

struct tensor_pair {
  // Number of elements in the float arrays
  uint32_t n_elements;

  // If true, we pre-generate host buffers and use copy host to dev
  bool b_copy_h2d;

  // If true we download the output buffer to simulate output download
  bool b_copy_d2h;

  // GPU buffers
  float* d_A;
  float* d_B;
  float* d_C;

  // Host buffers
  float* h_A;
  float* h_B;
  float* h_C;
  bool h_C_pinned;
};

// The memory pool object
extern tensor_pair* p_memory_pool;

struct tensor_cache_args {
  // Sets GPU memory utilization
  uint32_t sz_cache_KB{1024 * 128};

  // If small, density is higher due to shorter kernel times
  uint32_t sz_min_tensor_KB{16};

  // If large, we will have kernels with high duration thus smaller
  // event density. That's because kernels will have to run on larger
  // buffer sizes.
  uint32_t sz_max_tensor_KB{2048};

  // Sets the maximum GPU memory
  uint32_t sz_GPU_memory_KB{1024 * 1024 * 16};

  // Simulates the chance of uploading a batch to the GPU.
  // It reduces event density if it's set too high
  double prob_h2d{0.005};

  // Simulates the chance of downloading results from the GPU.
  // It reduces event density if it's set too high
  double prob_d2h{0.0001};

  // Number of increments in the GPU memory usage to see what happens at the
  // peak memory usage.
  uint32_t num_increments{1};
  uint32_t num_pairs_per_increment{1};
};

// Generates all the buffer pairs, using a minimum and a maximum size.
// prob_h2d is the probability that we will generate a copy from host to
// device, in which case we need to pre-generate the buffers on the host.
void generate_tensor_cache(tensor_cache_args cache_args);

// Empties the tensor cache and reallocates it
void free_and_realloc_tensor_pairs(
    tensor_pair* tensor_pair,
    cudaStream_t stream);

// For some experiments we may need to add additional pairs to stress the
// GPU memory limits
void add_pairs_to_tensor_cache(
    tensor_cache_args cache_args,
    uint32_t num_added_pairs);

// Re-initializes the random values in the device buffers
void re_initialize_buffer_values();

// Frees the host and device tensors
void free_tensor_cache();

// Host code for initializing the buffers
void simple_lcg_host(float* h_A, int num_elements);

// A CUDA kernel that fills a device buffer with random values
__global__ void simple_rng_lcg(float* d_A, int num_elements);

} // namespace kineto_stress_test
