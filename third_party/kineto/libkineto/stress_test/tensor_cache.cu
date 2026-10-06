/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdio.h>
#include <unistd.h>

#include "tensor_cache.cuh"

namespace kineto_stress_test {

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#define RNG_SEED 1025

// A kernel that fills a device buffer with random values
__global__ void simple_rng_lcg(float* d_A, int num_elements) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < num_elements) {
    uint32_t xn = tid * (tid + 1);
    d_A[tid] = (float)((LCG_A * xn + LCG_C) % LCG_M);
  }
}

// Fill the buffers on the host with random values
void simple_lcg_host(float* h_A, int num_elements) {
  uint32_t xn = rand();
  for (int i = 0; i < num_elements; ++i) {
    h_A[i] = (float)((LCG_A * xn + LCG_C) % LCG_M);
    xn = h_A[i];
  }
}

// We pre-create a memory pool of buffers on which we do various operations.
// This is similar to the tensor cache that PyTorch is managing.
tensor_pair* p_memory_pool;

// Size of the memory pool in kilobytes
uint32_t sz_memory_pool_KB;

// Number of tensor pairs in the memory pool
uint32_t num_tensor_pairs;

void add_pairs_to_tensor_cache(
    tensor_cache_args cache_args,
    uint32_t num_added_pairs) {
  uint32_t num_current_pairs = num_tensor_pairs;

  for (uint32_t i = num_current_pairs; i < num_current_pairs + num_added_pairs;
       ++i) {
    uint32_t num_KB =
        rand() % (cache_args.sz_max_tensor_KB - cache_args.sz_min_tensor_KB) +
        cache_args.sz_min_tensor_KB;
    uint32_t num_elements = num_KB * 1024 / sizeof(float);

    // Allocate device buffers
    p_memory_pool[i].n_elements = num_elements;
    checkCudaStatus(
        cudaMalloc(&p_memory_pool[i].d_A, num_elements * sizeof(float)),
        __LINE__);
    checkCudaStatus(
        cudaMalloc(&p_memory_pool[i].d_B, num_elements * sizeof(float)),
        __LINE__);
    checkCudaStatus(
        cudaMalloc(&p_memory_pool[i].d_C, num_elements * sizeof(float)),
        __LINE__);

    // Initialize device buffers with random values
    uint32_t thread_blocks = num_elements / 256;
    simple_rng_lcg<<<thread_blocks, 256>>>(
        p_memory_pool[i].d_A, p_memory_pool[i].n_elements);
    CUDA_KERNEL_LAUNCH_CHECK();
    simple_rng_lcg<<<thread_blocks, 256>>>(
        p_memory_pool[i].d_B, p_memory_pool[i].n_elements);
    CUDA_KERNEL_LAUNCH_CHECK();
    simple_rng_lcg<<<thread_blocks, 256>>>(
        p_memory_pool[i].d_C, p_memory_pool[i].n_elements);
    CUDA_KERNEL_LAUNCH_CHECK();

    // Throw a dice to see if we will do memcopy host to device for this one and
    // use pinned memory
    if (((float)(rand() % 32767) / 32767.0) < cache_args.prob_h2d) {
      p_memory_pool[i].b_copy_h2d = true;
      checkCudaStatus(
          cudaHostAlloc(
              &p_memory_pool[i].h_A,
              num_elements * sizeof(float),
              cudaHostAllocDefault),
          __LINE__);
      // checkCudaStatus(cudaHostAlloc(&p_memory_pool[i].h_B, num_elements *
      // sizeof(float), cudaHostAllocDefault), __LINE__);
      p_memory_pool[i].h_B = (float*)malloc(sizeof(float) * num_elements);

      simple_lcg_host(p_memory_pool[i].h_A, num_elements);
      simple_lcg_host(p_memory_pool[i].h_B, num_elements);
    } else {
      p_memory_pool[i].b_copy_h2d = false;
      p_memory_pool[i].h_A = NULL;
      p_memory_pool[i].h_B = NULL;
    }

    // Simulate output download
    if (((float)(rand() % 32767) / 32767.0) < cache_args.prob_d2h) {
      p_memory_pool[i].b_copy_d2h = true;
      // Make 50% of the D2H on pageable and 50% on pinned memory
      if (rand() % 2 == 1) {
        checkCudaStatus(
            cudaHostAlloc(
                &p_memory_pool[i].h_C,
                num_elements * sizeof(float),
                cudaHostAllocDefault),
            __LINE__);
        p_memory_pool[i].h_C_pinned = true;
      } else {
        p_memory_pool[i].h_C = (float*)malloc(sizeof(float) * num_elements);
        p_memory_pool[i].h_C_pinned = false;
      }
      simple_lcg_host(p_memory_pool[i].h_C, num_elements);
    } else {
      p_memory_pool[i].b_copy_d2h = false;
      p_memory_pool[i].h_C = NULL;
    }

    // Now we have a new tensor pair
    num_tensor_pairs++;
    sz_memory_pool_KB += (3 * num_KB);
  }
}

void generate_tensor_cache(tensor_cache_args cache_args) {
  // Estimate the number of tensor pairs
  uint32_t num_pairs_max = cache_args.sz_GPU_memory_KB /
      (3 * (cache_args.sz_max_tensor_KB - cache_args.sz_min_tensor_KB) / 2);

  // Number of actual pairs
  num_tensor_pairs = 0;

  // At firs the pool is empty
  sz_memory_pool_KB = 0;

  // Pre-allocate num_pairs_max and if num_tensor_pairs comes lower, well,
  // that's life
  p_memory_pool = (tensor_pair*)malloc(num_pairs_max * sizeof(tensor_pair));

  // Start creating the pool
  srand(RNG_SEED);
  for (int i = 0; i < num_pairs_max; ++i) {
    // If we allocated too much, just exit
    if (sz_memory_pool_KB >= cache_args.sz_cache_KB) {
      printf("Allocated %d tensor pairs.\n", num_tensor_pairs);
      break;
    }

    add_pairs_to_tensor_cache(cache_args, 1);
  }
}

void re_initialize_buffer_values() {
  for (uint32_t i = 0; i < num_tensor_pairs; ++i) {
    uint32_t num_elements = p_memory_pool[i].n_elements;

    // Initialize device buffers with random values
    uint32_t thread_blocks = num_elements / 256;
    simple_rng_lcg<<<thread_blocks, 256>>>(
        p_memory_pool[i].d_A, p_memory_pool[i].n_elements);
    CUDA_KERNEL_LAUNCH_CHECK();
    simple_rng_lcg<<<thread_blocks, 256>>>(
        p_memory_pool[i].d_B, p_memory_pool[i].n_elements);
    CUDA_KERNEL_LAUNCH_CHECK();
    simple_rng_lcg<<<thread_blocks, 256>>>(
        p_memory_pool[i].d_C, p_memory_pool[i].n_elements);
    CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void free_and_realloc_tensor_pairs(
    tensor_pair* tensor_pair,
    cudaStream_t stream) {
  checkCudaStatus(cudaFree(tensor_pair->d_A), __LINE__);
  checkCudaStatus(cudaFree(tensor_pair->d_B), __LINE__);
  checkCudaStatus(cudaFree(tensor_pair->d_C), __LINE__);

  // Allocate device buffers
  uint32_t num_elements = tensor_pair->n_elements;
  checkCudaStatus(
      cudaMalloc(&tensor_pair->d_A, num_elements * sizeof(float)), __LINE__);
  checkCudaStatus(
      cudaMalloc(&tensor_pair->d_B, num_elements * sizeof(float)), __LINE__);
  checkCudaStatus(
      cudaMalloc(&tensor_pair->d_C, num_elements * sizeof(float)), __LINE__);

  if (tensor_pair->b_copy_h2d) {
    checkCudaStatus(cudaFreeHost(tensor_pair->h_A), __LINE__);
    checkCudaStatus(cudaFreeHost(tensor_pair->h_B), __LINE__);

    checkCudaStatus(
        cudaHostAlloc(
            &tensor_pair->h_A,
            num_elements * sizeof(float),
            cudaHostAllocDefault),
        __LINE__);
    checkCudaStatus(
        cudaHostAlloc(
            &tensor_pair->h_B,
            num_elements * sizeof(float),
            cudaHostAllocDefault),
        __LINE__);

    simple_lcg_host(tensor_pair->h_A, num_elements);
    simple_lcg_host(tensor_pair->h_B, num_elements);
  }

  if (tensor_pair->b_copy_d2h) {
    checkCudaStatus(cudaFreeHost(tensor_pair->h_C), __LINE__);
    checkCudaStatus(
        cudaHostAlloc(
            &tensor_pair->h_C,
            num_elements * sizeof(float),
            cudaHostAllocDefault),
        __LINE__);
    simple_lcg_host(tensor_pair->h_C, num_elements);
  }
}

void free_tensor_cache() {
  for (uint32_t i = 0; i < num_tensor_pairs; ++i) {
    checkCudaStatus(cudaFree(p_memory_pool[i].d_A), __LINE__);
    checkCudaStatus(cudaFree(p_memory_pool[i].d_B), __LINE__);
    checkCudaStatus(cudaFree(p_memory_pool[i].d_C), __LINE__);

    if (p_memory_pool[i].b_copy_h2d) {
      if (p_memory_pool[i].h_A) {
        checkCudaStatus(cudaFreeHost(p_memory_pool[i].h_A), __LINE__);
        p_memory_pool[i].h_A = NULL;
      }

      if (p_memory_pool[i].h_B) {
        // checkCudaStatus(cudaFreeHost(p_memory_pool[i].h_B), __LINE__);
        free(p_memory_pool[i].h_B);
        p_memory_pool[i].h_B = NULL;
      }

      if (p_memory_pool[i].h_C) {
        if (p_memory_pool[i].h_C_pinned) {
          checkCudaStatus(cudaFreeHost(p_memory_pool[i].h_C), __LINE__);
        } else {
          free(p_memory_pool[i].h_C);
        }
        p_memory_pool[i].h_C = NULL;
      }
    }
  }

  if (p_memory_pool) {
    free(p_memory_pool);
    p_memory_pool = NULL;
  }

  size_t mem_free = 0;
  size_t mem_total = 0;
  checkCudaStatus(cudaMemGetInfo(&mem_free, &mem_total), __LINE__);
  size_t mem_used = (mem_total - mem_free) / 1024 / 1024;

  printf("GPU MB after freeing tensor cache: %6zu\n", mem_used);
}

} // namespace kineto_stress_test
