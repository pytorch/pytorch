// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <mma.h>
#include <stdint.h>

#define WARP_SIZE 32
#define F_M 16
#define F_N 16
#define F_K 16

using namespace nvcuda;

// CUDA kernel that initializes the GPU buffers

__global__ void
set_array(float* __restrict__ const a, float value, uint32_t len) {
  uint32_t num_threads = blockDim.x * gridDim.x;
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    a[idx] = value;
    idx += num_threads;
  }
}

// Generates a couple of floating point random numbers using tensor cores
// matrix multiplications.

__global__ void tensor_cores_rng(
    float* __restrict__ const d_c,
    float val_init,
    uint32_t iters) {
  wmma::fragment<wmma::matrix_a, F_M, F_N, F_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, F_M, F_N, F_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, F_M, F_N, F_K, float> c_frag;

  // Thread index
  uint32_t num_warps = gridDim.x * blockDim.x / WARP_SIZE;
  uint32_t warp_idx = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;

  // Use simple linear congruential generators for random value generators
  // X0 = (a * Seed + c) mod m
  // X1 = (a * X0 + c) mod m ...
  uint32_t a = 48271;
  uint32_t c = 0;
  uint32_t m = 2 ^ 31 - 1;
  uint32_t x0 = (a * warp_idx + c) % m;
  uint32_t x1 = (a * x0 + c) % m;
  uint32_t x2 = (a * x1 + c) % m;

  float val_init_a = ((float)x0 / (float)m) + 1.0;
  float val_init_b = ((float)x1 / (float)m) - 1.0;
  float val_init_c = ((float)x2 / (float)m) + 1.5;

  // wmma::fill_fragment(a_frag, val_init_a);
  // wmma::fill_fragment(b_frag, val_init_b);
  // wmma::fill_fragment(c_frag, val_init_c + val_init);

  // Run several iterations
  for (int i = 0; i < iters; ++i) {
    // Perform the matrix multiplication using tensor cores
    // Do the unrolling manually to improve latency hiding
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  // Store the output
  int output_offset = warp_idx * F_M * F_N;
  wmma::store_matrix_sync(d_c + output_offset, c_frag, 16, wmma::mem_row_major);
}
