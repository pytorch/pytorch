#pragma once

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"

#ifdef __HIP_PLATFORM_HCC__
#define SEGREDUCE_MINBLOCKS 8
#else
#define SEGREDUCE_MINBLOCKS 16
#endif

namespace caffe2 {


static inline __device__ void gpuAtomicAdd(float* address, float val) {
  atomicAdd(address, val);
}

static inline __device__ void gpuAtomicAdd(c10::Half* address, c10::Half val) {
#if (                         \
    (CUDA_VERSION < 10000) || \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    at::Half hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = hsum + val;
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
#else
  atomicAdd(reinterpret_cast<__half*>(address), val);
#endif
}

template <typename SIndex, typename TParam, typename T, bool ExactBlock = false>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void rowwise_sparse_adagrad_fused_length_sum_gradient_kernel(
    const int* __restrict__ prefix_sum_length_data, // prefix of lengths
                                                    // (offsets for the
                                                    // segments)
    int N, // number of rows (hash size) of embedding table
    int post, // embedding dimension size
    int len_length, // number of segments
    const float epsilon,
    TParam* param,
    T* param_mom,
    const SIndex* indices,
    const T* __restrict__ grad,
    const float* lr,
    float weight_decay = 0.f) {
  const float LR = lr[0];
  // len_length blocks, each block process one segment
  int group = blockIdx.x; // the group-th segment
  int start = group == 0
      ? 0
      : prefix_sum_length_data[group - 1]; // start offset of the segment
  int end = prefix_sum_length_data[group]; // end offset of the segment
  CUDA_KERNEL_ASSERT(start <= N);
  CUDA_KERNEL_ASSERT(end <= N);

  if (ExactBlock) {
    // Specialize WarpReduce for type float
    typedef cub::WarpReduce<float> WarpReduce;
    // Allocate WarpReduce shared memory for 32 warps, 1024 / 32 = 32
    __shared__ typename WarpReduce::TempStorage temp_storage[32];

    const size_t gradIdx = group * post + threadIdx.x; // index for grad
    for (int line = start + threadIdx.y; line < end; line += blockDim.y) {
      // line: the idx in the indices
      // threadIdx.x: index in the embedding dimension
      const SIndex index =
          indices[line]; // the index-th row in the embedding table
      float sum_squares = 0.0;
      __shared__ float row_sum_squares_avg;

      // post == blockDim.x
      const size_t paramIdx = index * post + threadIdx.x; // index for param
      const float x_ij = grad[gradIdx] + weight_decay * param[paramIdx];
      sum_squares += x_ij * x_ij;

      // Return the warp-wide sums to each lane0 (threads 0, 32, 64, 96, ...)
      int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
      float reduce_result = WarpReduce(temp_storage[warp_id]).Sum(sum_squares);

      if ((threadIdx.y * blockDim.x + threadIdx.x) % 32 == 0) {
        row_sum_squares_avg = reduce_result / static_cast<float>(post);
        // AtomicAdd when the embedding dim is larger than 32.
        // param_mom[index] += row_sum_squares_avg;
        gpuAtomicAdd(&param_mom[index], static_cast<T>(row_sum_squares_avg));
      }
      __syncthreads();

      // update param
      float step = LR / (sqrtf(param_mom[index]) + epsilon);
      param[paramIdx] = param[paramIdx] + x_ij * step;
    }
  } else {
    // TODO: Tuning NumThreads for sum_squares
    typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
    __shared__ BlockReduce::TempStorage temp_storage;
    int valid = min(post, blockDim.x);

    for (int line = start; line < end; ++line) {
      // line: the idx in the indices
      const SIndex index = indices[line]; // the index row in the embedding
      float sum_squares = 0.0;
      __shared__ float row_sum_squares_avg;

      for (int i = threadIdx.x; i < post; i += blockDim.x) {
        // i: index in the embedding dimension
        const float x_ij =
            grad[group * post + i] + weight_decay * param[index * post + i];
        sum_squares += x_ij * x_ij;
      }
      float reduce_result = BlockReduce(temp_storage).Sum(sum_squares, valid);

      if (threadIdx.x == 0) {
        row_sum_squares_avg = reduce_result / static_cast<float>(post);
        float mom_new = param_mom[index] + static_cast<T>(row_sum_squares_avg);
        param_mom[index] = mom_new;
      }
      __syncthreads();

      // update param
      float step = LR / (sqrtf(param_mom[index]) + epsilon);
      for (int i = threadIdx.x; i < post; i += blockDim.x) {
        size_t paramIdx = index * post + i; // index for param
        float x_ij = grad[group * post + i] + weight_decay * param[paramIdx];
        float param_new = param[paramIdx] + x_ij * step;
        // float param_new1 = param[paramIdx];
        // printf("step %f, x_ij %f", step, x_ij);
        param[paramIdx] = param_new;
      }
    }
  }
}

} // namespace caffe2
