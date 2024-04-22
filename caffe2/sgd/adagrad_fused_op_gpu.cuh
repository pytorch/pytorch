#pragma once

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <curand_kernel.h>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/GpuAtomics.cuh"

#if defined(USE_ROCM)
#define SEGREDUCE_MINBLOCKS 8
#else
#define SEGREDUCE_MINBLOCKS 16
#endif

// Whoever include this header should define REDUCE_BLOCK_SIZE
// which is the maximum row-wise length
// Default is 1024 (maxThreads per block in Volta GPU)
#ifdef REDUCE_BLOCK_SIZE
#define REDUCE_SIZE REDUCE_BLOCK_SIZE
#else
#define REDUCE_SIZE 1024
#endif

namespace caffe2 {

constexpr int kWarpSize = 32;

template <typename T>
inline __device__ T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
#if !defined(USE_ROCM)
  return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
  return __shfl_xor(val, laneMask, width);
#endif
}

/// Sums a register value across all warp threads
template <typename T, int ReduceWidth = kWarpSize>
inline __device__ T warpReduceAllSum(T val) {
#pragma unroll
  for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
    val += shfl_xor(val, mask);
  }
  return val;
}

enum roundOption : int { NEAREST = 0, STOCHASTIC = 1 };

template <typename paramType, typename targetType, roundOption roundOpt>
class randFactor {
 public:
  curandStatePhilox4_32_10_t state;
  inline __device__ randFactor(ulong2 seed, int thread_id) {}
  inline __device__ targetType convertTypeFromParamToTarget(paramType param) {
    return param;
  }
  inline __device__ paramType convertTypeFromTargetToParam(targetType target) {
    return target;
  }
};

template <>
inline __device__ randFactor<at::Half, float, STOCHASTIC>::randFactor(
    ulong2 seed,
    int thread_id) {
  curand_init(seed.x, thread_id, seed.y, &state);
}

template <>
inline __device__ float
randFactor<at::Half, float, NEAREST>::convertTypeFromParamToTarget(
    at::Half param) {
  return __half2float(param);
}

template <>
inline __device__ float
randFactor<at::Half, float, STOCHASTIC>::convertTypeFromParamToTarget(
    at::Half param) {
  return __half2float(param);
}

template <>
inline __device__ at::Half
randFactor<at::Half, float, STOCHASTIC>::convertTypeFromTargetToParam(
    float target) {
  uint8_t rand = curand(&state) >> 24;
  unsigned w_int = __float_as_uint(target);
  unsigned assmebles = (w_int & 0xff800000) | (rand << 5);
  unsigned subtract = (w_int & 0xff800000);
  float assmeble_float = __uint_as_float(assmebles) - __uint_as_float(subtract);
  return __float2half_rz(target + assmeble_float);
}

template <>
inline __device__ at::Half
randFactor<at::Half, float, NEAREST>::convertTypeFromTargetToParam(
    float target) {
  return __float2half(target);
}

static inline __device__ void gpuAtomicAdd(float* address, float val) {
  gpu_atomic_add(address, val);
}

static inline __device__ void gpuAtomicAdd(c10::Half* address, c10::Half val) {
#if (                      \
    (defined(USE_ROCM)) || \
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

template <
    typename SIndex,
    typename TParam,
    typename T,
    bool ExactBlock = false,
    roundOption roundOpt = NEAREST>
#if defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void rowwise_sparse_adagrad_fused_length_sum_gradient_kernel(
    const int* __restrict__ prefix_sum_length_data, // prefix of lengths
                                                    // (offsets for the
                                                    // segments)
    int num_indices, // size of the indices array
    int block_size, // embedding dimension size
    int num_lengths, // number of segments
    const float epsilon,
    TParam* param,
    T* param_mom,
    const SIndex* indices,
    const T* __restrict__ grad,
    const float* lr,
    ulong2 seed,
    float weight_decay = 0.f) {
  const float LR = lr[0];
  // num_lengths blocks, each block process one segment
  int group = blockIdx.x; // the group-th segment
  int start = group == 0
      ? 0
      : prefix_sum_length_data[group - 1]; // start offset of the segment
  int end = prefix_sum_length_data[group]; // end offset of the segment
  CUDA_KERNEL_ASSERT(start <= num_indices);
  CUDA_KERNEL_ASSERT(end <= num_indices);

  class randFactor<TParam, T, roundOpt> rand_factor(
      seed,
      blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x);

  if (ExactBlock) {
    // Specialize WarpReduce for type float
    typedef cub::WarpReduce<float> WarpReduce;
    // Allocate WarpReduce shared memory for 32 warps, 1024 / 32 = 32
    __shared__ typename WarpReduce::TempStorage temp_storage[32];

    const size_t gradIdx = group * block_size + threadIdx.x; // index for grad
    for (int line = start + threadIdx.y; line < end; line += blockDim.y) {
      // line: the idx in the indices
      // threadIdx.x: index in the embedding dimension
      const SIndex index =
          indices[line]; // the index-th row in the embedding table
      float sum_squares = 0.0;
      __shared__ float row_sum_squares_avg;

      // block_size == blockDim.x
      const size_t paramIdx =
          index * block_size + threadIdx.x; // index for param
      const float x_ij = grad[gradIdx] +
          weight_decay * rand_factor.convertTypeFromParamToTarget(param[paramIdx]);
      sum_squares += x_ij * x_ij;

      // Return the warp-wide sums to each lane0 (threads 0, 32, 64, 96, ...)
      int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
      float reduce_result = WarpReduce(temp_storage[warp_id]).Sum(sum_squares);

      if ((threadIdx.y * blockDim.x + threadIdx.x) % 32 == 0) {
        row_sum_squares_avg = reduce_result / static_cast<float>(block_size);
        gpuAtomicAdd(&param_mom[index], static_cast<T>(row_sum_squares_avg));
      }
      __syncthreads();

      // update param
      float step = LR / (sqrtf(param_mom[index]) + epsilon);
      param[paramIdx] = rand_factor.convertTypeFromTargetToParam(
          rand_factor.convertTypeFromParamToTarget(param[paramIdx]) + x_ij * step);
    }
  } else {
    // TODO: Tuning NumThreads for sum_squares
    // TODO: Not compatible with embedding dim larger than maxThread
    typedef cub::BlockReduce<float, REDUCE_SIZE> BlockReduce;
    __shared__ BlockReduce::TempStorage temp_storage;
    int valid = min(block_size, blockDim.x);

    for (int line = start; line < end; ++line) {
      // line: the idx in the indices
      const SIndex index = indices[line]; // the index row in the embedding
      float sum_squares = 0.0;
      __shared__ float row_sum_squares_avg;

      for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        // i: index in the embedding dimension
        const float x_ij = grad[group * block_size + i] +
            weight_decay *
                rand_factor.convertTypeFromParamToTarget(
                    param[index * block_size + i]);
        sum_squares += x_ij * x_ij;
      }
      float reduce_result = BlockReduce(temp_storage).Sum(sum_squares, valid);

      if (threadIdx.x == 0) {
        row_sum_squares_avg = reduce_result / static_cast<float>(block_size);
        float mom_new = param_mom[index] + static_cast<T>(row_sum_squares_avg);
        param_mom[index] = mom_new;
      }
      __syncthreads();

      // update param
      float step = LR / (sqrtf(param_mom[index]) + epsilon);
      for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        size_t paramIdx = index * block_size + i; // index for param
        float x_ij = grad[group * block_size + i] +
            weight_decay *
                rand_factor.convertTypeFromParamToTarget(param[paramIdx]);
        float param_new =
            rand_factor.convertTypeFromParamToTarget(param[paramIdx]) + x_ij * step;
        param[paramIdx] = rand_factor.convertTypeFromTargetToParam(param_new);
      }
    }
  }
}

} // namespace caffe2
