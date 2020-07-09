#pragma once

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <curand_kernel.h>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"

#ifdef __HIP_PLATFORM_HCC__
#define SEGREDUCE_MINBLOCKS 8
#else
#define SEGREDUCE_MINBLOCKS 16
#endif


namespace caffe2 {

enum roundOption : int { NEAREST = 0, STOCHASTIC = 1 };

template <typename srcType, typename dstType, roundOption roundOpt>
class randFactor {
 public:
  curandStatePhilox4_32_10_t state;
  inline __device__ randFactor(ulong2 seed, int thread_id) {}
  inline __device__ dstType convertTypeFromSrcToDest(srcType param) {
    return param;
  }
  inline __device__ srcType convertTypeFromDestToSrc(dstType param) {
    return param;
  }
};

template <>
inline __device__ randFactor<float, at::Half, STOCHASTIC>::randFactor(
    ulong2 seed,
    int thread_id) {
  curand_init(seed.x, thread_id, seed.y, &state);
}

template <>
inline __device__ at::Half
randFactor<float, at::Half, NEAREST>::convertTypeFromSrcToDest(float param) {
  return __float2half(param);
}

template <>
inline __device__ at::Half
randFactor<float, at::Half, STOCHASTIC>::convertTypeFromSrcToDest(float param) {
  uint8_t rand = curand(&state) >> 24;
  unsigned w_int = __float_as_uint(param);
  unsigned assmebles = (w_int & 0xff800000) | (rand << 5);
  unsigned subtract = (w_int & 0xff800000);
  float assmeble_float = __uint_as_float(assmebles) - __uint_as_float(subtract);
  return __float2half_rz(param + assmeble_float);
}

template <>
inline __device__ float
randFactor<float, at::Half, STOCHASTIC>::convertTypeFromDestToSrc(at::Half param) {
  return __half2float(param);
}

template <>
inline __device__ float
randFactor<float, at::Half, NEAREST>::convertTypeFromDestToSrc(at::Half param) {
  return __half2float(param);
}

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

template <typename SIndex, typename TParam, typename T, bool ExactBlock = false, roundOption roundOpt = NEAREST>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void rowwise_sparse_adagrad_fused_length_sum_gradient_kernel(
    const int* __restrict__ prefix_sum_length_data, // prefix of lengths
                                                    // (offsets for the
                                                    // segments)
    int N, // number of rows (hash size) of embedding table
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
  CUDA_KERNEL_ASSERT(start <= N);
  CUDA_KERNEL_ASSERT(end <= N);

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
          weight_decay * rand_factor.convertTypeFromSrcToDest(param[paramIdx]);
      sum_squares += x_ij * x_ij;

      // Return the warp-wide sums to each lane0 (threads 0, 32, 64, 96, ...)
      int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
      float reduce_result = WarpReduce(temp_storage[warp_id]).Sum(sum_squares);

      if ((threadIdx.y * blockDim.x + threadIdx.x) % 32 == 0) {
        row_sum_squares_avg = reduce_result / static_cast<float>(block_size);
        // AtomicAdd when the embedding dim is larger than 32.
        // param_mom[index] += row_sum_squares_avg;
        gpuAtomicAdd(&param_mom[index], static_cast<T>(row_sum_squares_avg));
      }
      __syncthreads();

      // update param
      float step = LR / (sqrtf(param_mom[index]) + epsilon);
      param[paramIdx] = rand_factor.convertTypeFromDestToSrc(
          rand_factor.convertTypeFromSrcToDest(param[paramIdx]) + x_ij * step);
    }
  } else {
    // TODO: Tuning NumThreads for sum_squares
    typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
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
                rand_factor.convertTypeFromSrcToDest(param[index * block_size + i]);
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
                rand_factor.convertTypeFromSrcToDest(param[paramIdx]);
        float param_new =
            rand_factor.convertTypeFromSrcToDest(param[paramIdx]) + x_ij * step;
        // float param_new1 = param[paramIdx];
        // printf("step %f, x_ij %f", step, x_ij);
        param[paramIdx] = rand_factor.convertTypeFromDestToSrc(param_new);
      }
    }
  }
}

} // namespace caffe2
