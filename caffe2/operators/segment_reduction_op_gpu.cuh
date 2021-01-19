#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include "caffe2/core/context_gpu.h"


#ifdef __HIP_PLATFORM_HCC__
#define SEGREDUCE_MINBLOCKS 8
#else
#define SEGREDUCE_MINBLOCKS 16
#endif

namespace caffe2{

template <typename T>
struct SharedMemory;

template <>
struct SharedMemory<double> {
  __device__ double* getPointer() {
    extern __shared__ double s_double[];
    return s_double;
  }
};

template <>
struct SharedMemory<float> {
  __device__ float* getPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

template <>
struct SharedMemory<at::Half> {
  __device__ at::Half* getPointer() {
    extern __shared__ at::Half s_half[];
    return s_half;
  }
};


template <typename InType, typename OutType>
__device__ inline OutType convert_type(const InType in) {
  return in;
}

template <>
__device__ inline float convert_type<at::Half, float>(const at::Half in) {
  return __half2float(in);
}

template <
    typename InType,
    typename OutType,
    typename IndexType,
    bool ExactBlock = false,
    bool Average = false>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(1024,SEGREDUCE_MINBLOCKS)
#endif
__global__ void sparse_length_sum_kernel(
    const InType* __restrict__ in,
    OutType* __restrict__ out,
    const int* __restrict__ prefix_sum_length_data,
    const IndexType* __restrict__ indices,
    int N,
    int post,
    int len_length,
    int len_indices) {
  // len_length blocks
  int group = blockIdx.x;

  int start = group == 0 ? 0 : prefix_sum_length_data[group - 1];
  int end = prefix_sum_length_data[group];
  CUDA_KERNEL_ASSERT(start <= len_indices);
  CUDA_KERNEL_ASSERT(end <= len_indices);

  struct SharedMemory<OutType> smem;
  OutType* reduceVals = smem.getPointer();

  if (ExactBlock) {
    OutType sum = (OutType)0;

    in += threadIdx.x;
    for (int line = start + threadIdx.y; line < end; line += blockDim.y) {
      sum += convert_type<InType, OutType>(in[indices[line] * post]);
    }

    reduceVals[threadIdx.y * blockDim.x + threadIdx.x] = sum;
    __syncthreads();

    if (threadIdx.y == 0) {
      sum = (OutType)0;
      for (int i = 0; i < blockDim.y; ++i) {
        sum += reduceVals[i * blockDim.x + threadIdx.x];
      }
      if (Average && (end - start) > 1) {
        sum /= (end - start);
      }

      out[group * post + threadIdx.x] = sum;
    }
  } else {
    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      OutType sum = (OutType)0;
      for (int line = start; line < end; ++line) {
        sum += convert_type<InType, OutType>(in[indices[line] * post + i]);
      }
      if (Average && (end - start) > 1) {
        sum /= (end - start);
      }
      out[group * post + i] = sum;
    }
  }
}

} //namespace caffe2
