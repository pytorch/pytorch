#include "THCUNN.h"
#include "THCHalf.h"
#include "THCTensorTypeUtils.cuh"
#include "THCHalfAutoNumerics.cuh"
#include "SharedMem.cuh"

template <typename T, typename AccumT>
__global__ void cunn_SpatialLogSoftMax_updateOutput_kernel(T *output, T *input, uint32_t outer_size, uint32_t dim_size, uint32_t inner_size)
{
  const uint32_t outer_stride = inner_size * dim_size;
  const uint32_t dim_stride = inner_size;

  for (uint32_t outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x) {
    const uint32_t outer_offset = outer_index * outer_stride;
    for (uint32_t inner_index = blockIdx.y * blockDim.x + threadIdx.x; inner_index < inner_size; inner_index += blockDim.x * gridDim.y) {
      const uint32_t data_offset = outer_offset + inner_index;

      T max_input = input[data_offset];
      for (uint32_t d = 1; d < dim_size; d++) {
        const T value = input[data_offset + d * dim_stride];
        max_input = THCNumerics<T>::ge(max_input, value) ? max_input : value;
      }

      AccumT sum = 0;
      for (uint32_t d = 0; d < dim_size; d++)
        sum += THCNumerics<T>::exp(input[data_offset + d * dim_stride] - max_input);
      const T logsum = max_input + ScalarConvert<AccumT, T>::to(THCNumerics<AccumT>::log(sum));

      for (uint32_t d = 0; d < dim_size; d++)
        output[data_offset + d * dim_stride] = input[data_offset + d * dim_stride] - logsum;
    }
  }
}

template <typename T, typename AccumT>
__global__ void cunn_SpatialLogSoftMax_updateGradInput_kernel(T *gradInput, T *output, T *gradOutput, uint32_t outer_size, uint32_t dim_size, uint32_t inner_size)
{
  const uint32_t outer_stride = inner_size * dim_size;
  const uint32_t dim_stride = inner_size;

  for (uint32_t outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x) {
    const uint32_t outer_offset = outer_index * outer_stride;
    for (uint32_t inner_index = blockIdx.y * blockDim.x + threadIdx.x; inner_index < inner_size; inner_index += blockDim.x * gridDim.y) {
      const uint32_t data_offset = outer_offset + inner_index;

      AccumT sum = 0;
      for (uint32_t d = 0; d < dim_size; d++) {
        sum += gradOutput[data_offset + d * dim_stride];
      }
      const T real_sum = ScalarConvert<AccumT, T>::to(sum);

      for (uint32_t d = 0; d < dim_size; d++) {
        gradInput[data_offset + d * dim_stride] = gradOutput[data_offset + d * dim_stride] -
          THCNumerics<T>::exp(output[data_offset + d * dim_stride]) * real_sum;
      }
    }
  }
}

static void LogSoftMax_getSpatialGridSize(
    uint32_t block_size, uint32_t max_active_blocks,
    uint64_t outer_size, uint64_t dim_size, uint64_t inner_size,
    dim3& grid, dim3& block) {
  // First, tile as many blocks as we can over the y axis
  uint32_t y_size = (inner_size + block_size - 1) / block_size;
  if (y_size > max_active_blocks)
    y_size = max_active_blocks;
  // Fill the x axis with as many blocks as we can fit
  uint32_t x_size = (max_active_blocks + y_size - 1) / y_size;
  if (x_size > outer_size)
    x_size = outer_size;
  grid = dim3(x_size, y_size);
  block = dim3(block_size);
}

template <typename T, typename AccumT>
struct MaxFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT max, T v) const
  {
    return fmaxType(max, v);
  }
};

template<typename T, typename AccumT>
struct SumFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const
  {
    return sum + v;
  }
};

template<typename T, typename AccumT>
struct SumExpFloat
{
  __device__ __forceinline__ SumExpFloat(T v)
    : max_k(v)
  {}

  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const
  {
    return sum + THCNumerics<T>::exp(v - max_k);
  }

  const T max_k;
};

template<typename AccumT>
struct NoFinal
{
  __device__ __forceinline__ AccumT operator()(AccumT v) const
  {
    return v;
  }
};

template<typename AccumT>
struct LSMFinal
{
  __device__ __forceinline__ LSMFinal(AccumT m)
    : max_k(m)
  {}

  __device__ __forceinline__ AccumT operator()(AccumT v) const
  {
    return max_k + THCNumerics<AccumT>::log(v);
  }

  const AccumT max_k;
};

template <template<typename, typename> class Reduction, template<typename> class Finalize, typename AccumT>
__device__ __forceinline__ AccumT
blockReduce(AccumT* smem, AccumT val,
            const Reduction<AccumT, AccumT>& r,
            AccumT defaultVal,
            const Finalize<AccumT>& f)
{
  // To avoid RaW races from chaining blockReduce calls together, we
  // need a sync here
  __syncthreads();

  smem[threadIdx.x] = val;

  __syncthreads();

  AccumT warpVal = defaultVal;

  // First warp will perform per-warp reductions for the remaining warps
  if ((threadIdx.x / 32) == 0) // only threads in warp1 go into this (if)
  {
    int lane = threadIdx.x % 32; // from 0 to 31

    // if less than 1024 threads per block, then only activate the relevant lanes
    if (lane < blockDim.x / 32)
    {
#pragma unroll
      for (int i = 0; i < 32; ++i)
      {
        warpVal = r(warpVal, smem[lane * 32 + i]);
      }

      smem[lane] = warpVal;
    }
  }

  __syncthreads();

  // First thread will perform a reduction of the above per-warp reductions
  AccumT blockVal = defaultVal;

  if (threadIdx.x == 0)
  {
    for (int i = 0; i < blockDim.x / 32; ++i)
    {
      blockVal = r(blockVal, smem[i]);
    }

    smem[0] = f(blockVal);
  }

  // Sync and broadcast
  __syncthreads();
  return smem[0];
}

template <template<typename, typename> class Reduction, typename AccumT>
__device__ __forceinline__ AccumT
blockReduce(AccumT* smem, AccumT val,
            const Reduction<AccumT, AccumT>& r,
            AccumT defaultVal)
{
  return blockReduce<Reduction, NoFinal, AccumT>(smem, val, r, defaultVal, NoFinal<AccumT>());
}

template <template<typename, typename> class Reduction, int ILP, typename T, typename AccumT>
__device__ __forceinline__ AccumT
ilpReduce(T* data,
          int size,
          const Reduction<T, AccumT>& r,
          AccumT defaultVal)
{
  AccumT threadVal = defaultVal;
  int offset = threadIdx.x;

  int last = size % (ILP * blockDim.x);

  // Body (unroll by ILP times)
  for (; offset < size - last; offset += blockDim.x * ILP)
  {
    T tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      tmp[j] = data[offset + j * blockDim.x];
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      threadVal = r(threadVal, tmp[j]);
    }
  }

  // Epilogue
  for (; offset < size; offset += blockDim.x)
  {
    threadVal = r(threadVal, data[offset]);
  }

  return threadVal;
}

template <int ILP, typename T, typename AccumT>
__global__ void
cunn_LogSoftMax_updateOutput_kernel(T *output, T *input, int classes)
{
  SharedMem<AccumT> smem;
  AccumT *buffer = smem.getPointer();
  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * classes;
  output += blockIdx.x * classes;

  // find the max of the batch
  AccumT threadMax = ilpReduce<MaxFloat, ILP, T, AccumT>(
      input, classes, MaxFloat<T, AccumT>(), -THCNumerics<AccumT>::max());
  // find the max over all batches
  AccumT max_k = blockReduce<MaxFloat, AccumT>(
      buffer, threadMax, MaxFloat<AccumT, AccumT>(), -THCNumerics<AccumT>::max());
  T max_k_non_accum = ScalarConvert<AccumT, T>::to(max_k);

  AccumT threadExp = ilpReduce<SumExpFloat, ILP, T, AccumT>(
      input, classes, SumExpFloat<T, AccumT>(max_k_non_accum), AccumT(0));
  T logsum_k = ScalarConvert<AccumT, T>::to(
      blockReduce<SumFloat, LSMFinal, AccumT>(
          buffer, threadExp, SumFloat<AccumT, AccumT>(), AccumT(0), LSMFinal<AccumT>(max_k)));

  // Output LSM (hand ILP)
  int offset = threadIdx.x;

  int last = classes % (ILP * blockDim.x);
  for (; offset < classes - last; offset += blockDim.x * ILP)
  {
    T tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      tmp[j] = input[offset + j * blockDim.x];
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      output[offset + j * blockDim.x] = tmp[j] - logsum_k;
    }
  }

  for (; offset < classes; offset += blockDim.x)
  {
    output[offset] = input[offset] - logsum_k;
  }
}

template <int ILP, typename T, typename AccumT>
__global__ void
cunn_LogSoftMax_updateGradInput_kernel(T *gradInput,
                                       T *output,
                                       T *gradOutput,
                                       int classes)
{
  SharedMem<AccumT> smem;
  AccumT *buffer = smem.getPointer();
  gradInput += blockIdx.x * classes;
  output += blockIdx.x * classes;
  gradOutput += blockIdx.x * classes;

  AccumT threadSum = ilpReduce<SumFloat, 4, T, AccumT>(
      gradOutput, classes, SumFloat<T, AccumT>(), AccumT(0));
  T sum_k = ScalarConvert<AccumT, T>::to(
      blockReduce<SumFloat, AccumT>(
          buffer, threadSum, SumFloat<AccumT, AccumT>(), AccumT(0)));

  // Update gradInput (hand ILP)
  int offset = threadIdx.x;
  int last = classes % (ILP * blockDim.x);
  for (; offset < classes - last; offset += blockDim.x * ILP)
  {
    T tmpGradOutput[ILP];
    T tmpOutput[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      tmpGradOutput[j] = gradOutput[offset + j * blockDim.x];
      tmpOutput[j] = output[offset + j * blockDim.x];
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j)
    {
      gradInput[offset + j * blockDim.x] =
        tmpGradOutput[j] - THCNumerics<T>::exp(tmpOutput[j]) * sum_k;
    }
  }

  for (; offset < classes; offset += blockDim.x)
  {
    gradInput[offset] =
      gradOutput[offset] - THCNumerics<T>::exp(output[offset]) * sum_k;
  }
}

#include "generic/LogSoftMax.cu"
#include "THCGenerateFloatTypes.h"
