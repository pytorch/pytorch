#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "SharedMem.cuh"

template <typename T, typename AccumT>
__global__ void cunn_SpatialLogSoftMax_updateOutput_kernel(T *output, T *input, int classSize, int height, int width)
{
  int batchIndex = blockIdx.x;
  int index = threadIdx.x;

  while (index < height*width) {
    int y = index / width;
    int x = index % width;
    if (y >= height)
      break;

    // calculate input starting index in cuda layout (B x H x W x C)
    int inputStartIndex =
      (height*width*classSize)*batchIndex +
      (width*classSize)*y +
      (classSize)*x;

    T maxInput = input[inputStartIndex];
    for (int i = 1; i < classSize; i++) {
      T value = input[inputStartIndex + i];
      maxInput = THCNumerics<T>::ge(maxInput, value) ? maxInput : value;
    }

    AccumT sum = 0;
    for (int i = 0; i < classSize; i++) {
      sum += THCNumerics<T>::exp(input[inputStartIndex + i] - maxInput);
    }
    T logsum = maxInput + ScalarConvert<AccumT, T>::to(THCNumerics<AccumT>::log(sum));

    for (int i = 0; i < classSize; i++) {
      // calculate output index in torch layout (B x C x H x W)
      int outputIndex =
        (classSize*height*width)*batchIndex +
        (height*width)*i +
        (width)*y +
        x;
      output[outputIndex] = input[inputStartIndex + i] - logsum;
    }
    index += blockDim.x;
  }
}

template <typename T, typename AccumT>
__global__ void cunn_SpatialLogSoftMax_updateGradInput_kernel(T *gradInput, T *output, T *gradOutput, int classSize, int height, int width)
{
  int batchIndex = blockIdx.x;
  int index = threadIdx.x;

  while (index < height*width) {
    int y = index / width;
    int x = index % width;
    if (y >= height)
      break;

    // calculate output starting index in cuda layout (B x H x W x C)
    int outputStartIndex =
      (height*width*classSize)*batchIndex +
      (width*classSize)*y +
      (classSize)*x;

    AccumT sum = 0;
    for (int i = 0; i < classSize; i++) {
      sum += gradOutput[outputStartIndex + i];
    }

    for (int i = 0; i < classSize; i++) {
      // calculate input index in torch layout (B x C x H x W)
      int inputIndex =
        (classSize*height*width)*batchIndex +
        (height*width)*i +
        (width)*y +
        x;
      gradInput[inputIndex] = ScalarConvert<AccumT, T>::to(
        gradOutput[outputStartIndex + i] - THCNumerics<T>::exp(output[outputStartIndex + i]) * sum);
    }
    index += blockDim.x;
  }
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
