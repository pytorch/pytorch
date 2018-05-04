#include "ATen/ATen.h"
#include "ATen/TensorUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include <THC/THCNumerics.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <THCUNN/THCHalfAutoNumerics.cuh>

#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDATensorMethods.cuh"
#include "ATen/cuda/CUDATypeConversion.cuh"


namespace at {
namespace native {

namespace {

template<typename T, typename AccumT>
struct LogSoftMaxForwardEpilogue {
  __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
//log is at::native::log, so we have to use THCNumerics here
    : logsum(max_input + THCNumerics<AccumT>::log(sum)) {}

  __device__ __forceinline__ T operator()(T input) const {
    return scalar_cast<T>(input - logsum); //input is cast to AccumT automatically in THCHalfAutoNumerics
}

  const AccumT logsum;
};

template<typename T, typename AccumT>
struct LogSoftMaxBackwardEpilogue {
  __device__ __forceinline__ LogSoftMaxBackwardEpilogue(AccumT sum)
    : sum(sum) {}

  __device__ __forceinline__ T operator()(T gradOutput, T output) const {
    return scalar_cast<T>(gradOutput - THCNumerics<AccumT>::exp(scalar_cast<AccumT>(output)) * sum);
  }

  const AccumT sum;
};

template<typename T, typename AccumT>
struct SoftMaxForwardEpilogue {
  __device__ __forceinline__ SoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
    : max_input(max_input)
    , sum(sum) {}

  __device__ __forceinline__ T operator()(T input) const {
    return scalar_cast<T>(THCNumerics<AccumT>::exp(input - max_input) / sum);
  }
  
  const AccumT max_input;
  const AccumT sum;
};

template<typename T, typename AccumT>
struct SoftMaxBackwardEpilogue {
  __device__ __forceinline__ SoftMaxBackwardEpilogue(AccumT sum)
    : sum(sum) {}

  // XXX: gradOutput that we get here is really gradOutput * output
  // Look for cmul in SoftMax_updateGradInput
  __device__ __forceinline__ T operator()(T gradOutput, T output) const {
    return scalar_cast<T>(gradOutput - output * sum);
  }

  const AccumT sum;
};




////////////////////////////////////////////////////////////////////////////////
// Spatial kernel (fast with large inner_size and small dim_size)
////////////////////////////////////////////////////////////////////////////////
// Let's assume that our input has been flattened to have only three dimension:
//     outer x dim x inner
// The spatial algorithm tries to paralellize along all of them.
// Within a 2d block threadIdx.y paralellizes over dim slices, and threads that
// share it will speed up reductions over dim (along axis x).
// The 2d grid is used to paralellize inner dimension over y axis and outer over x.
inline dim3 SpatialSoftMax_getGridSize(
    dim3 block, uint32_t max_active_blocks,
    uint64_t outer_size, uint64_t dim_size, uint64_t inner_size) {
  // First, tile as many blocks as we can over the y axis
  uint32_t inner_blocks = (inner_size + block.y - 1) / block.y;
  if (inner_blocks > max_active_blocks)
    inner_blocks = max_active_blocks;
  // Fill the x axis with as many blocks as we can fit (a little more is ok too)
  uint32_t outer_blocks = (max_active_blocks + inner_blocks - 1) / inner_blocks;
  if (outer_blocks > outer_size)
    outer_blocks = outer_size;
  return dim3(outer_blocks, inner_blocks);
}

const int max_threads = 1024;

inline dim3 SpatialSoftMax_getBlockSize(
  uint64_t outer_size, uint64_t dim_size, uint64_t inner_size) {
  uint32_t inner_threads = inner_size;
  inner_threads = std::min(inner_threads, static_cast<uint32_t>(max_threads));
  uint32_t dim_threads = 1;
  if (inner_threads <= 64 && dim_size >= 64) {
    while (inner_threads * dim_threads <= max_threads && dim_threads <= dim_size)
      dim_threads *= 2;
    dim_threads /= 2;
  }
  return dim3(dim_threads, inner_threads);
}


template<typename accscalar_t, typename Kernel>
void SpatialSoftMax_getLaunchSizes(
    Kernel k,
    uint64_t outer_size, uint64_t dim_size, uint64_t inner_size,
    dim3& grid, dim3& block, uint32_t& smem_size) {
  block = SpatialSoftMax_getBlockSize(outer_size, dim_size, inner_size);
  uint32_t block_threads = block.x * block.y;
  smem_size = block.x == 1 ? 0 : block_threads * sizeof(accscalar_t);
  int max_active_blocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                                                k, block_threads, smem_size);
  max_active_blocks *= at::globalContext().getCurrentDeviceProperties()->multiProcessorCount;
  grid = SpatialSoftMax_getGridSize(block, max_active_blocks, outer_size, dim_size, inner_size);
}

inline dim3 SoftMax_getBlockSize(int ILP, uint64_t dim_size) {
  uint64_t block_size = 1;
  uint64_t max_block_size = std::min(dim_size / ILP, static_cast<uint64_t>(max_threads));
  while (block_size < max_block_size) block_size *= 2;
  // Launch at least a single warp - the kernel assumes that.
  block_size = std::max(block_size, static_cast<uint64_t>(32));
  return dim3(block_size);
}

template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

// Note that it's not a complete block-wide reduction.
// Only threads that share threadIdx.y reduce values.
template<typename T, template<typename> class ReduceOp>
__forceinline__ __device__
T spatialBlockReduceX(T *shared, T val) {
  ReduceOp<T> r;
  shared += threadIdx.y * blockDim.x;

  __syncthreads();

  shared[threadIdx.x] = val;

  // NOTE: loop starts with __syncthreads()
  int offset = blockDim.x / 2;
  while (offset > 0) {
    __syncthreads();
    if (threadIdx.x < offset)
      shared[threadIdx.x] = r(shared[threadIdx.x], shared[threadIdx.x + offset]);
    offset /= 2;
  }

  __syncthreads();

  return shared[0];
}

template <typename scalar_t, typename accscalar_t, template<typename, typename> class Epilogue>
__global__ void cunn_SpatialSoftMaxForward(
    scalar_t *output, scalar_t *input,
    uint32_t outer_size, uint32_t dim_size, uint32_t inner_size)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  const uint32_t outer_stride = inner_size * dim_size;
  const uint32_t dim_stride = inner_size;

  for (uint32_t outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x) {
    const uint32_t outer_offset = outer_index * outer_stride;
    for (uint32_t inner_index = blockIdx.y * blockDim.y + threadIdx.y; inner_index < inner_size; inner_index += blockDim.y * gridDim.y) {
      const uint32_t data_offset = outer_offset + inner_index;
      ////////////////////////////////////////////////////////////
      // These two blocks are really eqivalent, but specializing on
      // blockDim.x == 1 makes the kernel faster when it's unused.
      // I didn't want to thread an extra template parameter, and nvcc
      // seems to be smart enough to hoist the if outside of the loops.
      ////////////////////////////////////////////////////////////
 
      if (blockDim.x > 1) {
        accscalar_t max_input = THCNumerics<accscalar_t>::min();
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
          const accscalar_t value = scalar_cast<accscalar_t>(input[data_offset + d * dim_stride]);
          max_input = Max<accscalar_t>()(max_input, value);
        }
        max_input = spatialBlockReduceX<accscalar_t, Max>(sdata,max_input);

        accscalar_t sum = 0;
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          sum += THCNumerics<accscalar_t>::exp(scalar_cast<accscalar_t>(input[data_offset + d * dim_stride]) 
                 - max_input);
        sum = spatialBlockReduceX<accscalar_t, Add>(sdata, sum);

        Epilogue<scalar_t, accscalar_t> epilogue(max_input, sum);
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          output[data_offset + d * dim_stride] = epilogue(input[data_offset + d * dim_stride]);
      } else {
        accscalar_t max_input = THCNumerics<accscalar_t>::min();
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
          const accscalar_t value = scalar_cast<accscalar_t>(input[data_offset + d * dim_stride]);
          max_input = Max<accscalar_t>()(max_input, value);
        }
        accscalar_t sum = 0;
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          sum += THCNumerics<accscalar_t>::exp(scalar_cast<accscalar_t>(input[data_offset + d * dim_stride]) 
                 - max_input);
        Epilogue<scalar_t, accscalar_t> epilogue(max_input, sum);
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          output[data_offset + d * dim_stride] = epilogue(input[data_offset + d * dim_stride]);
      }
    }
  }
}



template <typename scalar_t, typename accscalar_t, template<typename, typename> class Epilogue>
__global__ void cunn_SpatialSoftMaxBackward(
    scalar_t *gradInput, scalar_t *output, scalar_t *gradOutput,
    uint32_t outer_size, uint32_t dim_size, uint32_t inner_size)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  const uint32_t outer_stride = inner_size * dim_size;
  const uint32_t dim_stride = inner_size;

  for (uint32_t outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x) {
    const uint32_t outer_offset = outer_index * outer_stride;
    for (uint32_t inner_index = blockIdx.y * blockDim.y + threadIdx.y; inner_index < inner_size; inner_index += blockDim.y * gridDim.y) {
      const uint32_t data_offset = outer_offset + inner_index;
      // See the comment in forward kernel
      if (blockDim.x > 1) {
        accscalar_t sum = 0;
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          sum += gradOutput[data_offset + d * dim_stride];
        sum = spatialBlockReduceX<accscalar_t, Add>(sdata, sum);

        Epilogue<scalar_t, accscalar_t> epilogue(sum);
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
          gradInput[data_offset + d * dim_stride] =
            epilogue(gradOutput[data_offset + d * dim_stride],
                    output[data_offset + d * dim_stride]);
        }
      } else {
        accscalar_t sum = 0;
        for (uint32_t d = 0; d < dim_size; d++)
          sum += gradOutput[data_offset + d * dim_stride];

        Epilogue<scalar_t, accscalar_t> epilogue(sum);
        for (uint32_t d = 0; d < dim_size; d++) {
          gradInput[data_offset + d * dim_stride] =
            epilogue(gradOutput[data_offset + d * dim_stride],
                    output[data_offset + d * dim_stride]);
        }
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
// Regular kernel (fast when dim_size is large; requires inner_size == 1)
////////////////////////////////////////////////////////////////////////////////


template <typename T, typename AccumT>
struct MaxFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
    return fmaxType(max, v);
  }
};

template<typename T, typename AccumT>
struct AddFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + v;
  }
};

template<typename T, typename AccumT>
struct SumExpFloat
{
  __device__ __forceinline__ SumExpFloat(AccumT v)
    : max_k(v) {}

  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + THCNumerics<AccumT>::exp(v - max_k);
  }

  const AccumT max_k;
};

template <template<typename> class Reduction, typename AccumT>
__device__ __forceinline__ AccumT
blockReduce(AccumT* smem, AccumT val,
            const Reduction<AccumT>& r,
            AccumT defaultVal)
{
  // To avoid RaW races from chaining blockReduce calls together, we need a sync here
  __syncthreads();

  smem[threadIdx.x] = val;

  __syncthreads();

  AccumT warpVal = defaultVal;

  // First warp will perform per-warp reductions for the remaining warps
  if (threadIdx.x < 32) {
    int lane = threadIdx.x % 32;
    if (lane < blockDim.x / 32) {
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        warpVal = r(warpVal, smem[lane * 32 + i]);
      }
      smem[lane] = warpVal;
    }
  }

  __syncthreads();

  // First thread will perform a reduction of the above per-warp reductions
  AccumT blockVal = defaultVal;

  if (threadIdx.x == 0) {
    for (int i = 0; i < blockDim.x / 32; ++i) {
      blockVal = r(blockVal, smem[i]);
    }
    smem[0] = blockVal;
  }

  // Sync and broadcast
  __syncthreads();
  return smem[0];
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
  for (; offset < size - last; offset += blockDim.x * ILP) {
    T tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      tmp[j] = data[offset + j * blockDim.x];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      threadVal = r(threadVal, tmp[j]);
  }

  // Epilogue
  for (; offset < size; offset += blockDim.x)
    threadVal = r(threadVal, data[offset]);

  return threadVal;
}

template <int ILP, typename scalar_t, typename accscalar_t, template <typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxForward(scalar_t *output, scalar_t *input, int classes)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * classes;
  output += blockIdx.x * classes;

  // find the max
  accscalar_t threadMax = ilpReduce<MaxFloat, ILP, scalar_t, accscalar_t>(
      input, classes, MaxFloat<scalar_t, accscalar_t>(), -THCNumerics<accscalar_t>::max());
  accscalar_t max_k = blockReduce<Max, accscalar_t>(
      sdata, threadMax, Max<accscalar_t>(), -THCNumerics<accscalar_t>::max());

  // reduce all values
  accscalar_t threadExp = ilpReduce<SumExpFloat, ILP, scalar_t, accscalar_t>(
      input, classes, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
  accscalar_t sumAll = blockReduce<Add, accscalar_t>(
      sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));

  Epilogue<scalar_t, accscalar_t> epilogue(max_k, sumAll);
  int offset = threadIdx.x;
  int last = classes % (ILP * blockDim.x);
  for (; offset < classes - last; offset += blockDim.x * ILP) {
    scalar_t tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      tmp[j] = input[offset + j * blockDim.x];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      output[offset + j * blockDim.x] = epilogue(tmp[j]);
  }

  for (; offset < classes; offset += blockDim.x)
    output[offset] = epilogue(input[offset]);
}

template <int ILP, typename scalar_t, typename accscalar_t, template<typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxBackward(scalar_t *gradInput, scalar_t *output, scalar_t *gradOutput, int classes)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  gradInput += blockIdx.x * classes;
  output += blockIdx.x * classes;
  gradOutput += blockIdx.x * classes;

  accscalar_t threadSum = ilpReduce<AddFloat, 4, scalar_t, accscalar_t>(
      gradOutput, classes, AddFloat<scalar_t, accscalar_t>(), accscalar_t(0));
  accscalar_t sum_k = blockReduce<Add, accscalar_t>(
        sdata, threadSum, Add<accscalar_t>(), accscalar_t(0));

  Epilogue<scalar_t, accscalar_t> epilogue(sum_k);
  int offset = threadIdx.x;
  int last = classes % (ILP * blockDim.x);
  for (; offset < classes - last; offset += blockDim.x * ILP) {
    scalar_t tmpGradOutput[ILP];
    scalar_t tmpOutput[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      tmpGradOutput[j] = gradOutput[offset + j * blockDim.x];
      tmpOutput[j] = output[offset + j * blockDim.x];
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      gradInput[offset + j * blockDim.x] = epilogue(tmpGradOutput[j], tmpOutput[j]);
  }

  for (; offset < classes; offset += blockDim.x)
    gradInput[offset] = epilogue(gradOutput[offset], output[offset]);
}






template<template<typename, typename> class Epilogue>
Tensor host_softmax(const Tensor & input_, const int64_t dim_){
  auto input = input_.contiguous();
  Tensor output = at::native::empty_like(input);
  if (input.dim() == 0) input = input.view(1);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  AT_CHECK(dim >=0 && dim < input.dim(), "dim must be non-negative and less than input dimensions");
  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);
  int64_t inner_size = 1;
  cudaStream_t stream = globalContext().getCurrentCUDAStream();
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= input.size(i);
  for (int64_t i = dim + 1; i < input.dim(); ++i)
    inner_size *= input.size(i);
  // This kernel spawns a block per each element in the batch.
  // XXX: it assumes that inner_size == 1
  if (inner_size == 1) {
    const int ILP = 2;
    dim3 grid(outer_size);
    dim3 block = SoftMax_getBlockSize(ILP, dim_size);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "host_softmax", [&] {
    using cuda_scalar_t = cuda::type<scalar_t>;
    using accscalar_t = acc_type<cuda_scalar_t, true>;
    cunn_SoftMaxForward<ILP, cuda_scalar_t, accscalar_t, Epilogue>
      <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
        output.data<cuda_scalar_t>(), input.data<cuda_scalar_t>(), dim_size
    );
    });
  // This kernel runs in a 2D grid, where each application along y dimension has a fixed
  // outer_size, and runs in parallel over inner_size. Dimension x is parallel over outer_size.
  // Reductions over dim are done in a single-threaded manner.
  } else {
    uint32_t smem_size;
    dim3 grid, block;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "host_softmax", [&] {
    using cuda_scalar_t = cuda::type<scalar_t>;
    using accscalar_t = acc_type<cuda_scalar_t, true>;
    SpatialSoftMax_getLaunchSizes<accscalar_t>(
        &cunn_SpatialSoftMaxForward<cuda_scalar_t, accscalar_t, Epilogue>,
        outer_size, dim_size, inner_size,
        grid, block, smem_size);
    cunn_SpatialSoftMaxForward<cuda_scalar_t, accscalar_t, Epilogue>
      <<<grid, block, smem_size, stream>>>(
        output.data<cuda_scalar_t>(), input.data<cuda_scalar_t>(), outer_size, dim_size, inner_size
    );
    });
  }
  THCudaCheck(cudaGetLastError());
  return output;
}

template<template<typename, typename> class Epilogue>
Tensor host_softmax_backward(const Tensor &grad_, const Tensor &output_, int64_t dim_){
  int64_t dim = maybe_wrap_dim(dim_, grad_.dim());
  auto grad = grad_.contiguous();
  Tensor gI = at::native::empty_like(grad);
  if (grad.dim() == 0) grad = grad.view(1);
  AT_CHECK(dim >=0 && dim < grad.dim(), "dim must be non-negative and less than input dimensions");
  auto output = output_.contiguous();
  if (output.dim() == 0) output = output.view(1);
  int64_t outer_size = 1;
  int64_t dim_size = output.size(dim);
  int64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= output.size(i);
  for (int64_t i = dim + 1; i < output.dim(); ++i)
    inner_size *= output.size(i);
// See descriptions of kernels above.
  cudaStream_t stream = globalContext().getCurrentCUDAStream();
  if (inner_size == 1) {
    const int ILP = 2;
    dim3 grid(outer_size);
    dim3 block = SoftMax_getBlockSize(ILP, dim_size);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "host_softmax_backward", [&] {
    using cuda_scalar_t = cuda::type<scalar_t>;
    using accscalar_t = acc_type<cuda_scalar_t, true>;
    cunn_SoftMaxBackward<ILP, cuda_scalar_t, accscalar_t, Epilogue>
      <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
        gI.data<cuda_scalar_t>(), output.data<cuda_scalar_t>(), grad.data<cuda_scalar_t>(), dim_size
    );
    });
  } else {
    uint32_t smem_size;
    dim3 grid, block;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "host_softmax_backward", [&] {
    using cuda_scalar_t = cuda::type<scalar_t>;
    using accscalar_t = acc_type<cuda_scalar_t, true>;
    SpatialSoftMax_getLaunchSizes<accscalar_t>(
        &cunn_SpatialSoftMaxBackward<cuda_scalar_t, accscalar_t, Epilogue>,
        outer_size, dim_size, inner_size,
        grid, block, smem_size);

    cunn_SpatialSoftMaxBackward<cuda_scalar_t, accscalar_t, Epilogue>
      <<<grid, block, smem_size, stream>>>(
        gI.data<cuda_scalar_t>(), output.data<cuda_scalar_t>(), grad.data<cuda_scalar_t>(), 
        outer_size, dim_size, inner_size
    );
    });
  }
  THCudaCheck(cudaGetLastError());
  return gI;
}
}

Tensor log_softmax_cuda(const Tensor &input, const int64_t dim){
  return host_softmax<LogSoftMaxForwardEpilogue>(input, dim);
}

Tensor log_softmax_backward_cuda(const Tensor &grad, const Tensor &output, int64_t dim, const Tensor &input){
  return host_softmax_backward<LogSoftMaxBackwardEpilogue>(grad, output, dim);
}

Tensor softmax_cuda(const Tensor &input, const int64_t dim){
  return host_softmax<SoftMaxForwardEpilogue>(input, dim);
}

Tensor softmax_backward_cuda(const Tensor &grad, const Tensor &output, int64_t dim, const Tensor &input){
  
  Tensor tmp = grad * output;
  return host_softmax_backward<SoftMaxBackwardEpilogue>(tmp, output, dim);
}

}
}

