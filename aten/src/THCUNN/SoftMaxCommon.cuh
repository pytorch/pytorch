#include "THCUNN.h"
#include "THCHalf.h"
#include "THCTensorTypeUtils.cuh"
#include "THCHalfAutoNumerics.cuh"
#include "SharedMem.cuh"
#include <algorithm>
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

inline dim3 SpatialSoftMax_getBlockSize(
    uint64_t outer_size, uint64_t dim_size, uint64_t inner_size) {
  uint32_t inner_threads = inner_size;
  inner_threads = std::min(inner_threads, static_cast<uint32_t>(1024));
  uint32_t dim_threads = 1;
  if (inner_threads <= 64 && dim_size >= 64) {
    while (inner_threads * dim_threads <= 1024 && dim_threads <= dim_size)
      dim_threads *= 2;
    dim_threads /= 2;
  }
  return dim3(dim_threads, inner_threads);
}

template<typename AccumT, typename Kernel>
void SpatialSoftMax_getLaunchSizes(
    THCState *state, Kernel k,
    uint64_t outer_size, uint64_t dim_size, uint64_t inner_size,
    dim3& grid, dim3& block, uint32_t& smem_size) {
  block = SpatialSoftMax_getBlockSize(outer_size, dim_size, inner_size);
  uint32_t block_threads = block.x * block.y;
  smem_size = block.x == 1 ? 0 : block_threads * sizeof(AccumT);
  int max_active_blocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                                                k, block_threads, smem_size);
  max_active_blocks *= THCState_getCurrentDeviceProperties(state)->multiProcessorCount;
  grid = SpatialSoftMax_getGridSize(block, max_active_blocks, outer_size, dim_size, inner_size);
}

inline dim3 SoftMax_getBlockSize(int ILP, uint64_t dim_size) {
  uint64_t block_size = 1;
  uint64_t max_block_size = std::min(dim_size / ILP, static_cast<uint64_t>(1024));
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

template <typename T, typename AccumT, template<typename, typename> class Epilogue>
__global__ void cunn_SpatialSoftMaxForward(
    T *output, T *input,
    uint32_t outer_size, uint32_t dim_size, uint32_t inner_size)
{
  SharedMem<AccumT> smem;
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
        T max_input = THCNumerics<T>::min();
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
          const T value = input[data_offset + d * dim_stride];
          max_input = THCNumerics<T>::ge(max_input, value) ? max_input : value;
        }
        max_input = ScalarConvert<AccumT, T>::to(
            spatialBlockReduceX<AccumT, Max>(smem.getPointer(),
                                        ScalarConvert<T, AccumT>::to(max_input)));

        AccumT sum = 0;
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          sum += THCNumerics<T>::exp(input[data_offset + d * dim_stride] - max_input);
        sum = spatialBlockReduceX<AccumT, Add>(smem.getPointer(), sum);

        Epilogue<T, AccumT> epilogue(max_input, sum);
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          output[data_offset + d * dim_stride] = epilogue(input[data_offset + d * dim_stride]);
      } else {
        T max_input = THCNumerics<T>::min();
        for (uint32_t d = 0; d < dim_size; d++) {
          const T value = input[data_offset + d * dim_stride];
          max_input = THCNumerics<T>::ge(max_input, value) ? max_input : value;
        }

        AccumT sum = 0;
        for (uint32_t d = 0; d < dim_size; d++)
          sum += THCNumerics<T>::exp(input[data_offset + d * dim_stride] - max_input);

        Epilogue<T, AccumT> epilogue(max_input, sum);
        for (uint32_t d = 0; d < dim_size; d++)
          output[data_offset + d * dim_stride] = epilogue(input[data_offset + d * dim_stride]);
      }
    }
  }
}

template <typename T, typename AccumT, template<typename, typename> class Epilogue>
__global__ void cunn_SpatialSoftMaxBackward(
    T *gradInput, T *output, T *gradOutput,
    uint32_t outer_size, uint32_t dim_size, uint32_t inner_size)
{
  SharedMem<AccumT> smem;
  const uint32_t outer_stride = inner_size * dim_size;
  const uint32_t dim_stride = inner_size;

  for (uint32_t outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x) {
    const uint32_t outer_offset = outer_index * outer_stride;
    for (uint32_t inner_index = blockIdx.y * blockDim.y + threadIdx.y; inner_index < inner_size; inner_index += blockDim.y * gridDim.y) {
      const uint32_t data_offset = outer_offset + inner_index;
      // See the comment in forward kernel
      if (blockDim.x > 1) {
        AccumT sum = 0;
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          sum += gradOutput[data_offset + d * dim_stride];
        sum = spatialBlockReduceX<AccumT, Add>(smem.getPointer(), sum);

        Epilogue<T, AccumT> epilogue(sum);
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
          gradInput[data_offset + d * dim_stride] =
            epilogue(gradOutput[data_offset + d * dim_stride],
                    output[data_offset + d * dim_stride]);
        }
      } else {
        AccumT sum = 0;
        for (uint32_t d = 0; d < dim_size; d++)
          sum += gradOutput[data_offset + d * dim_stride];

        Epilogue<T, AccumT> epilogue(sum);
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
  __device__ __forceinline__ SumExpFloat(T v)
    : max_k(v) {}

  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + THCNumerics<T>::exp(v - max_k);
  }

  const T max_k;
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

template <int ILP, typename T, typename AccumT, template <typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxForward(T *output, T *input, int classes)
{
  SharedMem<AccumT> smem;
  AccumT *buffer = smem.getPointer();
  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * classes;
  output += blockIdx.x * classes;

  // find the max
  AccumT threadMax = ilpReduce<MaxFloat, ILP, T, AccumT>(
      input, classes, MaxFloat<T, AccumT>(), -THCNumerics<AccumT>::max());
  AccumT max_k = blockReduce<Max, AccumT>(
      buffer, threadMax, Max<AccumT>(), -THCNumerics<AccumT>::max());
  T max_k_non_accum = ScalarConvert<AccumT, T>::to(max_k);

  // reduce all values
  AccumT threadExp = ilpReduce<SumExpFloat, ILP, T, AccumT>(
      input, classes, SumExpFloat<T, AccumT>(max_k_non_accum), static_cast<AccumT>(0));
  AccumT sumAll = blockReduce<Add, AccumT>(
      buffer, threadExp, Add<AccumT>(), static_cast<AccumT>(0));

  Epilogue<T, AccumT> epilogue(max_k_non_accum, sumAll);
  int offset = threadIdx.x;
  int last = classes % (ILP * blockDim.x);
  for (; offset < classes - last; offset += blockDim.x * ILP) {
    T tmp[ILP];

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

template <int ILP, typename T, typename AccumT, template<typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxBackward(T *gradInput, T *output, T *gradOutput, int classes)
{
  SharedMem<AccumT> smem;
  AccumT *buffer = smem.getPointer();
  gradInput += blockIdx.x * classes;
  output += blockIdx.x * classes;
  gradOutput += blockIdx.x * classes;

  AccumT threadSum = ilpReduce<AddFloat, 4, T, AccumT>(
      gradOutput, classes, AddFloat<T, AccumT>(), AccumT(0));
  AccumT sum_k = blockReduce<Add, AccumT>(
        buffer, threadSum, Add<AccumT>(), AccumT(0));

  Epilogue<T, AccumT> epilogue(sum_k);
  int offset = threadIdx.x;
  int last = classes % (ILP * blockDim.x);
  for (; offset < classes - last; offset += blockDim.x * ILP) {
    T tmpGradOutput[ILP];
    T tmpOutput[ILP];

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

template<typename T, typename AccumT, template<typename, typename> class Epilogue>
void HostSoftMaxForward(
          THCState *state,
          T *input, T *output,
          uint64_t outer_size, uint64_t dim_size, uint64_t inner_size,
          int dim)
{
  // This kernel spawns a block per each element in the batch.
  // XXX: it assumes that inner_size == 1
  if (inner_size == 1) {
    const int ILP = 2;
    dim3 grid(outer_size);
    dim3 block = SoftMax_getBlockSize(ILP, dim_size);

    cunn_SoftMaxForward<ILP, T, AccumT, Epilogue>
      <<<grid, block, block.x * sizeof(AccumT), THCState_getCurrentStream(state)>>>(
        output, input, dim_size
    );
  // This kernel runs in a 2D grid, where each application along y dimension has a fixed
  // outer_size, and runs in parallel over inner_size. Dimension x is parallel over outer_size.
  // Reductions over dim are done in a single-threaded manner.
  } else {
    uint32_t smem_size;
    dim3 grid, block;
    SpatialSoftMax_getLaunchSizes<AccumT>(
        state, &cunn_SpatialSoftMaxForward<T, AccumT, Epilogue>,
        outer_size, dim_size, inner_size,
        grid, block, smem_size);

    cunn_SpatialSoftMaxForward<T, AccumT, Epilogue>
      <<<grid, block, smem_size, THCState_getCurrentStream(state)>>>(
        output, input, outer_size, dim_size, inner_size
    );
  }
  THCudaCheck(cudaGetLastError());
}

template<typename T, typename AccumT, template<typename, typename> class Epilogue>
void HostSoftMaxBackward(
          THCState *state,
          T *gradOutput, T *gradInput, T *output,
          uint64_t outer_size, uint64_t dim_size, uint64_t inner_size,
          int dim)
{
  // See descriptions of kernels above.
  if (inner_size == 1) {
    const int ILP = 2;
    dim3 grid(outer_size);
    dim3 block = SoftMax_getBlockSize(ILP, dim_size);

    cunn_SoftMaxBackward<ILP, T, AccumT, Epilogue>
      <<<grid, block, block.x * sizeof(AccumT), THCState_getCurrentStream(state)>>>(
        gradInput, output, gradOutput, dim_size
    );
  } else {
    uint32_t smem_size;
    dim3 grid, block;
    SpatialSoftMax_getLaunchSizes<AccumT>(
        state, &cunn_SpatialSoftMaxBackward<T, AccumT, Epilogue>,
        outer_size, dim_size, inner_size,
        grid, block, smem_size);

    cunn_SpatialSoftMaxBackward<T, AccumT, Epilogue>
      <<<grid, block, smem_size, THCState_getCurrentStream(state)>>>(
        gradInput, output, gradOutput, outer_size, dim_size, inner_size
    );
  }
  THCudaCheck(cudaGetLastError());
}
