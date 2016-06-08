#ifndef THC_REDUCEALL_INC
#define THC_REDUCEALL_INC

//
// This file contains dimension reduction operation functions and
// kernels that work on both contiguous and non-contiguous tensor
// arguments of arbitrary (up to MAX_CUTORCH_DIMS) dimensioned
// arguments without copying or temporary storage, for reducing an
// entire tensor to one value.
//

#include "THCReduceApplyUtils.cuh"

// Size per each reduction block
#define THC_REDUCE_ALL_BLOCK_SIZE 1024L

// Cutoff size for two-pass reduction
#define THC_TWO_PASS_REDUCTION_SIZE 2048L

// Kernel that handles an entire reduction of a tensor in one pass
template <typename ModifyOp,
          typename ReduceOp,
          typename T,
          typename IndexType,
          int ADims>
__global__ void
THCudaTensor_reduceAll(TensorInfo<T, IndexType> in,
                       IndexType totalElements,
                       T init,
                       ModifyOp modifyOp,
                       ReduceOp reduceOp,
                       T* out) {
  // With a block-wide stride, have each thread perform its own reduction.
  T r = init;
  for (IndexType i = threadIdx.x; i < totalElements; i += blockDim.x) {
    const IndexType inOffset = IndexToOffset<T, IndexType, ADims>::get(i, in);
    r = reduceOp(r, modifyOp(in.data[inOffset]));
  }

  // Reduce within the block
  extern __shared__ T smem[];
  r = reduceBlock<T, ReduceOp>(smem, blockDim.x, r, reduceOp, init);

  if (threadIdx.x == 0) {
    // Write out reduced value
    *out = r;
  }
}

template <typename IndexType>
__device__ __forceinline__ IndexType getStartIndex(IndexType totalSize) {
  IndexType sizePerBlock = THCCeilDiv(totalSize, (IndexType) gridDim.x);
  return blockIdx.x * sizePerBlock;
}

template <typename IndexType>
__device__ __forceinline__ IndexType getEndIndex(IndexType totalSize) {
  IndexType sizePerBlock = THCCeilDiv(totalSize, (IndexType) gridDim.x);
  return min((IndexType) ((blockIdx.x + 1) * sizePerBlock), totalSize);
}

// Kernel that handles an entire reduction of a tensor in two passes
template <typename ModifyOp,
          typename ReduceOp,
          typename T,
          typename IndexType,
          int ADims>
__global__ void
THCudaTensor_reduceAllPass1(TensorInfo<T, IndexType> in,
                            IndexType totalElements,
                            T init,
                            ModifyOp modifyOp,
                            ReduceOp reduceOp,
                            T* scratchSpace) {
  const IndexType startIndex = getStartIndex<IndexType>(totalElements);
  const IndexType endIndex = getEndIndex<IndexType>(totalElements);

  // With a block-wide stride, have each thread perform its own reduction.
  T r = init;
  for (IndexType i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
    const IndexType inOffset = IndexToOffset<T, IndexType, ADims>::get(i, in);
    r = reduceOp(r, modifyOp(in.data[inOffset]));
  }

  // Reduce within the block
  extern __shared__ T smem[];
  r = reduceBlock<T, ReduceOp>(smem, blockDim.x, r, reduceOp, init);

  if (threadIdx.x == 0) {
    // Write out block-wide reduced value
    scratchSpace[blockIdx.x] = r;
  }
}

template <typename ReduceOp, typename T, typename IndexType>
__global__ void
THCudaTensor_reduceAllPass2(int numPass1Blocks,
                            T init,
                            ReduceOp reduceOp,
                            T* scratchSpace,
                            T* out) {
  T r = init;
  if (threadIdx.x < numPass1Blocks) {
    r = scratchSpace[threadIdx.x];
  }

  // Reduce within the block
  extern __shared__ T smem[];
  r = reduceBlock<T, ReduceOp>(smem, numPass1Blocks, r, reduceOp, init);

  if (threadIdx.x == 0) {
    *out = r;
  }
}

// Perform a two-pass reduction if the tensor is large enough to
// warrant it.
inline bool isTwoPassReductionSize(long elements) {
  return (elements > THC_TWO_PASS_REDUCTION_SIZE);
}

template <typename T>
inline long getTwoPassBlocks(THCState* state, long elements) {
  long numBlocks = THCCeilDiv(elements, THC_REDUCE_ALL_BLOCK_SIZE);

  // We can only have as many blocks as there is scratch space
  long scratchSpace =
    THCState_getCurrentDeviceScratchSpaceSize(state) / sizeof(T);
  THAssert(scratchSpace > 0);

  if (numBlocks > scratchSpace) {
    numBlocks = scratchSpace;
  }

  return numBlocks;
}

// Get the block/grid size that we want
template <typename T>
inline void getPass1ReduceBlockGrid(THCState* state, long elements,
                                    dim3& grid, dim3& block) {
  grid = dim3(getTwoPassBlocks<T>(state, elements));
  block = dim3(THC_REDUCE_ALL_BLOCK_SIZE);
}

template <typename T>
inline void getPass2ReduceBlockGrid(THCState* state, long elements,
                                    dim3& grid, dim3& block) {
  grid = dim3(1);
  // We only need as many threads as there were blocks originally
  block = dim3(getTwoPassBlocks<T>(state, elements));
}

template <typename T>
inline void getSinglePassReduceBlockGrid(long elements,
                                         dim3& grid, dim3& block) {
  grid = dim3(1);
  block = dim3(THC_REDUCE_ALL_BLOCK_SIZE);
}

template <typename ModifyOp,
          typename ReduceOp,
          typename T,
          typename IndexType,
          int ADims>
void callReduceAll(THCState* state,
                   const TensorInfo<T, IndexType>& in,
                   long totalElements,
                   T init,
                   const ModifyOp& modifyOp,
                   const ReduceOp& reduceOp,
                   T* devOut) {
  dim3 grid;
  dim3 block;

  if (isTwoPassReductionSize(totalElements)) {
    getPass1ReduceBlockGrid<T>(state, totalElements, grid, block);
    size_t smemSize = block.x * sizeof(T);

    THCudaTensor_reduceAllPass1<ModifyOp, ReduceOp, T, IndexType, ADims>
      <<<grid, block, smemSize, THCState_getCurrentStream(state)>>>(
        in, (IndexType) totalElements, init, modifyOp, reduceOp,
        (T*) THCState_getCurrentDeviceScratchSpace(state));

    int numPass1Blocks = grid.x;
    getPass2ReduceBlockGrid<T>(state, totalElements, grid, block);
    smemSize = block.x * sizeof(T);

    THCudaTensor_reduceAllPass2<ReduceOp, T, IndexType>
      <<<grid, block, smemSize, THCState_getCurrentStream(state)>>>(
        numPass1Blocks, init, reduceOp,
        (T*) THCState_getCurrentDeviceScratchSpace(state),
        devOut);

  } else {
    getSinglePassReduceBlockGrid<T>(totalElements, grid, block);
    size_t smemSize = block.x * sizeof(T);

    THCudaTensor_reduceAll<ModifyOp, ReduceOp, T, IndexType, ADims>
      <<<grid, block, smemSize, THCState_getCurrentStream(state)>>>(
        in, (IndexType) totalElements, init, modifyOp, reduceOp, devOut);
  }
}

// Reduces the entire tensor to one floating-point value. `out` points
// to host-resident memory.
template <typename ModifyOp, typename ReduceOp>
bool THCudaTensor_reduceAll(THCState* state,
                            THCudaTensor* in,
                            const ModifyOp& modifyOp,
                            const ReduceOp& reduceOp,
                            float init,
                            float* out,
                            int outOnDevice) {
  long inElements = THCudaTensor_nElement(state, in);

  if (THCudaTensor_nDimension(state, in) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (THCudaTensor_nDimension(state, in) == 0) {
    // Zero-dim tensor; do nothing
    *out = init;
    return true;
  }

  float* devOut = out;
  if (!outOnDevice) {
    // Use the stream-specific scratch space for the reduction kernel
    // to write out its value
    devOut = (float*) THCState_getCurrentDeviceScratchSpace(state);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, IN)                                           \
  callReduceAll<ModifyOp, ReduceOp,                                     \
                typename TensorUtils<THCudaTensor>::DataType,           \
                TYPE, IN>(                                              \
                  state, inInfo, inElements, init, modifyOp, reduceOp, devOut);

#define HANDLE_IN_CASE(TYPE, IN)                    \
  {                                                 \
    if (inInfo.isContiguous()) {                    \
      HANDLE_CASE(TYPE, -2);                        \
    } else {                                        \
      switch (IN) {                                 \
        case 1:                                     \
          HANDLE_CASE(TYPE, 1);                     \
          break;                                    \
        case 2:                                     \
          HANDLE_CASE(TYPE, 2);                     \
          break;                                    \
        case 3:                                     \
          HANDLE_CASE(TYPE, 3);                     \
          break;                                    \
        default:                                    \
          HANDLE_CASE(TYPE, -1);                    \
          break;                                    \
      }                                             \
    }                                               \
  }

  if (TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, in)) {
    TensorInfo<float, unsigned int> inInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, in);
    inInfo.collapseDims();

    HANDLE_IN_CASE(unsigned int, inInfo.dims);
  } else {
    TensorInfo<float, unsigned long long> inInfo =
      getTensorInfo<THCudaTensor, unsigned long long>(state, in);
    inInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (inInfo.isContiguous()) {
      HANDLE_IN_CASE(unsigned long long, -2);
    } else {
      HANDLE_IN_CASE(unsigned long long, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_IN_CASE

  // If our destination is not on the device, copy the value back to
  // the host (synchronous!)
  if (!outOnDevice) {
    cudaMemcpy(out, devOut, sizeof(float), cudaMemcpyDeviceToHost);
  }

  return true;
}

#undef THC_REDUCE_ALL_BLOCK_SIZE
#undef THC_TWO_PASS_REDUCTION_SIZE

#endif // THC_REDUCEALL_INC
