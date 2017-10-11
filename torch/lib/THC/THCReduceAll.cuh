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
          typename ReduceAccOp,
          typename InT,
          typename AccT,
          typename IndexType,
          int ADims>
__global__ void
kernelReduceAll(TensorInfo<InT, IndexType> in,
                IndexType totalElements,
                AccT init,
                ModifyOp modifyOp,
                ReduceOp reduceOp,
                ReduceAccOp reduceAccOp,
                AccT* out) {
  // With a block-wide stride, have each thread perform its own reduction.
  AccT r = init;
  for (IndexType i = threadIdx.x; i < totalElements; i += blockDim.x) {
    const IndexType inOffset = IndexToOffset<InT, IndexType, ADims>::get(i, in);
    r = reduceOp(r, modifyOp(in.data[inOffset]));
  }

  // Reduce within the block
  extern __shared__ char smemChar[];
  AccT* smem = (AccT*) smemChar;
  r = reduceBlock<AccT, ReduceAccOp>(smem, blockDim.x, r, reduceAccOp, init);

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
          typename ReduceAccOp,
          typename InT,
          typename AccT,
          typename IndexType,
          int ADims>
__global__ void
kernelReduceAllPass1(TensorInfo<InT, IndexType> in,
                     IndexType totalElements,
                     AccT init,
                     ModifyOp modifyOp,
                     ReduceOp reduceOp,
                     ReduceAccOp reduceAccOp,
                     AccT* scratchSpace) {
  const IndexType startIndex = getStartIndex<IndexType>(totalElements);
  const IndexType endIndex = getEndIndex<IndexType>(totalElements);

  // With a block-wide stride, have each thread perform its own reduction.
  AccT r = init;
  for (IndexType i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
    const IndexType inOffset = IndexToOffset<InT, IndexType, ADims>::get(i, in);
    r = reduceOp(r, modifyOp(in.data[inOffset]));
  }

  // Reduce within the block
  extern __shared__ char smemChar[];
  AccT* smem = (AccT*) smemChar;
  r = reduceBlock<AccT, ReduceAccOp>(smem, blockDim.x, r, reduceAccOp, init);

  if (threadIdx.x == 0) {
    // Write out block-wide reduced value
    scratchSpace[blockIdx.x] = r;
  }
}

template <typename ReduceOp, typename T, typename IndexType>
__global__ void
kernelReduceAllPass2(int numPass1Blocks,
                     T init,
                     ReduceOp reduceOp,
                     T* scratchSpace,
                     T* out) {
  T r = init;
  if (threadIdx.x < numPass1Blocks) {
    r = scratchSpace[threadIdx.x];
  }

  // Reduce within the block
  extern __shared__ char smemChar[];
  T* smem = (T*) smemChar;
  r = reduceBlock<T, ReduceOp>(smem, numPass1Blocks, r, reduceOp, init);

  if (threadIdx.x == 0) {
    *out = r;
  }
}

// Perform a two-pass reduction if the tensor is large enough to
// warrant it.
inline bool isTwoPassReductionSize(ptrdiff_t elements) {
  return (elements > THC_TWO_PASS_REDUCTION_SIZE);
}

template <typename InT, typename AccT>
inline ptrdiff_t getTwoPassBlocks(THCState* state, ptrdiff_t elements) {
  ptrdiff_t numBlocks = THCCeilDiv(elements, (ptrdiff_t)THC_REDUCE_ALL_BLOCK_SIZE);

  // We can only have as many blocks as there is scratch space
  ptrdiff_t scratchSpace =
    THCState_getCurrentDeviceScratchSpaceSize(state) / sizeof(AccT);
  THAssert(scratchSpace > 0);

  // Limit to 1024 due to dimensionality constraint
  if (scratchSpace > 1024) {
    scratchSpace = 1024;
  }

  if (numBlocks > scratchSpace) {
    numBlocks = scratchSpace;
  }

  return numBlocks;
}

// Get the block/grid size that we want
template <typename InT, typename AccT>
inline void getPass1ReduceBlockGrid(THCState* state, ptrdiff_t elements,
                                    dim3& grid, dim3& block) {
  grid = dim3(getTwoPassBlocks<InT, AccT>(state, elements));
  block = dim3(THC_REDUCE_ALL_BLOCK_SIZE);
}

template <typename InT, typename AccT>
inline void getPass2ReduceBlockGrid(THCState* state, ptrdiff_t elements,
                                    dim3& grid, dim3& block) {
  grid = dim3(1);
  // We only need as many threads as there were blocks originally
  block = dim3(getTwoPassBlocks<InT, AccT>(state, elements));
}

template <typename InT, typename AccT>
inline void getSinglePassReduceBlockGrid(ptrdiff_t elements,
                                         dim3& grid, dim3& block) {
  grid = dim3(1);
  block = dim3(THC_REDUCE_ALL_BLOCK_SIZE);
}

template <typename ModifyOp,
          typename ReduceOp,
          typename ReduceAccOp,
          typename InT,
          typename AccT,
          typename IndexType,
          int ADims>
void callReduceAll(THCState* state,
                   const TensorInfo<InT, IndexType>& in,
                   ptrdiff_t totalElements,
                   AccT init,
                   const ModifyOp& modifyOp,
                   const ReduceOp& reduceOp,
                   const ReduceAccOp& reduceAccOp,
                   AccT* devOut) {
  dim3 grid;
  dim3 block;

  if (isTwoPassReductionSize(totalElements)) {
    bool freeScratchSpace = false;
    void* scratchSpace = THCState_getCurrentDeviceScratchSpace(state);
    if (!scratchSpace) {
      THCudaCheck(THCudaMalloc(state, &scratchSpace,
          THCState_getCurrentDeviceScratchSpaceSize(state)));
      freeScratchSpace = true;
    }

    getPass1ReduceBlockGrid<InT, AccT>(state, totalElements, grid, block);
    size_t smemSize = block.x * sizeof(AccT);

    kernelReduceAllPass1<ModifyOp, ReduceOp, ReduceAccOp, InT, AccT, IndexType, ADims>
      <<<grid, block, smemSize, THCState_getCurrentStream(state)>>>(
        in, (IndexType) totalElements, init, modifyOp, reduceOp, reduceAccOp,
        (AccT*) scratchSpace);

    int numPass1Blocks = grid.x;
    getPass2ReduceBlockGrid<InT, AccT>(state, totalElements, grid, block);
    smemSize = block.x * sizeof(AccT);

    kernelReduceAllPass2<ReduceAccOp, AccT, IndexType>
      <<<grid, block, smemSize, THCState_getCurrentStream(state)>>>(
        numPass1Blocks, init, reduceAccOp,
        (AccT*) scratchSpace, devOut);

    if (freeScratchSpace) {
      THCudaCheck(THCudaFree(state, scratchSpace));
    }
  } else {
    getSinglePassReduceBlockGrid<InT, AccT>(totalElements, grid, block);
    size_t smemSize = block.x * sizeof(AccT);

    kernelReduceAll<ModifyOp, ReduceOp, ReduceAccOp, InT, AccT, IndexType, ADims>
      <<<grid, block, smemSize, THCState_getCurrentStream(state)>>>(
        in, (IndexType) totalElements, init, modifyOp, reduceOp, reduceAccOp, devOut);
  }
}

// Reduces the entire tensor to one value. `out` points to
// host-resident memory.
template <typename TensorType,
          typename ModifyOp,
          typename ReduceOp,
          typename ReduceAccOp,
          typename AccT>
bool THC_reduceAll(THCState* state,
                   TensorType* in,
                   const ModifyOp& modifyOp,
                   const ReduceOp& reduceOp,
                   const ReduceAccOp& reduceAccOp,
                   AccT init,
                   AccT* out,
                   int outOnDevice) {
  ptrdiff_t inElements = TensorUtils<TensorType>::getNumElements(state, in);

  if (TensorUtils<TensorType>::getDims(state, in) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (TensorUtils<TensorType>::getDims(state, in) == 0) {
    // Zero-dim tensor; do nothing
    *out = init;
    return true;
  }

  bool freeDevOut = false;
  AccT* devOut = out;
  if (!outOnDevice) {
    // Use the stream-specific scratch space for the reduction kernel
    // to write out its value
    devOut = (AccT*) THCState_getCurrentDeviceScratchSpace(state);
    if (!devOut) {
      THCudaCheck(THCudaMalloc(state, (void**)&devOut,
          THCState_getCurrentDeviceScratchSpaceSize(state)));
      freeDevOut = true;
    }
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
  callReduceAll<ModifyOp, ReduceOp, ReduceAccOp,                        \
                typename TensorUtils<TensorType>::DataType,             \
                AccT,                                                   \
                TYPE, IN>(                                              \
                  state, inInfo, inElements, init, modifyOp,            \
                  reduceOp, reduceAccOp, devOut);

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
        default:                                    \
          HANDLE_CASE(TYPE, -1);                    \
          break;                                    \
      }                                             \
    }                                               \
  }

  if (TensorUtils<TensorType>::canUse32BitIndexMath(state, in)) {
    TensorInfo<typename TensorUtils<TensorType>::DataType, unsigned int> inInfo =
      getTensorInfo<TensorType, unsigned int>(state, in);
    inInfo.collapseDims();

    HANDLE_IN_CASE(unsigned int, inInfo.dims);
  } else {
    TensorInfo<typename TensorUtils<TensorType>::DataType,
               uint64_t> inInfo =
      getTensorInfo<TensorType, uint64_t>(state, in);
    inInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (inInfo.isContiguous()) {
      HANDLE_IN_CASE(uint64_t, -2);
    } else {
      HANDLE_IN_CASE(uint64_t, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_IN_CASE

  // If our destination is not on the device, copy the value back to
  // the host (synchronous!)
  if (!outOnDevice) {
    cudaStream_t stream = THCState_getCurrentStream(state);
    THCudaCheck(cudaMemcpyAsync(out, 
                                devOut, 
                                sizeof(AccT), 
                                cudaMemcpyDeviceToHost, 
                                stream));
    THCudaCheck(cudaStreamSynchronize(stream));
  }

  if (freeDevOut) {
    THCudaCheck(THCudaFree(state, devOut));
  }

  return true;
}

#undef THC_REDUCE_ALL_BLOCK_SIZE
#undef THC_TWO_PASS_REDUCTION_SIZE

#endif // THC_REDUCEALL_INC
