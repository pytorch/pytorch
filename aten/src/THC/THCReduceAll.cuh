#ifndef THC_REDUCEALL_INC
#define THC_REDUCEALL_INC

//
// This file contains dimension reduction operation functions and
// kernels that work on both contiguous and non-contiguous tensor
// arguments of arbitrary (up to MAX_CUTORCH_DIMS) dimensioned
// arguments without copying or temporary storage, for reducing an
// entire tensor to one value.
//

#include <THC/THCReduceApplyUtils.cuh>
#include <c10/macros/Macros.h>

// Size per each reduction block
#define THC_REDUCE_ALL_BLOCK_SIZE 1024L

// Cutoff size for two-pass reduction
#define THC_TWO_PASS_REDUCTION_SIZE 2048L

// Kernel that handles an entire reduction of a tensor in one pass
template <typename T,
          typename IndexType,
          typename AccT,
          typename ModifyOp,
          typename ReduceOp,
          int ADims>
__global__ void
#if defined(__HIP_PLATFORM_HCC__)
C10_LAUNCH_BOUNDS_1(THC_REDUCE_ALL_BLOCK_SIZE)
#endif
kernelReduceAll(TensorInfo<T, IndexType> in,
                IndexType totalElements,
                AccT init,
                ModifyOp modifyOp,
                ReduceOp reduceOp,
                AccT* out) {
  // With a block-wide stride, have each thread perform its own reduction.
  AccT r = init;
  for (IndexType i = threadIdx.x; i < totalElements; i += blockDim.x) {
    const IndexType inOffset = IndexToOffset<T, IndexType, ADims>::get(i, in);
    const AccT val = scalar_cast<AccT>(in.data[inOffset]);
    r = reduceOp(r, modifyOp(val));
  }

  // Reduce within the block
  extern __shared__ char smemChar[];
  AccT* smem = (AccT*) smemChar;
  r = reduceBlock(smem, blockDim.x, r, reduceOp, init);

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
template <typename T,
          typename IndexType,
          typename AccT,
          typename ModifyOp,
          typename ReduceOp,
          int ADims>
#if defined(__HIP_PLATFORM_HCC__)
C10_LAUNCH_BOUNDS_1(THC_REDUCE_ALL_BLOCK_SIZE)
#endif
__global__ void
kernelReduceAllPass1(TensorInfo<T, IndexType> in,
                     IndexType totalElements,
                     AccT init,
                     ModifyOp modifyOp,
                     ReduceOp reduceOp,
                     AccT* scratchSpace) {
  const IndexType startIndex = getStartIndex<IndexType>(totalElements);
  const IndexType endIndex = getEndIndex<IndexType>(totalElements);

  // With a block-wide stride, have each thread perform its own reduction.
  AccT r = init;
  for (IndexType i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
    const IndexType inOffset = IndexToOffset<T, IndexType, ADims>::get(i, in);
    const AccT val = scalar_cast<AccT>(in.data[inOffset]);
    r = reduceOp(r, modifyOp(val));
  }

  // Reduce within the block
  extern __shared__ char smemChar[];
  AccT* smem = (AccT*) smemChar;
  r = reduceBlock(smem, blockDim.x, r, reduceOp, init);

  if (threadIdx.x == 0) {
    // Write out block-wide reduced value
    scratchSpace[blockIdx.x] = r;
  }
}

template <typename T, typename ReduceOp>
#if defined(__HIP_PLATFORM_HCC__)
C10_LAUNCH_BOUNDS_1(THC_REDUCE_ALL_BLOCK_SIZE)
#endif
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
  r = reduceBlock(smem, numPass1Blocks, r, reduceOp, init);

  if (threadIdx.x == 0) {
    *out = r;
  }
}

// Perform a two-pass reduction if the tensor is large enough to
// warrant it.
inline bool isTwoPassReductionSize(ptrdiff_t elements) {
  return (elements > THC_TWO_PASS_REDUCTION_SIZE);
}

template <typename T>
inline ptrdiff_t getTwoPassBlocks(THCState* state, ptrdiff_t elements) {
  ptrdiff_t numBlocks = THCCeilDiv(elements, (ptrdiff_t)THC_REDUCE_ALL_BLOCK_SIZE);

  // We can only have as many blocks as there is scratch space
  ptrdiff_t scratchSpace =
    THCState_getCurrentDeviceScratchSpaceSize(state) / sizeof(T);
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
template <typename T>
inline void getPass1ReduceBlockGrid(THCState* state, ptrdiff_t elements,
                                    dim3& grid, dim3& block) {
  grid = dim3(getTwoPassBlocks<T>(state, elements));
  block = dim3(THC_REDUCE_ALL_BLOCK_SIZE);
}

template <typename T>
inline void getPass2ReduceBlockGrid(THCState* state, ptrdiff_t elements,
                                    dim3& grid, dim3& block) {
  grid = dim3(1);
  // We only need as many threads as there were blocks originally
  block = dim3(getTwoPassBlocks<T>(state, elements));
}

inline void getSinglePassReduceBlockGrid(ptrdiff_t elements,
                                         dim3& grid, dim3& block) {
  grid = dim3(1);
  block = dim3(THC_REDUCE_ALL_BLOCK_SIZE);
}

template <typename T,
          typename IndexType,
          typename AccT,
          typename ModifyOp,
          typename ReduceOp,
          int ADims>
void callReduceAll(THCState* state,
                   const TensorInfo<T, IndexType>& in,
                   ptrdiff_t totalElements,
                   AccT init,
                   const ModifyOp& modifyOp,
                   const ReduceOp& reduceOp,
                   AccT* devOut) {
  dim3 grid;
  dim3 block;

  if (isTwoPassReductionSize(totalElements)) {
    void* scratchSpace = THCudaMalloc(state, THCState_getCurrentDeviceScratchSpaceSize(state));

    getPass1ReduceBlockGrid<AccT>(state, totalElements, grid, block);
    size_t smemSize = block.x * sizeof(AccT);

    kernelReduceAllPass1<T, IndexType, AccT, ModifyOp, ReduceOp, ADims>
      <<<grid, block, smemSize, c10::cuda::getCurrentCUDAStream()>>>(
        in, (IndexType) totalElements, init, modifyOp, reduceOp,
        (AccT*) scratchSpace);

    int numPass1Blocks = grid.x;
    getPass2ReduceBlockGrid<AccT>(state, totalElements, grid, block);
    smemSize = block.x * sizeof(AccT);

    kernelReduceAllPass2<AccT, ReduceOp>
      <<<grid, block, smemSize, c10::cuda::getCurrentCUDAStream()>>>(
        numPass1Blocks, init, reduceOp,
        (AccT*) scratchSpace, devOut);

    THCudaFree(state, scratchSpace);
  } else {
    getSinglePassReduceBlockGrid(totalElements, grid, block);
    size_t smemSize = block.x * sizeof(AccT);

    kernelReduceAll<T, IndexType, AccT, ModifyOp, ReduceOp, ADims>
      <<<grid, block, smemSize, c10::cuda::getCurrentCUDAStream()>>>(
        in, (IndexType) totalElements, init, modifyOp, reduceOp, devOut);
  }
}

// Reduces the entire tensor to one value. `out` points to
// host-resident memory.
template <typename ScalarType,
          typename TensorType,
          typename ModifyOp,
          typename ReduceOp,
          typename AccT>
bool THC_reduceAll(THCState* state,
                   TensorType* in,
                   const ModifyOp& modifyOp,
                   const ReduceOp& reduceOp,
                   AccT init,
                   AccT* out,
                   int outOnDevice) {
  ptrdiff_t inElements = THCTensor_nElement(state, in);

  if (THCTensor_nDimensionLegacyAll(state, in) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (THCTensor_nDimensionLegacyAll(state, in) == 0) {
    // Zero-dim tensor; do nothing
    *out = init;
    return true;
  }

  bool freeDevOut = false;
  AccT* devOut = out;
  if (!outOnDevice) {
    // Use the stream-specific scratch space for the reduction kernel
    // to write out its value
    devOut = static_cast<AccT*>(THCudaMalloc(state,
        THCState_getCurrentDeviceScratchSpaceSize(state)));
    freeDevOut = true;
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
  callReduceAll<ScalarType,                                             \
                TYPE, AccT, ModifyOp, ReduceOp, IN>(                    \
                  state, inInfo, inElements, init, modifyOp,            \
                  reduceOp, devOut);

#define HANDLE_IN_CASE(TYPE, IN)                    \
  {                                                 \
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
  }

  if (THCTensor_canUse32BitIndexMath(state, in)) {
    TensorInfo<ScalarType, unsigned int> inInfo =
      getTensorInfo<ScalarType, TensorType, unsigned int>(state, in);
    inInfo.collapseDims();

    HANDLE_IN_CASE(unsigned int, inInfo.dims);
  } else {
    TensorInfo<ScalarType,
               uint64_t> inInfo =
      getTensorInfo<ScalarType, TensorType, uint64_t>(state, in);
    inInfo.collapseDims();

    /*
    Only instantiates the all 1D special case and the fallback all nD case for
    large (64-bit indexed) tensors to reduce compilation time.
    */
    if (inInfo.dims == 1) {
      HANDLE_IN_CASE(uint64_t, 1);
    } else {
      HANDLE_IN_CASE(uint64_t, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_IN_CASE

  // If our destination is not on the device, copy the value back to
  // the host (synchronous!)
  if (!outOnDevice) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    THCudaCheck(cudaMemcpyAsync(out,
                                devOut,
                                sizeof(AccT),
                                cudaMemcpyDeviceToHost,
                                stream));
    THCudaCheck(cudaStreamSynchronize(stream));
  }

  if (freeDevOut) {
    THCudaFree(state, devOut);
  }

  return true;
}

#undef THC_REDUCE_ALL_BLOCK_SIZE
#undef THC_TWO_PASS_REDUCTION_SIZE

#endif // THC_REDUCEALL_INC
