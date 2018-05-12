#ifndef THC_REDUCE_INC
#define THC_REDUCE_INC

//
// This file contains dimension reduction operation functions and
// kernels that work on both contiguous and non-contiguous tensor
// arguments of arbitrary (up to MAX_CUTORCH_DIMS) dimensioned
// arguments without copying or temporary storage.
//

#include "THCTensorTypeUtils.cuh"
#include "THCReduceApplyUtils.cuh"
#include "THCNumerics.cuh"

// Threads per thread block
#define THC_NONCONTIG_REDUCE_BLOCK_SIZE 32 * 16

template <typename IndexType>
__device__ __forceinline__ IndexType getReduceNoncontigDimSliceIndex() {
  // Each thread handles one slice
  return getLinearBlockId<IndexType>() * THC_NONCONTIG_REDUCE_BLOCK_SIZE + threadIdx.x;
}

// Kernel that handles an entire reduction of a slice of a tensor per each thread
template <typename T,
          typename IndexType,
          typename AccT,
          typename ModifyOp,
          typename ReduceOp,
          typename FinalizeOp,
          int ADims, int BDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
kernelReduceNoncontigDim_shared(TensorInfo<T, IndexType> out,
                         TensorInfo<T, IndexType> in,
                         IndexType reductionStride,
                         IndexType reductionSize,
                         IndexType totalSlices,
                         AccT init,
                         ModifyOp modifyOp,
                         ReduceOp reduceOp,
                         FinalizeOp finalizeOp) {

  IndexType sliceIndex  = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType sliceStride = gridDim.x * blockDim.x;

  __shared__ AccT local_reduce[THC_NONCONTIG_REDUCE_BLOCK_SIZE];
  AccT* shmem = &local_reduce[threadIdx.x + threadIdx.y * blockDim.x];
  AccT load_reg[4];
  AccT local_reg;

  for(;sliceIndex<totalSlices; sliceIndex+=sliceStride) {
    local_reg = init;

    const IndexType outOffset =
      IndexToOffset<T, IndexType, ADims>::get(sliceIndex, out);
    const IndexType inOffset =
      IndexToOffset<T, IndexType, BDims>::get(sliceIndex, in);

    //Unroll this loop
    //for(IndexType i=threadIdx.y; i<reductionSize; i+=blockDim.y){
    //  local_reg += in[inOffset + i * reductionStride];
    //}
    for(IndexType i = threadIdx.y; i < reductionSize; i += blockDim.y * 4) {
      if (i + blockDim.y * 3 < reductionSize) {
        const AccT val0 = scalar_cast<AccT>(in.data[inOffset + i * reductionStride]);
        load_reg[0] = modifyOp(val0);
        const AccT val1 = scalar_cast<AccT>(in.data[inOffset + (i + blockDim.y) * reductionStride]);
        load_reg[1] = modifyOp(val1);
        const AccT val2 = scalar_cast<AccT>(in.data[inOffset + (i + blockDim.y * 2) * reductionStride]);
        load_reg[2] = modifyOp(val2);
        const AccT val3 = scalar_cast<AccT>(in.data[inOffset + (i + blockDim.y * 3) * reductionStride]);
        load_reg[3] = modifyOp(val3);
        local_reg = reduceOp(local_reg, load_reg[0]);
        local_reg = reduceOp(local_reg, load_reg[1]);
        local_reg = reduceOp(local_reg, load_reg[2]);
        local_reg = reduceOp(local_reg, load_reg[3]);
      } else if (i + blockDim.y * 2 < reductionSize) {
        const AccT val0 = scalar_cast<AccT>(in.data[inOffset + i * reductionStride]);
        load_reg[0] = modifyOp(val0);
        const AccT val1 = scalar_cast<AccT>(in.data[inOffset + (i + blockDim.y) * reductionStride]);
        load_reg[1] = modifyOp(val1);
        const AccT val2 = scalar_cast<AccT>(in.data[inOffset + (i + blockDim.y * 2) * reductionStride]);
        load_reg[2] = modifyOp(val2);
        local_reg = reduceOp(local_reg, load_reg[0]);
        local_reg = reduceOp(local_reg, load_reg[1]);
        local_reg = reduceOp(local_reg, load_reg[2]);
      } else if (i + blockDim.y < reductionSize) {
        const AccT val0 = scalar_cast<AccT>(in.data[inOffset + i * reductionStride]);
        load_reg[0] = modifyOp(val0);
        const AccT val1 = scalar_cast<AccT>(in.data[inOffset + (i + blockDim.y) * reductionStride]);
        load_reg[1] = modifyOp(val1);
        local_reg = reduceOp(local_reg, load_reg[0]);
        local_reg = reduceOp(local_reg, load_reg[1]);
      } else if (i < reductionSize) {
        const AccT val0 = scalar_cast<AccT>(in.data[inOffset + i * reductionStride]);
        local_reg = reduceOp(local_reg, modifyOp(val0));
      }
    }

    *shmem = local_reg;
    int dimy = blockDim.y;
    while (dimy > 1) {
      __syncthreads();
      if (threadIdx.y == 0 && (dimy % 2 != 0) ) {
        *shmem = reduceOp(*shmem, *(shmem + (dimy - 1) * blockDim.x));
      }
      if (threadIdx.y < dimy / 2) {
        *shmem = reduceOp(*shmem, *(shmem + (dimy / 2) * blockDim.x));
      }
      dimy /= 2;
    }
    if (threadIdx.y == 0)
      out.data[outOffset] = scalar_cast<T>(finalizeOp(*shmem));
  }
}


// Kernel that handles an entire reduction of a slice of a tensor per each thread
template <typename T,
          typename IndexType,
          typename AccT,
          typename ModifyOp,
          typename ReduceOp,
          typename FinalizeOp,
          int ADims, int BDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
kernelReduceNoncontigDim(TensorInfo<T, IndexType> out,
                         TensorInfo<T, IndexType> in,
                         IndexType reductionStride,
                         IndexType reductionSize,
                         IndexType totalSlices,
                         AccT init,
                         ModifyOp modifyOp,
                         ReduceOp reduceOp,
                         FinalizeOp finalizeOp) {
  const IndexType sliceIndex = getReduceNoncontigDimSliceIndex<IndexType>();

  if (sliceIndex >= totalSlices) {
    return;
  }

  // Each thread picks a point in `out` and `in` for which it is
  // producing the reduction
  const IndexType outOffset =
    IndexToOffset<T, IndexType, ADims>::get(sliceIndex, out);
  const IndexType inBaseOffset =
    IndexToOffset<T, IndexType, BDims>::get(sliceIndex, in);

  // For each point in reductionSize, reduce into `r`
  IndexType inOffset = inBaseOffset;
  AccT r = init;

  for (IndexType i = 0; i < reductionSize; ++i) {
    const AccT val = scalar_cast<AccT>(in.data[inOffset]);
    r = reduceOp(r, modifyOp(val));
    inOffset += reductionStride;
  }

  // Write out reduced value
  out.data[outOffset] = scalar_cast<T>(finalizeOp(r));
}

template <typename IndexType>
__device__ __forceinline__ IndexType getReduceContigDimSliceIndex() {
  // Each block handles one slice
  return getLinearBlockId<IndexType>();
}

// Kernel that handles an entire reduction of a slice of a tensor per
// each block
template <typename T,
          typename IndexType,
          typename AccT,
          typename ModifyOp,
          typename ReduceOp,
          typename FinalizeOp,
          int ADims, int BDims>
__global__ void
kernelReduceContigDim(TensorInfo<T, IndexType> out,
                      TensorInfo<T, IndexType> in,
                      IndexType reductionSize,
                      IndexType totalSlices,
                      AccT init,
                      ModifyOp modifyOp,
                      ReduceOp reduceOp,
                      FinalizeOp finalizeOp) {
  const IndexType sliceIndex = getReduceContigDimSliceIndex<IndexType>();

  if (sliceIndex >= totalSlices) {
    return;
  }

  // Get the offset in `out` for the reduction
  const IndexType outOffset =
    IndexToOffset<T, IndexType, ADims>::get(sliceIndex, out);

  // Get the base offset in `in` for this block's reduction
  const IndexType inBaseOffset =
    IndexToOffset<T, IndexType, BDims>::get(sliceIndex, in);

  // Each thread in the block will reduce some subset of elements in
  // the slice. The elements are guaranteed contiguous starting at
  // `inBaseOffset`.
  AccT r = init;
  for (IndexType i = threadIdx.x; i < reductionSize; i += blockDim.x) {
    const AccT val = scalar_cast<AccT>(in.data[inBaseOffset + i]);
    r = reduceOp(r, modifyOp(val));
  }

  // Reduce within the block
  // FIXME: extern name
  extern __shared__ char smemChar[];
  AccT* smem = (AccT*) smemChar;
  r = reduceBlock<AccT, ReduceOp>(smem, blockDim.x, r, reduceOp, init);

  if (threadIdx.x == 0) {
    // Write out reduced value
    out.data[outOffset] = scalar_cast<T>(finalizeOp(r));
  }
}

inline dim3 getNoncontigReduceBlock() {
  return dim3(THC_NONCONTIG_REDUCE_BLOCK_SIZE);
}

inline dim3 getContigReduceBlock(ptrdiff_t numSlices, int64_t reductionSize) {
  // If the number of slices is low but the reduction dimension size
  // is high, then we should increase block size for greater parallelism.
  // Aim for at least 32 warps per SM (assume 15 SMs; don't bother
  // inquiring the real number for now).
  int maxWarps = 4; // better occupancy if many blocks are around
  // For numSlices > 15 * 8, there are > 32 warps active per SM.
  if (numSlices < 15 * 8) {
    maxWarps = 8;
    if (numSlices < 15 * 4) {
      maxWarps = 16;
      if (numSlices < 15 * 2) {
        maxWarps = 32;
      }
    }
  }

  // Scale up block size based on the reduction dimension size
  int64_t warpsInReductionSize = THCCeilDiv(reductionSize, (int64_t) 32);
  int numWarps = warpsInReductionSize > (int64_t) maxWarps ?
    maxWarps : (int) warpsInReductionSize;

  return dim3(numWarps * 32);
}

inline bool getNoncontigReduceGrid(ptrdiff_t elements, dim3& grid) {
  // One output point per thread
  return THC_getGridFromTiles(THCCeilDiv(elements,
                                         (ptrdiff_t) THC_NONCONTIG_REDUCE_BLOCK_SIZE), grid);
}

inline bool getContigReduceGrid(ptrdiff_t elements, dim3& grid) {
  // One output point per block
  return THC_getGridFromTiles(elements, grid);
}

// Performs a reduction out[..., 0, ...] = reduce_i(modify(in[..., i, ...])) for
// all in where i and the out's 0 are indexed at dimension `dim`
template <typename TensorType, 
typename ModifyOp, 
typename ReduceOp,
typename FinalizeOp,
typename AccT>
bool THC_reduceDim(THCState* state,
                   TensorType* out,
                   TensorType* in,
                   const ModifyOp modifyOp,
                   const ReduceOp reduceOp,
                   const FinalizeOp finalizeOp,
                   AccT init,
                   int dim,
                   int keepdim) {
  ptrdiff_t inElements = TensorUtils<TensorType>::getNumElements(state, in);

  int64_t reductionSize = TensorUtils<TensorType>::getSize(state, in, dim);
  int64_t reductionStride = TensorUtils<TensorType>::getStride(state, in, dim);
  ptrdiff_t outElements = inElements / reductionSize;

  if (TensorUtils<TensorType>::getDims(state, out) > MAX_CUTORCH_DIMS ||
      TensorUtils<TensorType>::getDims(state, in) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (TensorUtils<TensorType>::getDims(state, in) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  // Is the reduction dimension contiguous? If so, then we can use a
  // shared memory reduction kernel to increase performance.
  bool contigReduction = (reductionStride == 1);

  dim3 block;
  dim3 grid;
  int smemSize = 0; // contiguous reduction uses smem
  if (contigReduction) {
    if (!getContigReduceGrid(outElements, grid)) {
      return false;
    }

    block = getContigReduceBlock(outElements, reductionSize);
    smemSize = sizeof(AccT) * block.x;
  } else {
    if (!getNoncontigReduceGrid(outElements, grid)) {
      return false;
    }

    block = getNoncontigReduceBlock();

    if(outElements <= 4096){
        //x dim does different columns
        //y dim helps with the same reduction
        //If we only have 8 loops, don't bother sharing work across ydim
        uint64_t ydim = THCCeilDiv(reductionSize, (int64_t) 8L);

        //don't want y dim any bigger than 16, leaving min x dim to 32
        ydim = min((uint64_t) 16, ydim);

        block = dim3(THC_NONCONTIG_REDUCE_BLOCK_SIZE, 1, 1);
        while(ydim > 1){
          block.x /= 2;
          block.y *= 2;
          ydim /= 2;
        }
        THC_getGridFromTiles(THCCeilDiv((int64_t)outElements, (int64_t)block.x), grid);

    }
  }

  // Resize out to correspond to the reduced size with keepdim=True.

  // Preserve noncontiguities by unsqueezing out if necessary
  TensorUtils<TensorType>::preserveReduceDimSemantics(
      state, out, TensorUtils<TensorType>::getDims(state, in), dim, keepdim);

  // Resize out
  THLongStorage* sizes = TensorUtils<TensorType>::newSizeOf(state, in);
  THLongStorage_set(sizes, dim, 1);
  TensorUtils<TensorType>::resize(state, out, sizes, NULL);
  THLongStorage_free(sizes);

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, OUT, IN)                                      \
  if (contigReduction) {                                                \
    kernelReduceContigDim<typename TensorUtils<TensorType>::DataType,   \
                          TYPE, AccT, ModifyOp, ReduceOp, FinalizeOp,   \
                          OUT, IN>                                      \
      <<<grid, block, smemSize, THCState_getCurrentStream(state)>>>     \
        (outInfo, inInfo, reductionSize,                                \
        (TYPE) outElements, init, modifyOp, reduceOp, finalizeOp);      \
  } else {                                                              \
    if(block.y == 1){                                                   \
        kernelReduceNoncontigDim<                                       \
                          typename TensorUtils<TensorType>::DataType,   \
                          TYPE, AccT, ModifyOp, ReduceOp, FinalizeOp,   \
                          OUT, IN>                                      \
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>          \
        (outInfo, inInfo, reductionStride, reductionSize,               \
        (TYPE) outElements, init, modifyOp, reduceOp, finalizeOp);      \
    }else{                                                              \
        kernelReduceNoncontigDim_shared<                                \
                          typename TensorUtils<TensorType>::DataType,   \
                          TYPE, AccT, ModifyOp, ReduceOp, FinalizeOp,   \
                          OUT, IN>                                      \
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>          \
        (outInfo, inInfo, reductionStride, reductionSize,               \
        (TYPE) outElements, init, modifyOp, reduceOp, finalizeOp);      \
    }                                                                   \
  }                                                                     \

#define HANDLE_IN_CASE(TYPE, OUT, IN)                     \
  {                                                       \
    switch (IN) {                                         \
      case 1:                                             \
        HANDLE_CASE(TYPE, OUT, 1);                        \
        break;                                            \
      case 2:                                             \
        HANDLE_CASE(TYPE, OUT, 2);                        \
        break;                                            \
      default:                                            \
        HANDLE_CASE(TYPE, OUT, -1);                       \
        break;                                            \
    }                                                     \
  }

#define HANDLE_OUT_CASE(TYPE, OUT, IN)                    \
  {                                                       \
    switch (OUT) {                                        \
      case 1:                                             \
        HANDLE_IN_CASE(TYPE, 1, IN);                      \
        break;                                            \
      case 2:                                             \
        HANDLE_IN_CASE(TYPE, 2, IN);                      \
        break;                                            \
      default:                                            \
        HANDLE_IN_CASE(TYPE, -1, IN);                     \
        break;                                            \
    }                                                     \
  }

  if (TensorUtils<TensorType>::canUse32BitIndexMath(state, out) &&
      TensorUtils<TensorType>::canUse32BitIndexMath(state, in)) {
    TensorInfo<typename TensorUtils<TensorType>::DataType,
               unsigned int> outInfo =
      getTensorInfo<TensorType, unsigned int>(state, out);
    outInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorType>::DataType,
               unsigned int> inInfo =
      getTensorInfo<TensorType, unsigned int>(state, in);
    inInfo.reduceDim(dim);
    inInfo.collapseDims();
    HANDLE_OUT_CASE(unsigned int, outInfo.dims, inInfo.dims);
  } else {
    TensorInfo<typename TensorUtils<TensorType>::DataType,
               uint64_t> outInfo =
      getTensorInfo<TensorType, uint64_t>(state, out);
    outInfo.collapseDims();

    TensorInfo<typename TensorUtils<TensorType>::DataType,
               uint64_t> inInfo =
      getTensorInfo<TensorType, uint64_t>(state, in);
    inInfo.reduceDim(dim);
    inInfo.collapseDims();

    /*
    Only instantiates the all 1D special case and the fallback all nD case for
    large (64-bit indexed) tensors to reduce compilation time. 
    */
    if (outInfo.dims == 1 && inInfo.dims == 1) {
      HANDLE_CASE(uint64_t, 1, 1);
    } else {
      HANDLE_CASE(uint64_t, -1, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_IN_CASE
#undef HANDLE_OUT_CASE


  if (!keepdim) {
    TensorUtils<TensorType>::squeeze1d(state, out, out, dim);
  }
  return true;
}

#undef THC_NONCONTIG_REDUCE_BLOCK_SIZE

#endif // THC_REDUCE_INC
