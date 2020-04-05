#ifndef THC_REDUCE_INC
#define THC_REDUCE_INC

//
// This file contains dimension reduction operation functions and
// kernels that work on both contiguous and non-contiguous tensor
// arguments of arbitrary (up to MAX_CUTORCH_DIMS) dimensioned
// arguments without copying or temporary storage.
//

#include <THC/THCTensorTypeUtils.cuh>
#include <THC/THCReduceApplyUtils.cuh>
#include <THC/THCNumerics.cuh>
#include <c10/macros/Macros.h>

// Threads per thread block
#define THC_NONCONTIG_REDUCE_BLOCK_SIZE 32 * 16
#define CHUNKPERBLOCK 256

template <typename IndexType>
__device__ __forceinline__ IndexType getReduceNoncontigDimSliceIndex() {
  // Each thread handles one slice
  return getLinearBlockId<IndexType>() * THC_NONCONTIG_REDUCE_BLOCK_SIZE + threadIdx.x;
}

// quick hack to enable two-stage use of reduceChunk
template <typename T>
struct SimpleCopyOp
{
  __device__ __forceinline__ T operator()(volatile const T val) const volatile
  {
    return val;
  }
};

__device__ __forceinline__ int lastpow2(int n)
{
  int out = 1 << (31 - __clz(n));
  if(n == out)
    out >>= 1;
  return out;
}

template
  <typename T,
   typename U,
   typename IndexType,
   typename AccT,
   typename ModifyOp,
   typename ReduceOp,
   typename FinalizeOp>
__device__ __forceinline__ void reduceChunk
  (T* out,
   U* in,
   const int& inbounds,
   const IndexType& reductionStride,
   const IndexType& reductionSize,
   const IndexType& inOffset,
   const IndexType& outOffset,
   const int& shmem_lim,
   AccT init,
   AccT* shmem,
   ModifyOp modifyOp,
   ReduceOp reduceOp,
   FinalizeOp finalizeOp)
{
  AccT load_reg[4];
  AccT local_reg = init;

  //Unroll this loop
  //for(IndexType i=threadIdx.y; i<reductionSize; i+=blockDim.y){
  //  local_reg += in[inOffset + i*reductionStride];
  //}
  if(inbounds)
    for(IndexType i = threadIdx.y; i < reductionSize; i += blockDim.y*4)
    {
      if (i + blockDim.y*3 < reductionSize)
      {
        const AccT val0 = scalar_cast<AccT>(in[inOffset + i*reductionStride]);
        load_reg[0] = modifyOp(val0);
        const AccT val1 = scalar_cast<AccT>(in[inOffset + (i + blockDim.y)*reductionStride]);
        load_reg[1] = modifyOp(val1);
        const AccT val2 = scalar_cast<AccT>(in[inOffset + (i + blockDim.y*2)*reductionStride]);
        load_reg[2] = modifyOp(val2);
        const AccT val3 = scalar_cast<AccT>(in[inOffset + (i + blockDim.y*3)*reductionStride]);
        load_reg[3] = modifyOp(val3);
        local_reg = reduceOp(local_reg, load_reg[0]);
        local_reg = reduceOp(local_reg, load_reg[1]);
        local_reg = reduceOp(local_reg, load_reg[2]);
        local_reg = reduceOp(local_reg, load_reg[3]);
      }
      else if (i + blockDim.y*2 < reductionSize)
      {
        const AccT val0 = scalar_cast<AccT>(in[inOffset + i*reductionStride]);
        load_reg[0] = modifyOp(val0);
        const AccT val1 = scalar_cast<AccT>(in[inOffset + (i + blockDim.y)*reductionStride]);
        load_reg[1] = modifyOp(val1);
        const AccT val2 = scalar_cast<AccT>(in[inOffset + (i + blockDim.y*2)*reductionStride]);
        load_reg[2] = modifyOp(val2);
        local_reg = reduceOp(local_reg, load_reg[0]);
        local_reg = reduceOp(local_reg, load_reg[1]);
        local_reg = reduceOp(local_reg, load_reg[2]);
      }
      else if (i + blockDim.y < reductionSize)
      {
        const AccT val0 = scalar_cast<AccT>(in[inOffset + i*reductionStride]);
        load_reg[0] = modifyOp(val0);
        const AccT val1 = scalar_cast<AccT>(in[inOffset + (i + blockDim.y)*reductionStride]);
        load_reg[1] = modifyOp(val1);
        local_reg = reduceOp(local_reg, load_reg[0]);
        local_reg = reduceOp(local_reg, load_reg[1]);
      }
      else if (i < reductionSize)
      {
        const AccT val0 = scalar_cast<AccT>(in[inOffset + i*reductionStride]);
        local_reg = reduceOp(local_reg, modifyOp(val0));
      }
    }

  *shmem = local_reg;
  for(int i = lastpow2(shmem_lim); i > 0; i >>= 1)
  {
    __syncthreads();
    if(threadIdx.y < i && threadIdx.y + i < shmem_lim)
       *shmem = reduceOp(*shmem, *(shmem + i*blockDim.x));
  }

  if(threadIdx.y == 0 && inbounds) {
    T &&o_ele = static_cast<T>(finalizeOp(*shmem));
    out[outOffset] = o_ele;
  }
}

// Kernel that handles an entire reduction of a slice of a tensor per each thread
template
  <typename T,
   typename IndexType,
   typename AccT,
   typename ModifyOp,
   typename ReduceOp,
   typename FinalizeOp,
   int ADims, int BDims>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__ void kernelReduceNoncontigDim_shared
  (TensorInfo<T, IndexType> out,
   TensorInfo<T, IndexType> in,
   IndexType reductionStride,
   IndexType reductionSize,
   IndexType totalSlices,
   AccT init,
   ModifyOp modifyOp,
   ReduceOp reduceOp,
   FinalizeOp finalizeOp,
   volatile AccT* stagingData,
   int* semaphores)
{
  IndexType sliceIndex  = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ int isLastBlockDone;
  __shared__ AccT local_reduce[THC_NONCONTIG_REDUCE_BLOCK_SIZE];
  AccT* shmem = &local_reduce[threadIdx.x + threadIdx.y*blockDim.x];

  // This kernel is intended for the latency-bound case, so we want to launch enough blocks
  // to cover the entire output.  This means we don't need grid-stride loops.
  const IndexType outOffset =
    IndexToOffset<T, IndexType, ADims>::get(sliceIndex, out);
  const IndexType inOffset =
    IndexToOffset<T, IndexType, BDims>::get(sliceIndex, in);
  const int inbounds = (sliceIndex < totalSlices);

  if(gridDim.y == 1)
    reduceChunk
      (out.data,
       in.data,
       inbounds,
       reductionStride,
       reductionSize,
       inOffset,
       outOffset,
       reductionSize < blockDim.y ? reductionSize : blockDim.y,
       init,
       shmem,
       modifyOp,
       reduceOp,
       finalizeOp);
  else
  {
    int* semaphore = semaphores + blockIdx.x;

    const IndexType chunkStart = blockIdx.y*CHUNKPERBLOCK;
    const IndexType chunkSize = reductionSize - chunkStart < CHUNKPERBLOCK ?
                                reductionSize - chunkStart : CHUNKPERBLOCK;
    const IndexType reductionStrideStaging = totalSlices;
    const IndexType stagingOffset = sliceIndex;

    reduceChunk
      (stagingData,
       in.data,
       inbounds,
       reductionStride,
       chunkSize,
       inOffset + chunkStart*reductionStride,
       stagingOffset + blockIdx.y*reductionStrideStaging,
       chunkSize < blockDim.y ? chunkSize : blockDim.y,
       init,
       shmem,
       modifyOp,
       reduceOp,
       SimpleCopyOp<AccT>());

    __threadfence(); // make sure writes are globally visible
    __syncthreads(); // if multiple warps in this block wrote to staging, make sure they're all done

    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
      int old = atomicAdd(semaphore, 1);
      isLastBlockDone = (old == gridDim.y - 1);
    }

    __syncthreads();

    // The staging area contains gridDim.y elements along each slice.  The final reduction
    // begins by treating the first blockDim.y elements as "init" values.
    if(isLastBlockDone)
    {
      if(threadIdx.y < gridDim.y)
        init = stagingData[stagingOffset + threadIdx.y*reductionStrideStaging];
      IndexType remaining = gridDim.y < blockDim.y ? 0 : gridDim.y - blockDim.y;
      reduceChunk
        (out.data,
         stagingData,
         inbounds,
         reductionStrideStaging,
         remaining, // if 0, loop in reduceChunk is skipped, otherwise...
         stagingOffset + blockDim.y*reductionStrideStaging, // ...loop begins at blockDim+1th element
         outOffset,
         gridDim.y < blockDim.y ? gridDim.y : blockDim.y,
         init,
         shmem,
         SimpleCopyOp<AccT>(),
         reduceOp,
         finalizeOp);
    }
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
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(512, 4)
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
template <typename ScalarType,
typename TensorType,
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
  ptrdiff_t inElements = THCTensor_nElement(state, in);

  int64_t reductionSize = THTensor_sizeLegacyNoScalars(in, dim);
  int64_t reductionStride = THTensor_strideLegacyNoScalars(in, dim);
  ptrdiff_t outElements = inElements / reductionSize;

  if (THCTensor_nDimensionLegacyAll(state, out) > MAX_CUTORCH_DIMS ||
      THCTensor_nDimensionLegacyAll(state, in) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (THCTensor_nDimensionLegacyAll(state, in) == 0) {
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

    if(outElements <= 4096)
    {
      // gridDim.x and blockDim.x parallelize work across slices.
      // blockDim.y enables some intra-block reduction within slices.
      // gridDim.y enables inter-block reduction within slices.

      // Each block covers 32 output elements.
      int blockdimx = 32;
      int griddimx = THCCeilDiv((int64_t)outElements, (int64_t)blockdimx);

      // Each warp reduces at most 4 slices.  This heuristic can be tuned,
      // but locking blockdimy to 16 is robust and reasonably performant.
      int blockdimy = 16;

      int griddimy = 1;
      bool coop = false;
      // Rough heuristics to decide if using cooperating blocks is worthwhile
      if(                      outElements <=   32 && reductionSize >= 4096) coop = true;
      if(  32 < outElements && outElements <=   64 && reductionSize >= 4096) coop = true;
      if(  64 < outElements && outElements <=  128 && reductionSize >= 4096) coop = true;
      if( 128 < outElements && outElements <=  256 && reductionSize >= 4096) coop = true;
      if( 256 < outElements && outElements <=  512 && reductionSize >= 4096) coop = true;
      if( 512 < outElements && outElements <= 1024 && reductionSize >= 4096) coop = true;
      if(1024 < outElements && outElements <= 2048 && reductionSize >= 2048) coop = true;
      if(2048 < outElements && outElements <= 4096 && reductionSize >= 2048) coop = true;
      // Each block reduces at most CHUNKPERBLOCK (currently 256) slices.
      if(coop)
        griddimy = THCCeilDiv((int64_t)reductionSize, (int64_t)CHUNKPERBLOCK);

      grid = dim3(griddimx, griddimy, 1);
      block = dim3(blockdimx, blockdimy, 1);
    }
  }

  // Resize out to correspond to the reduced size with keepdim=True.

  // Preserve noncontiguities by unsqueezing out if necessary
  THCTensor_preserveReduceDimSemantics(
      state, out, THCTensor_nDimensionLegacyAll(state, in), dim, keepdim);

  // Resize out
  std::vector<int64_t> sizes = THTensor_sizesLegacyNoScalars(in);
  sizes[dim] = 1;
  THCTensor_resize(state, out, sizes, {});

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
    kernelReduceContigDim<ScalarType,                                   \
                          TYPE, AccT, ModifyOp, ReduceOp, FinalizeOp,   \
                          OUT, IN>                                      \
      <<<grid, block, smemSize, c10::cuda::getCurrentCUDAStream()>>>    \
        (outInfo, inInfo, reductionSize,                                \
        (TYPE) outElements, init, modifyOp, reduceOp, finalizeOp);      \
  } else {                                                              \
    if(block.y == 1){                                                   \
        kernelReduceNoncontigDim<                                       \
                          ScalarType,                                   \
                          TYPE, AccT, ModifyOp, ReduceOp, FinalizeOp,   \
                          OUT, IN>                                      \
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>         \
        (outInfo, inInfo, reductionStride, reductionSize,               \
        (TYPE) outElements, init, modifyOp, reduceOp, finalizeOp);      \
    }                                                                   \
    else                                                                \
    {                                                                   \
        void* stagingData = nullptr;                                    \
        void* semaphores = nullptr;                                     \
                                                                             \
        if(grid.y > 1)                                                       \
        {                                                                    \
          stagingData = THCudaMalloc(state, sizeof(AccT)*outElements*grid.y);\
          semaphores = THCudaMalloc(state, sizeof(int)*grid.x);              \
          THCudaCheck(cudaMemsetAsync                                        \
            (semaphores,                                                     \
             0,                                                              \
             sizeof(int)*grid.x,                                             \
             c10::cuda::getCurrentCUDAStream()));                             \
        }                                                                    \
                                                                             \
        kernelReduceNoncontigDim_shared                                      \
          <ScalarType, TYPE, AccT, ModifyOp, ReduceOp, FinalizeOp,  OUT, IN> \
          <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>             \
          (outInfo,                                                          \
           inInfo,                                                           \
           reductionStride,                                                  \
           reductionSize,                                                    \
           (TYPE) outElements,                                               \
           init,                                                             \
           modifyOp,                                                         \
           reduceOp,                                                         \
           finalizeOp,                                                       \
           (volatile AccT*)stagingData,                                      \
           (int*)semaphores);                                                \
                                                                             \
        if(grid.y > 1)                                                       \
        {                                                                    \
          THCudaFree(state, stagingData);                                    \
          THCudaFree(state, semaphores);                                     \
        }                                                                    \
    }                                                                        \
  }

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

  if(THCTensor_canUse32BitIndexMath(state, out) &&
     THCTensor_canUse32BitIndexMath(state, in))
  {
    TensorInfo<ScalarType,
               unsigned int> outInfo =
      getTensorInfo<ScalarType, TensorType, unsigned int>(state, out);
    outInfo.collapseDims();

    TensorInfo<ScalarType,
               unsigned int> inInfo =
      getTensorInfo<ScalarType, TensorType, unsigned int>(state, in);
    inInfo.reduceDim(dim);
    inInfo.collapseDims();
    HANDLE_OUT_CASE(unsigned int, outInfo.dims, inInfo.dims);
  }
  else
  {
    TensorInfo<ScalarType,
               uint64_t> outInfo =
      getTensorInfo<ScalarType, TensorType, uint64_t>(state, out);
    outInfo.collapseDims();

    TensorInfo<ScalarType,
               uint64_t> inInfo =
      getTensorInfo<ScalarType, TensorType, uint64_t>(state, in);
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
    THCTensor_squeeze1d(state, out, out, dim);
  }
  return true;
}

#undef THC_NONCONTIG_REDUCE_BLOCK_SIZE
#undef CHUNKPERBLOCK

#endif // THC_REDUCE_INC
