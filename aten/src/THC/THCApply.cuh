#ifndef THC_APPLY_INC
#define THC_APPLY_INC

#include "THCTensorCopy.h"
#include "THCReduceApplyUtils.cuh"
#include "THCTensorTypeUtils.cuh"

//
// This file contains pointwise operation functions and kernels that
// work on both contiguous and non-contiguous tensor arguments of
// arbitrary (up to MAX_CUTORCH_DIMS) dimensioned arguments without
// copying or temporary storage.
//

// Rearrange dimensions for pointwise operations so that strides are in
// decreasing order as much as possible, so that kernels have better memory
// access patterns.
//
// For example, consider a binary operation on two "transposed" 2-dim tensors:
//    sizes:          256 512
//    aInfo->strides:   1 256
//    bInfo->strides:   1 256
//
// Given this, each concurrent memory access inside kernelPointwiseApply2() is
// exactly 256 elements apart, resulting in poor performance.
//
// This function exchanges dimensions so that memory access is contiguous:
//    sizes:          512 256
//    aInfo->strides: 256   1
//    bInfo->strides: 256   1
//
// (Actually, it becomes even better because now collapseDims() can turn each
// input into one contiguous array.)
//
// In general, given M (<=3) TensorInfo's with N dimensions, we can view each
// strides[i] (0 <= i < N) as an M-tuple.  Given each pair i < j, we exchange
// strides[i] and [j] if
//    (1) strides[i][k] < strides[j][k] for some k (0 <= k < M)
//        (exchanging them will benefit input #k), and
//    (2) strides[i][k] <= strieds[j][k] for all k
//        (exchanging them will not make any input worse).
template <typename T1, typename IndexType,
          typename T2 = void, typename T3 = void>
void rearrangeDims(TensorInfo<T1, IndexType>* aInfo,
                   TensorInfo<T2, IndexType>* bInfo = nullptr,
                   TensorInfo<T3, IndexType>* cInfo = nullptr) {
  int numInfos = 1;
  int dims = aInfo->dims;
  IndexType *sizes[3] = { aInfo->sizes, };
  IndexType *strides[3] = { aInfo->strides, };

  if (bInfo != nullptr) {
    ++numInfos;
    if (bInfo->dims != dims) return;
    sizes[1] = bInfo->sizes;
    strides[1] = bInfo->strides;
  }

  if (cInfo != nullptr) {
    ++numInfos;
    if (cInfo->dims != dims) return;
    sizes[2] = cInfo->sizes;
    strides[2] = cInfo->strides;
  }

  // Bail out if sizes do not match: we are using "deprecated pointwise
  // behavior" among tensors of different shapes but same number of elements.
  for (int i = 1; i < numInfos; ++i) {
    for (int j = 0; j < dims; ++j) {
      if (sizes[i][j] != sizes[0][j]) return;
    }
  }

  for (int i = 0; i < dims - 1; ++i) {
    // No need to consider dimensions of size 1.
    if (sizes[0][i] == 1) continue;

    for (int j = i + 1; j < dims; ++j) {
      if (sizes[0][j] == 1) continue;

      // Compare the relative sizes of strides between dim #i and dim #j.
      bool hasIncreasingStrides = false;
      bool hasDecreasingStrides = false;

      for (int k = 0; k < numInfos; k++) {
        IndexType stride_i = strides[k][i];
        IndexType stride_j = strides[k][j];
        if (stride_i < stride_j) {
          hasIncreasingStrides = true;
        } else if (stride_i > stride_j) {
          hasDecreasingStrides = true;
        }
      }

      if (hasIncreasingStrides && !hasDecreasingStrides) {
        for (int k = 0; k < numInfos; k++) {
          IndexType size = sizes[k][i];
          sizes[k][i] = sizes[k][j];
          sizes[k][j] = size;

          IndexType stride = strides[k][i];
          strides[k][i] = strides[k][j];
          strides[k][j] = stride;
        }
      }
    }
  }
}

// Threads per block for our apply kernel
// FIXME: use occupancy calculator instead
#define THC_APPLY_THREADS_PER_BLOCK (32 * 16)
#define THC_APPLY_BLOCKS_PER_SM 4
template <typename Op,
          typename Ta,
          typename IndexType,
          int ADims>
__global__ void
kernelPointwiseApply1(const OffsetInfo<Ta, IndexType, ADims> a,
                      IndexType totalElements,
                      Op op) {
  // NOTE: The two typecasts below are essential when IndexType is 64-bit;
  //       without them, results are silently truncated to 32 bits!
  for (IndexType linearIndex = (IndexType) blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += (IndexType) gridDim.x * blockDim.x) {
    op(a.get(linearIndex));
  }
}

template <typename Op,
          typename Ta, typename Tb,
          typename IndexType,
          int ADims, int BDims>
__global__ void
kernelPointwiseApply2(const OffsetInfo<Ta, IndexType, ADims> a,
                      const OffsetInfo<Tb, IndexType, BDims> b,
                      IndexType totalElements,
                      Op op) {
  for (IndexType linearIndex = (IndexType) blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += (IndexType) gridDim.x * blockDim.x) {
    op(a.get(linearIndex), b.get(linearIndex));
  }
}

template <typename Op,
          typename Ta, typename Tb, typename Tc,
          typename IndexType,
          int ADims, int BDims, int CDims>
__global__ void
kernelPointwiseApply3(const OffsetInfo<Ta, IndexType, ADims> a,
                      const OffsetInfo<Tb, IndexType, BDims> b,
                      const OffsetInfo<Tc, IndexType, CDims> c,
                      IndexType totalElements,
                      Op op) {
  for (IndexType linearIndex = (IndexType) blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += (IndexType) gridDim.x * blockDim.x) {
    op(a.get(linearIndex), b.get(linearIndex), c.get(linearIndex));
  }
}

inline dim3 getApplyBlock() {
  return dim3(THC_APPLY_THREADS_PER_BLOCK);
}

inline bool getApplyGrid(THCState* state, uint64_t totalElements, dim3& grid) {
  int curDevice = -1;
  cudaGetDevice(&curDevice);
  if (curDevice == -1) return false;

  uint64_t numBlocks = THCCeilDiv(totalElements, static_cast<uint64_t>(THC_APPLY_THREADS_PER_BLOCK));
  uint64_t maxGridX = THCState_getCurrentDeviceProperties(state)->maxGridSize[0];
  if (numBlocks > maxGridX)
      numBlocks = maxGridX;

  // For 32-bit indices, make sure that gridDim.x * blockDim.x fits in 32 bits.
  if (totalElements <= INT32_MAX &&
      numBlocks > INT32_MAX / THC_APPLY_THREADS_PER_BLOCK)
    numBlocks = INT32_MAX / THC_APPLY_THREADS_PER_BLOCK;

  grid = dim3(numBlocks);
  return true;
}

template <typename TensorTypeA,
          typename Op>
bool THC_pointwiseApply1(THCState* state,
                         TensorTypeA* a,
                         const Op& op,
                         TensorArgType aType = ReadWrite) {
  if (TensorUtils<TensorTypeA>::getDims(state, a) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  ptrdiff_t totalElements = TensorUtils<TensorTypeA>::getNumElements(state, a);

  if (!getApplyGrid(state, totalElements, grid)) {
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  TensorTypeA* oldA = NULL;

  if (aType == ReadWrite &&
      TensorUtils<TensorTypeA>::overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = TensorUtils<TensorTypeA>::newContiguous(state, a);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A)                                            \
  kernelPointwiseApply1<Op,                                             \
                        typename TensorUtils<TensorTypeA>::DataType,    \
                        TYPE, A>                                        \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(             \
      OffsetInfo<typename TensorUtils<TensorTypeA>::DataType, TYPE, A>  \
          (aInfo),                                                      \
      (TYPE) totalElements, op);

#define HANDLE_A_CASE(TYPE, A)                  \
  {                                             \
    if (aInfo.isContiguous()) {                 \
      HANDLE_CASE(TYPE, -2);                    \
    } else {                                    \
      switch (A) {                              \
        case 1:                                 \
        HANDLE_CASE(TYPE, 1);                   \
        break;                                  \
        case 2:                                 \
        HANDLE_CASE(TYPE, 2);                   \
        break;                                  \
        default:                                \
        HANDLE_CASE(TYPE, -1);                  \
        break;                                  \
      }                                         \
    }                                           \
  }

  // Can we use 32-bit integer math in the kernel (the linear ID for the copy
  // and the resulting non-linear offset is all computable using 32-bit math?)
  // We also use unsigned index math in the kernel, as signed div/mod has
  // additional overhead.
  if (TensorUtils<TensorTypeA>::canUse32BitIndexMath(state, a)) {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned int> aInfo =
      getTensorInfo<TensorTypeA, unsigned int>(state, a);
    rearrangeDims(&aInfo);
    aInfo.collapseDims();
#if CUDA_VERSION < 9000
    if (!aInfo.isContiguous())
        grid.x = min(THCState_getCurrentDeviceProperties(state)->multiProcessorCount * THC_APPLY_BLOCKS_PER_SM , grid.x);
#endif
    HANDLE_A_CASE(unsigned int, aInfo.dims);
  } else {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, uint64_t> aInfo =
      getTensorInfo<TensorTypeA, uint64_t>(state, a);
    rearrangeDims(&aInfo);
    aInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous()) {
      OffsetInfo<typename TensorUtils<TensorTypeA>::DataType, uint64_t, -2>
        aOffset(aInfo);
      kernelPointwiseApply1<Op,
                            typename TensorUtils<TensorTypeA>::DataType,
                            uint64_t, -2>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aOffset, (uint64_t) totalElements, op);
    } else {

#if CUDA_VERSION < 9000
        grid.x = min(THCState_getCurrentDeviceProperties(state)->multiProcessorCount * THC_APPLY_BLOCKS_PER_SM , grid.x);
#endif
      OffsetInfo<typename TensorUtils<TensorTypeA>::DataType, uint64_t, -1>
        aOffset(aInfo);
      kernelPointwiseApply1<Op,
                            typename TensorUtils<TensorTypeA>::DataType,
                            uint64_t, -1>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aOffset, (uint64_t) totalElements, op);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    TensorUtils<TensorTypeA>::copyIgnoringOverlaps(state, oldA, a);
    TensorUtils<TensorTypeA>::free(state, a);
    a = oldA;
  }

  return true;
}

template <typename TensorTypeA,
          typename TensorTypeB,
          typename Op>
bool THC_pointwiseApply2(THCState* state,
                         TensorTypeA* a,
                         TensorTypeB* b,
                         const Op& op,
                         TensorArgType aType = ReadWrite,
                         TensorArgType bType = ReadOnly) {
  ptrdiff_t totalElements = TensorUtils<TensorTypeA>::getNumElements(state, a);

  if (totalElements != TensorUtils<TensorTypeB>::getNumElements(state, b)) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) > MAX_CUTORCH_DIMS ||
      TensorUtils<TensorTypeB>::getDims(state, b) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  TensorTypeA* oldA = NULL;
  TensorTypeB* oldB = NULL;

  if (aType == ReadWrite &&
      TensorUtils<TensorTypeA>::overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = TensorUtils<TensorTypeA>::newContiguous(state, a);
  }
  if (bType == ReadWrite &&
      TensorUtils<TensorTypeB>::overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = TensorUtils<TensorTypeB>::newContiguous(state, b);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A, B)                                         \
  kernelPointwiseApply2<Op,                                             \
                        typename TensorUtils<TensorTypeA>::DataType,    \
                        typename TensorUtils<TensorTypeB>::DataType,    \
                        TYPE, A, B>                                     \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(             \
      OffsetInfo<typename TensorUtils<TensorTypeA>::DataType, TYPE, A>  \
          (aInfo),                                                      \
      OffsetInfo<typename TensorUtils<TensorTypeB>::DataType, TYPE, B>  \
          (bInfo),                                                      \
      (TYPE) totalElements, op);

#define HANDLE_B_CASE(TYPE, A, B)               \
  {                                             \
    if (bInfo.isContiguous()) {                 \
      HANDLE_CASE(TYPE, A, -2);                 \
    } else {                                    \
      switch (B) {                              \
        case 1:                                 \
        HANDLE_CASE(TYPE, A, 1);                \
        break;                                  \
        case 2:                                 \
        HANDLE_CASE(TYPE, A, 2);                \
        break;                                  \
        default:                                \
        HANDLE_CASE(TYPE, A, -1);               \
        break;                                  \
      }                                         \
    }                                           \
  }

#define HANDLE_A_CASE(TYPE, A, B)               \
  {                                             \
    if (aInfo.isContiguous()) {                 \
      HANDLE_B_CASE(TYPE, -2, B);               \
    } else {                                    \
      switch (A) {                              \
        case 1:                                 \
        HANDLE_B_CASE(TYPE, 1, B);              \
        break;                                  \
        case 2:                                 \
        HANDLE_B_CASE(TYPE, 2, B);              \
        break;                                  \
        default:                                \
        HANDLE_B_CASE(TYPE, -1, B);             \
        break;                                  \
      }                                         \
    }                                           \
  }

  if (TensorUtils<TensorTypeA>::canUse32BitIndexMath(state, a) &&
      TensorUtils<TensorTypeB>::canUse32BitIndexMath(state, b)) {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned int> aInfo =
      getTensorInfo<TensorTypeA, unsigned int>(state, a);

    TensorInfo<typename TensorUtils<TensorTypeB>::DataType, unsigned int> bInfo =
      getTensorInfo<TensorTypeB, unsigned int>(state, b);

    rearrangeDims(&aInfo, &bInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();
#if CUDA_VERSION < 9000
    if (!(aInfo.isContiguous() && bInfo.isContiguous()))
        grid.x = min(THCState_getCurrentDeviceProperties(state)->multiProcessorCount * THC_APPLY_BLOCKS_PER_SM , grid.x);
#endif

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims);
  } else {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, uint64_t> aInfo =
      getTensorInfo<TensorTypeA, uint64_t>(state, a);

    TensorInfo<typename TensorUtils<TensorTypeB>::DataType, uint64_t> bInfo =
      getTensorInfo<TensorTypeB, uint64_t>(state, b);

    rearrangeDims(&aInfo, &bInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous()) {
      OffsetInfo<typename TensorUtils<TensorTypeA>::DataType, uint64_t, -2>
        aOffset(aInfo);
      OffsetInfo<typename TensorUtils<TensorTypeB>::DataType, uint64_t, -2>
        bOffset(bInfo);
      kernelPointwiseApply2<Op,
                            typename TensorUtils<TensorTypeA>::DataType,
                            typename TensorUtils<TensorTypeB>::DataType,
                            uint64_t, -2, -2>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aOffset, bOffset, (uint64_t) totalElements, op);
    } else {
#if CUDA_VERSION < 9000
      grid.x = min(THCState_getCurrentDeviceProperties(state)->multiProcessorCount * THC_APPLY_BLOCKS_PER_SM , grid.x);
#endif
      OffsetInfo<typename TensorUtils<TensorTypeA>::DataType, uint64_t, -1>
        aOffset(aInfo);
      OffsetInfo<typename TensorUtils<TensorTypeB>::DataType, uint64_t, -1>
        bOffset(bInfo);
      kernelPointwiseApply2<Op,
                            typename TensorUtils<TensorTypeA>::DataType,
                            typename TensorUtils<TensorTypeB>::DataType,
                            uint64_t, -1, -1>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aOffset, bOffset, (uint64_t) totalElements, op);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    TensorUtils<TensorTypeA>::copyIgnoringOverlaps(state, oldA, a);
    TensorUtils<TensorTypeA>::free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    TensorUtils<TensorTypeB>::copyIgnoringOverlaps(state, oldB, b);
    TensorUtils<TensorTypeB>::free(state, b);
    b = oldB;
  }

  return true;
}

template <typename TensorTypeA,
          typename TensorTypeB,
          typename TensorTypeC,
          typename Op>
bool THC_pointwiseApply3(THCState* state,
                         TensorTypeA* a,
                         TensorTypeB* b,
                         TensorTypeC* c,
                         const Op& op,
                         TensorArgType aType = ReadWrite,
                         TensorArgType bType = ReadOnly,
                         TensorArgType cType = ReadOnly) {
  ptrdiff_t totalElements = TensorUtils<TensorTypeA>::getNumElements(state, a);

  if (totalElements != TensorUtils<TensorTypeB>::getNumElements(state, b) ||
      totalElements != TensorUtils<TensorTypeC>::getNumElements(state, c)) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) > MAX_CUTORCH_DIMS ||
      TensorUtils<TensorTypeB>::getDims(state, b) > MAX_CUTORCH_DIMS ||
      TensorUtils<TensorTypeC>::getDims(state, c) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (TensorUtils<TensorTypeA>::getDims(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  TensorTypeA* oldA = NULL;
  TensorTypeB* oldB = NULL;
  TensorTypeC* oldC = NULL;

  if (aType == ReadWrite &&
      TensorUtils<TensorTypeA>::overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = TensorUtils<TensorTypeA>::newContiguous(state, a);
  }
  if (bType == ReadWrite &&
      TensorUtils<TensorTypeB>::overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = TensorUtils<TensorTypeB>::newContiguous(state, b);
  }
  if (cType == ReadWrite &&
      TensorUtils<TensorTypeC>::overlappingIndices(state, c)) {
    // Must perform in contiguous space
    oldC = c;
    c = TensorUtils<TensorTypeC>::newContiguous(state, c);
  }

#define HANDLE_CASE(TYPE, A, B, C)                                      \
  kernelPointwiseApply3<Op,                                             \
                        typename TensorUtils<TensorTypeA>::DataType,    \
                        typename TensorUtils<TensorTypeB>::DataType,    \
                        typename TensorUtils<TensorTypeC>::DataType,    \
                        TYPE, A, B, C>                                  \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(             \
      OffsetInfo<typename TensorUtils<TensorTypeA>::DataType, TYPE, A>  \
          (aInfo),                                                      \
      OffsetInfo<typename TensorUtils<TensorTypeB>::DataType, TYPE, B>  \
          (bInfo),                                                      \
      OffsetInfo<typename TensorUtils<TensorTypeC>::DataType, TYPE, C>  \
          (cInfo),                                                      \
      (TYPE) totalElements, op);

#define HANDLE_C_CASE(TYPE, A, B, C)            \
  {                                             \
    if (cInfo.isContiguous()) {                 \
      HANDLE_CASE(TYPE, A, B, -2);              \
    } else {                                    \
      switch (C) {                              \
        case 1:                                 \
        HANDLE_CASE(TYPE, A, B, 1);             \
        break;                                  \
        case 2:                                 \
        HANDLE_CASE(TYPE, A, B, 2);             \
        break;                                  \
        default:                                \
        HANDLE_CASE(TYPE, A, B, -1);            \
        break;                                  \
      }                                         \
    }                                           \
  }

#define HANDLE_B_CASE(TYPE, A, B, C)            \
  {                                             \
    if (bInfo.isContiguous()) {                 \
      HANDLE_C_CASE(TYPE, A, -2, C);            \
    } else {                                    \
      switch (B) {                              \
        case 1:                                 \
        HANDLE_C_CASE(TYPE, A, 1, C);           \
        break;                                  \
        case 2:                                 \
        HANDLE_C_CASE(TYPE, A, 2, C);           \
        break;                                  \
        default:                                \
        HANDLE_C_CASE(TYPE, A, -1, C);          \
        break;                                  \
      }                                         \
    }                                           \
  }

#define HANDLE_A_CASE(TYPE, A, B, C)            \
  {                                             \
    if (aInfo.isContiguous()) {                 \
      HANDLE_B_CASE(TYPE, -2, B, C);            \
    } else {                                    \
      switch (A) {                              \
        case 1:                                 \
        HANDLE_B_CASE(TYPE, 1, B, C);           \
        break;                                  \
        case 2:                                 \
        HANDLE_B_CASE(TYPE, 2, B, C);           \
        break;                                  \
        default:                                \
        HANDLE_B_CASE(TYPE, -1, B, C);          \
        break;                                  \
      }                                         \
    }                                           \
  }

  if (TensorUtils<TensorTypeA>::canUse32BitIndexMath(state, a) &&
      TensorUtils<TensorTypeB>::canUse32BitIndexMath(state, b) &&
      TensorUtils<TensorTypeC>::canUse32BitIndexMath(state, c)) {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, unsigned int> aInfo =
      getTensorInfo<TensorTypeA, unsigned int>(state, a);

    TensorInfo<typename TensorUtils<TensorTypeB>::DataType, unsigned int> bInfo =
      getTensorInfo<TensorTypeB, unsigned int>(state, b);

    TensorInfo<typename TensorUtils<TensorTypeC>::DataType, unsigned int> cInfo =
      getTensorInfo<TensorTypeC, unsigned int>(state, c);

    rearrangeDims(&aInfo, &bInfo, &cInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();
    cInfo.collapseDims();

#if CUDA_VERSION < 9000
      if (!(aInfo.isContiguous() && bInfo.isContiguous() && cInfo.isContiguous()))
          grid.x = min(THCState_getCurrentDeviceProperties(state)->multiProcessorCount * THC_APPLY_BLOCKS_PER_SM , grid.x);
#endif
    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims, cInfo.dims);
  } else {
    TensorInfo<typename TensorUtils<TensorTypeA>::DataType, uint64_t> aInfo =
      getTensorInfo<TensorTypeA, uint64_t>(state, a);

    TensorInfo<typename TensorUtils<TensorTypeB>::DataType, uint64_t> bInfo =
      getTensorInfo<TensorTypeB, uint64_t>(state, b);

    TensorInfo<typename TensorUtils<TensorTypeC>::DataType, uint64_t> cInfo =
      getTensorInfo<TensorTypeC, uint64_t>(state, c);

    rearrangeDims(&aInfo, &bInfo, &cInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();
    cInfo.collapseDims();

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous() && cInfo.isContiguous()) {
      OffsetInfo<typename TensorUtils<TensorTypeA>::DataType, uint64_t, -2>
        aOffset(aInfo);
      OffsetInfo<typename TensorUtils<TensorTypeB>::DataType, uint64_t, -2>
        bOffset(bInfo);
      OffsetInfo<typename TensorUtils<TensorTypeC>::DataType, uint64_t, -2>
        cOffset(cInfo);
      kernelPointwiseApply3<Op,
                            typename TensorUtils<TensorTypeA>::DataType,
                            typename TensorUtils<TensorTypeB>::DataType,
                            typename TensorUtils<TensorTypeC>::DataType,
                            uint64_t, -2, -2, -2>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aOffset, bOffset, cOffset, (uint64_t) totalElements, op);
    } else {
#if CUDA_VERSION < 9000
      grid.x = min(THCState_getCurrentDeviceProperties(state)->multiProcessorCount * THC_APPLY_BLOCKS_PER_SM , grid.x);
#endif

      OffsetInfo<typename TensorUtils<TensorTypeA>::DataType, uint64_t, -1>
        aOffset(aInfo);
      OffsetInfo<typename TensorUtils<TensorTypeB>::DataType, uint64_t, -1>
        bOffset(bInfo);
      OffsetInfo<typename TensorUtils<TensorTypeC>::DataType, uint64_t, -1>
        cOffset(cInfo);
      kernelPointwiseApply3<Op,
                            typename TensorUtils<TensorTypeA>::DataType,
                            typename TensorUtils<TensorTypeB>::DataType,
                            typename TensorUtils<TensorTypeC>::DataType,
                            uint64_t, -1, -1, -1>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aOffset, bOffset, cOffset, (uint64_t) totalElements, op);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_C_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    TensorUtils<TensorTypeA>::copyIgnoringOverlaps(state, oldA, a);
    TensorUtils<TensorTypeA>::free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    TensorUtils<TensorTypeB>::copyIgnoringOverlaps(state, oldB, b);
    TensorUtils<TensorTypeB>::free(state, b);
    b = oldB;
  }

  if (oldC) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldC contiguous.
    TensorUtils<TensorTypeC>::copyIgnoringOverlaps(state, oldC, c);
    TensorUtils<TensorTypeC>::free(state, c);
    c = oldC;
  }

  return true;
}

#undef THC_APPLY_THREADS_PER_BLOCK
#undef THC_APPLY_BLOCKS_PER_SM

#endif // THC_APPLY_INC
