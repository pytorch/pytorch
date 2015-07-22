#ifndef THC_APPLY_INC
#define THC_APPLY_INC

#include "THCTensorCopy.h"
#include "THCReduceApplyUtils.cuh"

//
// This file contains pointwise operation functions and kernels that
// work on both contiguous and non-contiguous tensor arguments of
// arbitrary (up to MAX_CUTORCH_DIMS) dimensioned arguments without
// copying or temporary storage.
//

// Threads per block for our apply kernel
#define THC_APPLY_THREADS_PER_BLOCK 32 * 16

// Called when we are copying into an overlapping index `dst`, but
// we don't care which writer wins. Hacky but it works.
THC_API void THCudaTensor_copyIgnoringOverlaps(THCState* state,
                                       THCudaTensor* dst,
                                       THCudaTensor* src);

template <typename Op, typename IndexType, int ADims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
THCudaTensor_pointwiseApply1(TensorInfo<IndexType> a,
                             IndexType totalElements,
                             Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

    op(&a.data[aOffset]);
  }
}

template <typename Op, typename IndexType, int ADims, int BDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
THCudaTensor_pointwiseApply2(TensorInfo<IndexType> a,
                             TensorInfo<IndexType> b,
                             IndexType totalElements,
                             Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset =
      IndexToOffset<IndexType, BDims>::get(linearIndex, b);

    op(&a.data[aOffset], &b.data[bOffset]);
  }
}

template <typename Op, typename IndexType, int ADims, int BDims, int CDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
THCudaTensor_pointwiseApply3(TensorInfo<IndexType> a,
                             TensorInfo<IndexType> b,
                             TensorInfo<IndexType> c,
                             IndexType totalElements,
                             Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset =
      IndexToOffset<IndexType, BDims>::get(linearIndex, b);

    // Convert `linearIndex` into an offset of `c`
    const IndexType cOffset =
      IndexToOffset<IndexType, CDims>::get(linearIndex, c);

    op(&a.data[aOffset], &b.data[bOffset], &c.data[cOffset]);
  }
}

inline dim3 getApplyBlock() {
  return dim3(THC_APPLY_THREADS_PER_BLOCK);
}

inline bool getApplyGrid(THCState* state, long totalElements, dim3& grid) {
  int curDevice = -1;
  cudaGetDevice(&curDevice);

  if (curDevice == -1) {
    return false;
  }

  // Assume a reasonable number of SMs if no state is available
  int numSM =
    state ? THCState_getCurrentDeviceProperties(state)->multiProcessorCount : 15;

  // 16 warps per block * 4 per SM gives 64 warps per SM at maximum,
  // which seems to be a good sweetspot for latency hiding
  grid = dim3(min((long long) THCCeilDiv(totalElements,
                                         (long) THC_APPLY_THREADS_PER_BLOCK),
                  4LL * numSM));
  return true;
}

template <typename Op>
bool THCudaTensor_pointwiseApply1(THCState* state,
                                  THCudaTensor* a,
                                  const Op& op,
                                  TensorArgType aType = ReadWrite) {
  long totalElements = THCudaTensor_nElement(state, a);

  if (THCudaTensor_nDimension(state, a) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (THCudaTensor_nDimension(state, a) == 0) {
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
  THCudaTensor* oldA = NULL;

  if (aType == ReadWrite && THC_overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = THCudaTensor_newContiguous(state, a);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A)                                   \
  THCudaTensor_pointwiseApply1<Op, TYPE, A>                    \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(    \
      aInfo, (TYPE) totalElements, op);

#define HANDLE_A_CASE(TYPE, A)                      \
  {                                                 \
    if (aInfo.isContiguous()) {                     \
      HANDLE_CASE(TYPE, -2);                        \
    } else {                                        \
      switch (A) {                                  \
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

  // Can we use 32-bit integer math in the kernel (the linear ID for the copy
  // and the resulting non-linear offset is all computable using 32-bit math?)
  // We also use unsigned index math in the kernel, as signed div/mod has
  // additional overhead.
  if (THC_canUse32BitIndexMath(state, a)) {
    TensorInfo<unsigned int> aInfo(state, a);

    HANDLE_A_CASE(unsigned int, aInfo.dims);
  } else {
    TensorInfo<unsigned long> aInfo(state, a);

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous()) {
      THCudaTensor_pointwiseApply1<Op, unsigned long, -2>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, (unsigned long) totalElements, op);
    } else {
      THCudaTensor_pointwiseApply1<Op, unsigned long, -1>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, (unsigned long) totalElements, op);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldA, a);
    THCudaTensor_free(state, a);
    a = oldA;
  }

  return true;
}

template <typename Op>
bool THCudaTensor_pointwiseApply2(THCState* state,
                                  THCudaTensor* a,
                                  THCudaTensor* b,
                                  const Op& op,
                                  TensorArgType aType = ReadWrite,
                                  TensorArgType bType = ReadOnly) {
  long totalElements = THCudaTensor_nElement(state, a);

  if (totalElements != THCudaTensor_nElement(state, b)) {
    return false;
  }

  if (THCudaTensor_nDimension(state, a) > MAX_CUTORCH_DIMS ||
      THCudaTensor_nDimension(state, b) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (THCudaTensor_nDimension(state, a) == 0) {
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
  THCudaTensor* oldA = NULL;
  THCudaTensor* oldB = NULL;

  if (aType == ReadWrite && THC_overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = THCudaTensor_newContiguous(state, a);
  }
  if (bType == ReadWrite && THC_overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = THCudaTensor_newContiguous(state, b);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A, B)                                \
  THCudaTensor_pointwiseApply2<Op, TYPE, A, B>                 \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(    \
      aInfo, bInfo, (TYPE) totalElements, op);

#define HANDLE_B_CASE(TYPE, A, B)                   \
  {                                                 \
    if (bInfo.isContiguous()) {                     \
      HANDLE_CASE(TYPE, A, -2);                     \
    } else {                                        \
      switch (B) {                                  \
        case 1:                                     \
          HANDLE_CASE(TYPE, A, 1);                  \
          break;                                    \
        case 2:                                     \
          HANDLE_CASE(TYPE, A, 2);                  \
          break;                                    \
        case 3:                                     \
          HANDLE_CASE(TYPE, A, 3);                  \
          break;                                    \
        default:                                    \
          HANDLE_CASE(TYPE, A, -1);                 \
          break;                                    \
      }                                             \
    }                                               \
  }

#define HANDLE_A_CASE(TYPE, A, B)                   \
  {                                                 \
    if (aInfo.isContiguous()) {                     \
      HANDLE_B_CASE(TYPE, -2, B);                   \
    } else {                                        \
      switch (A) {                                  \
        case 1:                                     \
          HANDLE_B_CASE(TYPE, 1, B);                \
          break;                                    \
        case 2:                                     \
          HANDLE_B_CASE(TYPE, 2, B);                \
          break;                                    \
        case 3:                                     \
          HANDLE_B_CASE(TYPE, 3, B);                \
          break;                                    \
        default:                                    \
          HANDLE_B_CASE(TYPE, -1, B);               \
          break;                                    \
      }                                             \
    }                                               \
  }

  if (THC_canUse32BitIndexMath(state, a) &&
      THC_canUse32BitIndexMath(state, b)) {
    TensorInfo<unsigned int> aInfo(state, a);
    TensorInfo<unsigned int> bInfo(state, b);

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims);
  } else {
    TensorInfo<unsigned long> aInfo(state, a);
    TensorInfo<unsigned long> bInfo(state, b);

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous()) {
      THCudaTensor_pointwiseApply2<Op, unsigned long, -2, -2>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, bInfo, (unsigned long) totalElements, op);
    } else {
      THCudaTensor_pointwiseApply2<Op, unsigned long, -1, -1>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, bInfo, (unsigned long) totalElements, op);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldA, a);
    THCudaTensor_free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldB, b);
    THCudaTensor_free(state, b);
    b = oldB;
  }

  return true;
}

template <typename Op>
bool THCudaTensor_pointwiseApply3(THCState* state,
                                  THCudaTensor* a,
                                  THCudaTensor* b,
                                  THCudaTensor* c,
                                  const Op& op,
                                  TensorArgType aType = ReadWrite,
                                  TensorArgType bType = ReadOnly,
                                  TensorArgType cType = ReadOnly) {
  long totalElements = THCudaTensor_nElement(state, a);

  if (totalElements != THCudaTensor_nElement(state, b) ||
      totalElements != THCudaTensor_nElement(state, c)) {
    return false;
  }

  if (THCudaTensor_nDimension(state, a) > MAX_CUTORCH_DIMS ||
      THCudaTensor_nDimension(state, b) > MAX_CUTORCH_DIMS ||
      THCudaTensor_nDimension(state, c) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (THCudaTensor_nDimension(state, a) == 0) {
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
  THCudaTensor* oldA = NULL;
  THCudaTensor* oldB = NULL;
  THCudaTensor* oldC = NULL;

  if (aType == ReadWrite && THC_overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = THCudaTensor_newContiguous(state, a);
  }

  if (bType == ReadWrite && THC_overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = THCudaTensor_newContiguous(state, b);
  }

  if (cType == ReadWrite && THC_overlappingIndices(state, c)) {
    // Must perform in contiguous space
    oldC = c;
    c = THCudaTensor_newContiguous(state, c);
  }

#define HANDLE_CASE(TYPE, A, B, C)                                      \
  THCudaTensor_pointwiseApply3<Op, TYPE, A, B, C>                       \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(             \
      aInfo, bInfo, cInfo, (TYPE) totalElements, op);

#define HANDLE_C_CASE(TYPE, A, B, C)             \
  {                                              \
    if (cInfo.isContiguous()) {                  \
      HANDLE_CASE(TYPE, A, B, -2);               \
    } else {                                     \
      switch (C) {                               \
        case 1:                                  \
          HANDLE_CASE(TYPE, A, B, 1);            \
          break;                                 \
        case 2:                                  \
          HANDLE_CASE(TYPE, A, B, 2);            \
          break;                                 \
        case 3:                                  \
          HANDLE_CASE(TYPE, A, B, 3);            \
          break;                                 \
        default:                                 \
          HANDLE_CASE(TYPE, A, B, -1);           \
          break;                                 \
      }                                          \
    }                                            \
  }

#define HANDLE_B_CASE(TYPE, A, B, C)                 \
  {                                                  \
    if (bInfo.isContiguous()) {                      \
      HANDLE_C_CASE(TYPE, A, -2, C);                 \
    } else {                                         \
      switch (B) {                                   \
        case 1:                                      \
          HANDLE_C_CASE(TYPE, A, 1, C);              \
          break;                                     \
        case 2:                                      \
          HANDLE_C_CASE(TYPE, A, 2, C);              \
          break;                                     \
        case 3:                                      \
          HANDLE_C_CASE(TYPE, A, 3, C);              \
          break;                                     \
        default:                                     \
          HANDLE_C_CASE(TYPE, A, -1, C);             \
          break;                                     \
      }                                              \
    }                                                \
  }

#define HANDLE_A_CASE(TYPE, A, B, C)                 \
  {                                                  \
    if (aInfo.isContiguous()) {                      \
      HANDLE_B_CASE(TYPE, -2, B, C);                 \
    } else {                                         \
      switch (A) {                                   \
        case 1:                                      \
          HANDLE_B_CASE(TYPE, 1, B, C);              \
          break;                                     \
        case 2:                                      \
          HANDLE_B_CASE(TYPE, 2, B, C);              \
          break;                                     \
        case 3:                                      \
          HANDLE_B_CASE(TYPE, 3, B, C);              \
          break;                                     \
        default:                                     \
          HANDLE_B_CASE(TYPE, -1, B, C);             \
          break;                                     \
      }                                              \
    }                                                \
  }

  if (THC_canUse32BitIndexMath(state, a) &&
      THC_canUse32BitIndexMath(state, b) &&
      THC_canUse32BitIndexMath(state, c)) {
    TensorInfo<unsigned int> aInfo(state, a);
    TensorInfo<unsigned int> bInfo(state, b);
    TensorInfo<unsigned int> cInfo(state, c);

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims, cInfo.dims);
  } else {
    TensorInfo<unsigned long> aInfo(state, a);
    TensorInfo<unsigned long> bInfo(state, b);
    TensorInfo<unsigned long> cInfo(state, c);

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous() && cInfo.isContiguous()) {
      THCudaTensor_pointwiseApply3<Op, unsigned long, -2, -2, -2>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, bInfo, cInfo, (unsigned long) totalElements, op);
    } else {
      THCudaTensor_pointwiseApply3<Op, unsigned long, -1, -1, -1>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          aInfo, bInfo, cInfo, (unsigned long) totalElements, op);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_C_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldA, a);
    THCudaTensor_free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldB, b);
    THCudaTensor_free(state, b);
    b = oldB;
  }

  if (oldC) {
    // Ignore overlaps when copying back; if we use THCudaTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldC contiguous.
    THCudaTensor_copyIgnoringOverlaps(state, oldC, c);
    THCudaTensor_free(state, c);
    c = oldC;
  }

  return true;
}

#undef THC_APPLY_THREADS_PER_BLOCK

#endif // THC_APPLY_INC
