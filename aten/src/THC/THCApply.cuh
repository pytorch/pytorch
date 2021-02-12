#ifndef THC_APPLY_INC
#define THC_APPLY_INC

#include <THC/THCTensorCopy.h>
#include <THC/THCReduceApplyUtils.cuh>
#include <THC/THCTensorTypeUtils.cuh>
#include <THC/THCTensorCopy.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

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
#if defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(THC_APPLY_THREADS_PER_BLOCK, THC_APPLY_BLOCKS_PER_SM)
#endif
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
#if defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(THC_APPLY_THREADS_PER_BLOCK, THC_APPLY_BLOCKS_PER_SM)
#endif
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
#if defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(THC_APPLY_THREADS_PER_BLOCK, THC_APPLY_BLOCKS_PER_SM)
#endif
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

inline bool getApplyGrid(THCState* state, uint64_t totalElements, dim3& grid, int curDevice) {
  if (curDevice == -1) return false;

  uint64_t numBlocks = THCCeilDiv(totalElements, static_cast<uint64_t>(THC_APPLY_THREADS_PER_BLOCK));
  uint64_t maxGridX = at::cuda::getDeviceProperties(curDevice)->maxGridSize[0];
  if (numBlocks > maxGridX)
      numBlocks = maxGridX;

  // For 32-bit indices, make sure that gridDim.x * blockDim.x fits in 32 bits.
  if (totalElements <= INT32_MAX &&
      numBlocks > INT32_MAX / THC_APPLY_THREADS_PER_BLOCK)
    numBlocks = INT32_MAX / THC_APPLY_THREADS_PER_BLOCK;

  grid = dim3(numBlocks);
  return true;
}

template <typename ScalarTypeA,
          typename TensorTypeA,
          typename Op>
bool THC_pointwiseApply1(THCState* state,
                         TensorTypeA* a,
                         const Op& op,
                         TensorArgType aType = ReadWrite) {
  if (THCTensor_nDimensionLegacyAll(state, a) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (THCTensor_nDimensionLegacyAll(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  ptrdiff_t totalElements = THCTensor_nElement(state, a);

  int curDevice = -1;
  cudaGetDevice(&curDevice);
  if (!getApplyGrid(state, totalElements, grid, curDevice)) {
    return false;
  }

  /*
  Expands readable/writable tensors whose indices may be "overlapped."
  This ensures that each element of the tensor is operated on once and only
  once.
  */
  TensorTypeA* oldA = NULL;

  if (aType == ReadWrite &&
      THCTensor_maybeOverlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = (TensorTypeA*)THCTensor_newContiguous<ScalarTypeA>(state, a);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A)                                              \
  kernelPointwiseApply1<Op, ScalarTypeA, TYPE, A>                         \
    <<<grid, block, 0, c10::cuda::getCurrentCUDAStream(curDevice)>>>(     \
      OffsetInfo<ScalarTypeA, TYPE, A>(aInfo), (TYPE) totalElements, op); \
    C10_CUDA_KERNEL_LAUNCH_CHECK();

#define HANDLE_A_CASE(TYPE, A) {            \
  switch (A) {                              \
    case 1:                                 \
      HANDLE_CASE(TYPE, 1);                 \
      break;                                \
    case 2:                                 \
      HANDLE_CASE(TYPE, 2);                 \
      break;                                \
    default:                                \
      HANDLE_CASE(TYPE, -1);                \
      break;                                \
  }                                         \
}

  // Can we use 32-bit integer math in the kernel (the linear ID for the copy
  // and the resulting non-linear offset is all computable using 32-bit math?)
  // We also use unsigned index math in the kernel, as signed div/mod has
  // additional overhead.
  if (THCTensor_canUse32BitIndexMath(state, a)) {
    TensorInfo<ScalarTypeA, unsigned int> aInfo =
      getTensorInfo<ScalarTypeA, TensorTypeA, unsigned int>(state, a);
    rearrangeDims(&aInfo);
    aInfo.collapseDims();
#if CUDA_VERSION < 9000
    if (!aInfo.isContiguous()) {
        grid.x = min(at::cuda::getCurrentDeviceProperties()->multiProcessorCount * THC_APPLY_BLOCKS_PER_SM , grid.x);
    }
#endif
    HANDLE_A_CASE(unsigned int, aInfo.dims);
  } else {
    TensorInfo<ScalarTypeA, uint64_t> aInfo =
      getTensorInfo<ScalarTypeA, TensorTypeA, uint64_t>(state, a);
    rearrangeDims(&aInfo);
    aInfo.collapseDims();

    /*
    Only instantiates the all 1D special case and the fallback all nD case for
    large (64-bit indexed) tensors to reduce compilation time.
    */
    if (aInfo.dims == 1) {
      OffsetInfo<ScalarTypeA, uint64_t, 1>
        aOffset(aInfo);
      kernelPointwiseApply1<Op,
                            ScalarTypeA,
                            uint64_t, 1>
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
          aOffset, (uint64_t) totalElements, op);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {

#if CUDA_VERSION < 9000
        grid.x = min(at::cuda::getCurrentDeviceProperties()->multiProcessorCount * THC_APPLY_BLOCKS_PER_SM , grid.x);
#endif
      OffsetInfo<ScalarTypeA, uint64_t, -1>
        aOffset(aInfo);
      kernelPointwiseApply1<Op,
                            ScalarTypeA,
                            uint64_t, -1>
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
          aOffset, (uint64_t) totalElements, op);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
#undef HANDLE_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THCTensor_copyIgnoringOverlaps<ScalarTypeA>(state, oldA, a);
    THCTensor_free(state, a);
    a = oldA;
  }

  return true;
}

template <typename ScalarTypeA,
          typename ScalarTypeB,
          typename TensorTypeA,
          typename TensorTypeB,
          typename Op>
bool THC_pointwiseApply2(THCState* state,
                         TensorTypeA* a,
                         TensorTypeB* b,
                         const Op& op,
                         TensorArgType aType = ReadWrite,
                         TensorArgType bType = ReadOnly) {
  ptrdiff_t totalElements = THCTensor_nElement(state, a);
  if (totalElements != THCTensor_nElement(state, b)) {
    return false;
  }

  if (THCTensor_nDimensionLegacyAll(state, a) > MAX_CUTORCH_DIMS ||
      THCTensor_nDimensionLegacyAll(state, b) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (THCTensor_nDimensionLegacyAll(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  int curDevice = -1;
  cudaGetDevice(&curDevice);
  if (!getApplyGrid(state, totalElements, grid, curDevice)) {
    return false;
  }

  /*
  Expands readable/writable tensors whose indices may be "overlapped."
  This ensures that each element of the tensor is operated on once and only
  once.
  */
  TensorTypeA* oldA = NULL;
  TensorTypeB* oldB = NULL;

  if (aType == ReadWrite &&
      THCTensor_maybeOverlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = (TensorTypeA*)THCTensor_newContiguous<ScalarTypeA>(state, a);
  }
  if (bType == ReadWrite &&
      THCTensor_maybeOverlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = (TensorTypeB*)THCTensor_newContiguous<ScalarTypeB>(state, b);
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
  kernelPointwiseApply2<Op, ScalarTypeA, ScalarTypeB, TYPE, A, B>       \
    <<<grid, block, 0, c10::cuda::getCurrentCUDAStream(curDevice)>>>(   \
      OffsetInfo<ScalarTypeA, TYPE, A>(aInfo),                          \
      OffsetInfo<ScalarTypeB, TYPE, B>(bInfo),                          \
      (TYPE) totalElements, op);                                        \
  C10_CUDA_KERNEL_LAUNCH_CHECK();


#define HANDLE_B_CASE(TYPE, A, B) {         \
  switch (B) {                              \
    case 1:                                 \
      HANDLE_CASE(TYPE, A, 1);              \
      break;                                \
    case 2:                                 \
      HANDLE_CASE(TYPE, A, 2);              \
      break;                                \
    default:                                \
      HANDLE_CASE(TYPE, A, -1);             \
      break;                                \
  }                                         \
}

#define HANDLE_A_CASE(TYPE, A, B) {         \
  switch (A) {                              \
    case 1:                                 \
      HANDLE_B_CASE(TYPE, 1, B);            \
      break;                                \
    case 2:                                 \
      HANDLE_B_CASE(TYPE, 2, B);            \
      break;                                \
    default:                                \
      HANDLE_B_CASE(TYPE, -1, B);           \
      break;                                \
  }                                         \
}

  if (THCTensor_canUse32BitIndexMath(state, a) &&
      THCTensor_canUse32BitIndexMath(state, b)) {
    TensorInfo<ScalarTypeA, unsigned int> aInfo =
      getTensorInfo<ScalarTypeA, TensorTypeA, unsigned int>(state, a);

    TensorInfo<ScalarTypeB, unsigned int> bInfo =
      getTensorInfo<ScalarTypeB, TensorTypeB, unsigned int>(state, b);

    rearrangeDims(&aInfo, &bInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();
#if CUDA_VERSION < 9000
    if (!(aInfo.isContiguous() && bInfo.isContiguous()))
        grid.x = min(at::cuda::getCurrentDeviceProperties()->multiProcessorCount * THC_APPLY_BLOCKS_PER_SM , grid.x);
#endif

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims);
  } else {
    TensorInfo<ScalarTypeA, uint64_t> aInfo =
      getTensorInfo<ScalarTypeA, TensorTypeA, uint64_t>(state, a);

    TensorInfo<ScalarTypeB, uint64_t> bInfo =
      getTensorInfo<ScalarTypeB, TensorTypeB, uint64_t>(state, b);

    rearrangeDims(&aInfo, &bInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();

    /*
    Only instantiates the all 1D special case and the fallback all nD case for
    large (64-bit indexed) tensors to reduce compilation time.
    */
    if (aInfo.dims == 1 && bInfo.dims == 1) {
      OffsetInfo<ScalarTypeA, uint64_t, 1>
        aOffset(aInfo);
      OffsetInfo<ScalarTypeB, uint64_t, 1>
        bOffset(bInfo);
      kernelPointwiseApply2<Op,
                            ScalarTypeA,
                            ScalarTypeB,
                            uint64_t, 1, 1>
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
          aOffset, bOffset, (uint64_t) totalElements, op);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
#if CUDA_VERSION < 9000
      grid.x = min(at::cuda::getCurrentDeviceProperties()->multiProcessorCount * THC_APPLY_BLOCKS_PER_SM , grid.x);
#endif
      OffsetInfo<ScalarTypeA, uint64_t, -1>
        aOffset(aInfo);
      OffsetInfo<ScalarTypeB, uint64_t, -1>
        bOffset(bInfo);
      kernelPointwiseApply2<Op,
                            ScalarTypeA,
                            ScalarTypeB,
                            uint64_t, -1, -1>
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
          aOffset, bOffset, (uint64_t) totalElements, op);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
#undef HANDLE_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THCTensor_copyIgnoringOverlaps<ScalarTypeA>(state, oldA, a);
    THCTensor_free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    THCTensor_copyIgnoringOverlaps<ScalarTypeB>(state, oldB, b);
    THCTensor_free(state, b);
    b = oldB;
  }

  return true;
}

template <typename ScalarTypeA,
          typename ScalarTypeB,
          typename ScalarTypeC,
          typename TensorTypeA,
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
  ptrdiff_t totalElements = THCTensor_nElement(state, a);

  if (totalElements != THCTensor_nElement(state, b) ||
      totalElements != THCTensor_nElement(state, c)) {
    return false;
  }

  if (THCTensor_nDimensionLegacyAll(state, a) > MAX_CUTORCH_DIMS ||
      THCTensor_nDimensionLegacyAll(state, b) > MAX_CUTORCH_DIMS ||
      THCTensor_nDimensionLegacyAll(state, c) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (THCTensor_nDimensionLegacyAll(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock();

  dim3 grid;
  int curDevice = -1;
  cudaGetDevice(&curDevice);
  if (!getApplyGrid(state, totalElements, grid, curDevice)) {
    return false;
  }

  /*
  Expands readable/writable tensors whose indices may be "overlapped."
  This ensures that each element of the tensor is operated on once and only
  once.
  */
  TensorTypeA* oldA = NULL;
  TensorTypeB* oldB = NULL;
  TensorTypeC* oldC = NULL;

  if (aType == ReadWrite &&
      THCTensor_maybeOverlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = (TensorTypeA*)THCTensor_newContiguous<ScalarTypeA>(state, a);
  }
  if (bType == ReadWrite &&
      THCTensor_maybeOverlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = (TensorTypeB*)THCTensor_newContiguous<ScalarTypeB>(state, b);
  }
  if (cType == ReadWrite &&
      THCTensor_maybeOverlappingIndices(state, c)) {
    // Must perform in contiguous space
    oldC = c;
    c = (TensorTypeC*)THCTensor_newContiguous<ScalarTypeC>(state, c);
  }

#define HANDLE_CASE(TYPE, A, B, C)                                      \
  kernelPointwiseApply3<Op,                                             \
                        ScalarTypeA,                                    \
                        ScalarTypeB,                                    \
                        ScalarTypeC,                                    \
                        TYPE, A, B, C>                                  \
    <<<grid, block, 0, c10::cuda::getCurrentCUDAStream(curDevice)>>>(   \
      OffsetInfo<ScalarTypeA, TYPE, A>                                  \
          (aInfo),                                                      \
      OffsetInfo<ScalarTypeB, TYPE, B>                                  \
          (bInfo),                                                      \
      OffsetInfo<ScalarTypeC, TYPE, C>                                  \
          (cInfo),                                                      \
      (TYPE) totalElements, op);                                        \
      C10_CUDA_KERNEL_LAUNCH_CHECK();

#define HANDLE_C_CASE(TYPE, A, B, C) {      \
  switch (C) {                              \
    case 1:                                 \
      HANDLE_CASE(TYPE, A, B, 1);           \
      break;                                \
    case 2:                                 \
      HANDLE_CASE(TYPE, A, B, 2);           \
      break;                                \
    default:                                \
      HANDLE_CASE(TYPE, A, B, -1);          \
      break;                                \
  }                                         \
}

#define HANDLE_B_CASE(TYPE, A, B, C) {      \
  switch (B) {                              \
    case 1:                                 \
      HANDLE_C_CASE(TYPE, A, 1, C);         \
      break;                                \
    case 2:                                 \
      HANDLE_C_CASE(TYPE, A, 2, C);         \
      break;                                \
    default:                                \
      HANDLE_C_CASE(TYPE, A, -1, C);        \
      break;                                \
  }                                         \
}

#define HANDLE_A_CASE(TYPE, A, B, C) {      \
  switch (A) {                              \
    case 1:                                 \
      HANDLE_B_CASE(TYPE, 1, B, C);         \
      break;                                \
    case 2:                                 \
      HANDLE_B_CASE(TYPE, 2, B, C);         \
      break;                                \
    default:                                \
      HANDLE_B_CASE(TYPE, -1, B, C);        \
      break;                                \
  }                                         \
}

  if (THCTensor_canUse32BitIndexMath(state, a) &&
      THCTensor_canUse32BitIndexMath(state, b) &&
      THCTensor_canUse32BitIndexMath(state, c)) {
    TensorInfo<ScalarTypeA, unsigned int> aInfo =
      getTensorInfo<ScalarTypeA, TensorTypeA, unsigned int>(state, a);

    TensorInfo<ScalarTypeB, unsigned int> bInfo =
      getTensorInfo<ScalarTypeB, TensorTypeB, unsigned int>(state, b);

    TensorInfo<ScalarTypeC, unsigned int> cInfo =
      getTensorInfo<ScalarTypeC, TensorTypeC, unsigned int>(state, c);

    rearrangeDims(&aInfo, &bInfo, &cInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();
    cInfo.collapseDims();

#if CUDA_VERSION < 9000
      if (!(aInfo.isContiguous() && bInfo.isContiguous() && cInfo.isContiguous()))
          grid.x = min(at::cuda::getCurrentDeviceProperties()->multiProcessorCount * THC_APPLY_BLOCKS_PER_SM , grid.x);
#endif
    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims, cInfo.dims);
  } else {
    TensorInfo<ScalarTypeA, uint64_t> aInfo =
      getTensorInfo<ScalarTypeA, TensorTypeA, uint64_t>(state, a);

    TensorInfo<ScalarTypeB, uint64_t> bInfo =
      getTensorInfo<ScalarTypeB, TensorTypeB, uint64_t>(state, b);

    TensorInfo<ScalarTypeC, uint64_t> cInfo =
      getTensorInfo<ScalarTypeC, TensorTypeC, uint64_t>(state, c);

    rearrangeDims(&aInfo, &bInfo, &cInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();
    cInfo.collapseDims();

    /*
    Only instantiates the all 1D special case and the fallback all nD case for
    large (64-bit indexed) tensors to reduce compilation time.
    */
    if (aInfo.dims == 1 && bInfo.dims == 1 && cInfo.dims == 1) {
      OffsetInfo<ScalarTypeA, uint64_t, 1>
        aOffset(aInfo);
      OffsetInfo<ScalarTypeB, uint64_t, 1>
        bOffset(bInfo);
      OffsetInfo<ScalarTypeC, uint64_t, 1>
        cOffset(cInfo);
      kernelPointwiseApply3<Op,
                            ScalarTypeA,
                            ScalarTypeB,
                            ScalarTypeC,
                            uint64_t, 1, 1, 1>
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
          aOffset, bOffset, cOffset, (uint64_t) totalElements, op);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
#if CUDA_VERSION < 9000
      grid.x = min(at::cuda::getCurrentDeviceProperties()->multiProcessorCount * THC_APPLY_BLOCKS_PER_SM , grid.x);
#endif

      OffsetInfo<ScalarTypeA, uint64_t, -1>
        aOffset(aInfo);
      OffsetInfo<ScalarTypeB, uint64_t, -1>
        bOffset(bInfo);
      OffsetInfo<ScalarTypeC, uint64_t, -1>
        cOffset(cInfo);
      kernelPointwiseApply3<Op,
                            ScalarTypeA,
                            ScalarTypeB,
                            ScalarTypeC,
                            uint64_t, -1, -1, -1>
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
          aOffset, bOffset, cOffset, (uint64_t) totalElements, op);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
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
    THCTensor_copyIgnoringOverlaps<ScalarTypeA>(state, oldA, a);
    THCTensor_free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    THCTensor_copyIgnoringOverlaps<ScalarTypeB>(state, oldB, b);
    THCTensor_free(state, b);
    b = oldB;
  }

  if (oldC) {
    // Ignore overlaps when copying back; if we use THCTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldC contiguous.
    THCTensor_copyIgnoringOverlaps<ScalarTypeC>(state, oldC, c);
    THCTensor_free(state, c);
    c = oldC;
  }

  return true;
}

#undef THC_APPLY_THREADS_PER_BLOCK
#undef THC_APPLY_BLOCKS_PER_SM

#endif // THC_APPLY_INC
