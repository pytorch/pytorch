#pragma once

#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/core/TensorBase.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/macros/Macros.h>
#include <ATen/native/Copy.h>

#include <math.h>

//
// This file contains pointwise operation functions and kernels that
// work on both contiguous and non-contiguous tensor arguments of
// arbitrary (up to MAX_CUTORCH_DIMS) dimensioned arguments without
// copying or temporary storage.
//

/*
  NOTE [ CUDA_tensor_applyN helpers ]

  The following CUDA_tensor_applyN (where N currently can be 1, 2, 3, or 4)
  functions apply a pointwise operator to N tensor(s).

  The calling convention is

  1. The template arguments should be, sequentially,
    - First N typename args specify the scalar types of each of the N tensors.
    - (Optional) `int step` arg specifies the number of elements processed
      together at the same time.
      Default is 1.
    - A usually omitted (i.e., inferred) typename arg specifies the type of the
      function/functor applied on `N * step` values  in each iteration of each
      CUDA thread.
  2. The arguments should be, sequentially,
    - N tensors
    - op: a function/functor that processes `N * step` values at the same time.
      - If `step == 1`, it must have signature
        `void(*)(scalar1_t&, scalar2_t&, ..., scalarN_t&)`, where
        `scalar*_t`s are the first N typename template args, and the inputs
        are the `N` values from the `N` tensors retrieved at a common index.
      - Otherwise, it must must have signature
          void(*)(int n, scalar1_t&, scalar1_t&, ..., scalar1_t&,  // repeat `step` times
                         scalar2_t&, scalar2_t&, ..., scalar2_t&,  // repeat `step` times
                         ...,
                         scalarN_t&, scalarN_t&, ..., scalarN_t&)  // repeat `step` times
        Different from `step == 1` case, it processes `N * step` values taken
        from `step` common indices. Moreover, the first input `n` represents the
        number of valid indices (it will always have `0 < n <= step`). It will
        almost always be `step`, but at the boundary we may not have full `step`
        elements and `n` can be a lesser value.

        E.g., if `step == 4` and `N == 2`, `op` could be

          [](int n, scalar1_t &u1, scalar1_t &u2, scalar1_t &u3, scalar1_t &u4,
                    scalar2_t &v1, scalar2_t &v2, scalar2_t &v3, scalar2_t &v4) {
            // Only process u1, ..., un and v1, ..., vn.
            // So if `n == 3`, `u4` and `v4` need not to be considered.
          }

      In both cases, the references can actually be const, but at least one of
      them should be non-const in order to write the output.
    - (Optional, but recommended) N TensorArgType args that specify for each
      tensor whether `op` reads AND writes ] (i.e., TensorArgType::ReadWrite),
      or only reads (i.e., TensorArgType::ReadOnly).
      Default is TensorArgType::ReadWrite for first Tensor, and
                 TensorArgType::ReadOnly  for the rest.

  E.g.,

  to compute a = b^2 for a and b of same dtype, we can call

  CUDA_tensor_apply2<scalar, scalar>(
    a, b,
    [] __device__ (scalar &a_val, const scalar &b_val) { a_val = b_val * b_val; }
  );

  to work on 2 values at the same time, we can call

  CUDA_tensor_apply2<scalar1, scalar2, 2>(
    a, b,
    [] __device__ (int n, scalar1 &a_val1, scalar1 &a_val2,
                          const scalar2 &b_val1, const scalar2 &b_val2) {
      // call special vectorized op here, or just do elementwise and enjoy unrolling...
      // if n == 1, only process a_val1 and b_val1
    }
  );
*/

namespace at::cuda {

// TODO: combine with TensorArg?  So far that's been for debugging, and this is functional...
enum class TensorArgType { ReadWrite, ReadOnly };

namespace {

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
// In general, given M (<=4) TensorInfo's with N dimensions, we can view each
// strides[i] (0 <= i < N) as an M-tuple.  Given each pair i < j, we exchange
// strides[i] and [j] if
//    (1) strides[i][k] < strides[j][k] for some k (0 <= k < M)
//        (exchanging them will benefit input #k), and
//    (2) strides[i][k] <= strieds[j][k] for all k
//        (exchanging them will not make any input worse).
template <typename T1, typename IndexType,
          typename T2 = void, typename T3 = void, typename T4 = void>
inline void rearrangeDims(detail::TensorInfo<T1, IndexType>* aInfo,
                          detail::TensorInfo<T2, IndexType>* bInfo = nullptr,
                          detail::TensorInfo<T3, IndexType>* cInfo = nullptr,
                          detail::TensorInfo<T4, IndexType>* dInfo = nullptr) {
  int numInfos = 1;
  int dims = aInfo->dims;
  IndexType *sizes[4] = { aInfo->sizes, };
  IndexType *strides[4] = { aInfo->strides, };

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

  if (dInfo != nullptr) {
    ++numInfos;
    if (dInfo->dims != dims) return;
    sizes[3] = dInfo->sizes;
    strides[3] = dInfo->strides;
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

// The `remaining_steps` argument is used to support Op that operates on
// multiple elements at the same time. Generally, the strategy of ApplyOpN is to
//  1. Initialize `remaining_steps = step`, where `step` is the template arg of
//     CUDA_tensor_applyN helpers. The input arg `n` to `apply()` represents the
//     number of elements in bound for this call. It will almost always equal to
//     `step` except at boundaries.
//  2. If `remaining_steps > 0` convert the current linearIndex to offset (if in
//     bound), and recursively call `ApplyOpN` with `remaining_steps - 1`.
//  3. At `remaining_steps = 0`,
//       if `step = 1`, call `op(tensor1_val, tensor2_val, ...)`;
//       if `step > 1`, call `op(n, tensor1_val1, tensor1_val2, ..., tesor1_valstep,
//                                  tensor2_val1, tensor2_val2, ..., tesor2_valstep,
//                                       ...
//                                  tensorN_val1, tensorN_val2, ..., tesorN_valstep);`
//
// See NOTE [ CUDA_tensor_applyN helpers ] above for how Op may look like.

template <typename Op,
          typename scalar,
          typename IndexType,
          int ADims,
          int remaining_steps,
          typename... Offsets>
struct ApplyOp1 {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar, IndexType> &a, const Op &op, int n,
                  IndexType linearIndex, Offsets... aOffsets) {
  // Convert `linearIndex` into an offset of `a`
  const IndexType aOffset = sizeof...(Offsets) < n ?
    detail::IndexToOffset<scalar, IndexType, ADims>::get(linearIndex, a) : 0;

  ApplyOp1<Op, scalar, IndexType, ADims, remaining_steps - 1, const IndexType, Offsets...>::apply(
    a, op, n, linearIndex + 1, aOffsets..., aOffset
  );
}
};

// Specialize `step=1` case (i.e., `remaining_steps=0` and `len(Offsets)=1`).
// We don't need to pass in how many elements need to processed in this case.
template <typename Op,
          typename scalar,
          typename IndexType,
          int ADims,
          typename Offset>
struct ApplyOp1<Op, scalar, IndexType, ADims, 0, Offset> {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar, IndexType> &a, const Op &op,
                  int n, IndexType linearIndex, Offset offset) {
  op(a.data[offset]);
}
};

template <typename Op,
          typename scalar,
          typename IndexType,
          int ADims,
          typename... Offsets>
struct ApplyOp1<Op, scalar, IndexType, ADims, 0, Offsets...> {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar, IndexType> &a, const Op &op, int n,
                 IndexType linearIndex, Offsets... offsets) {
  op(n, a.data[offsets]...);
}
};

template <typename Op,
          typename scalar,
          typename IndexType,
          int ADims,
          int step>
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(AT_APPLY_THREADS_PER_BLOCK, AT_APPLY_BLOCKS_PER_SM)
#endif
__global__ void kernelPointwiseApply1(detail::TensorInfo<scalar, IndexType> a,
                                      IndexType totalElements, const Op op) {
  for (IndexType linearIndex = (blockIdx.x * blockDim.x + threadIdx.x) * step;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x * step) {
    ApplyOp1<Op, scalar, IndexType, ADims, step>::apply(
      a, op, ::min(step, static_cast<int>(totalElements - linearIndex)), linearIndex);
  }
}


template <typename Op,
          typename scalar1,
          typename scalar2,
          typename IndexType,
          int ADims,
          int BDims,
          int remaining_steps,
          typename... Offsets>
struct ApplyOp2 {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar1, IndexType> &a,
                  detail::TensorInfo<scalar2, IndexType> &b,
                  const Op &op, int64_t n, IndexType linearIndex,
                  Offsets... aOffsets, Offsets... bOffsets) {
  // Convert `linearIndex` into an offset of `a`
  const IndexType aOffset = static_cast<int64_t>(sizeof...(Offsets)) < n ?
    detail::IndexToOffset<scalar1, IndexType, ADims>::get(linearIndex, a) : 0;

  // Convert `linearIndex` into an offset of `b`
  const IndexType bOffset = static_cast<int64_t>(sizeof...(Offsets)) < n ?
    detail::IndexToOffset<scalar2, IndexType, BDims>::get(linearIndex, b) : 0;

  ApplyOp2<Op, scalar1, scalar2, IndexType, ADims, BDims, remaining_steps - 1, const IndexType, Offsets...>::apply(
    a, b, op, n, linearIndex + 1, aOffsets..., aOffset, bOffsets..., bOffset
  );
}
};

// Specialize `step=1` case (i.e., `remaining_steps=0` and `len(Offsets)=1`).
// We don't need to pass in how many elements need to processed in this case.
template <typename Op,
          typename scalar1,
          typename scalar2,
          typename IndexType,
          int ADims,
          int BDims,
          typename Offset>
struct ApplyOp2<Op, scalar1, scalar2, IndexType, ADims, BDims, 0, Offset> {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar1, IndexType> &a,
                  detail::TensorInfo<scalar2, IndexType> &b,
                  const Op &op, int /*n*/, IndexType /*linearIndex*/,
                  Offset aOffset, Offset bOffset) {
  op(a.data[aOffset], b.data[bOffset]);
}
};

template <typename Op,
          typename scalar1,
          typename scalar2,
          typename IndexType,
          int ADims,
          int BDims,
          typename... Offsets>
struct ApplyOp2<Op, scalar1, scalar2, IndexType, ADims, BDims, 0, Offsets...> {
__device__ __forceinline__
static void apply(detail::TensorInfo<scalar1, IndexType> &a,
                  detail::TensorInfo<scalar2, IndexType> &b,
                  const Op &op, int n, IndexType linearIndex,
                  Offsets... aOffsets, Offsets... bOffsets) {
  op(n, a.data[aOffsets]..., b.data[bOffsets]...);
}
};

template <typename Op,
          typename scalar1,
          typename scalar2,
          typename IndexType,
          int ADims, int BDims,
          int step,
          int max_threads_per_block=AT_APPLY_THREADS_PER_BLOCK,
          int min_blocks_per_sm=AT_APPLY_BLOCKS_PER_SM>
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(max_threads_per_block, min_blocks_per_sm)
#endif
__global__ void
kernelPointwiseApply2(detail::TensorInfo<scalar1, IndexType> a,
                      detail::TensorInfo<scalar2, IndexType> b,
                      IndexType totalElements,
                      const Op op) {
  for (IndexType linearIndex = (blockIdx.x * blockDim.x + threadIdx.x) * step;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x * step) {
    ApplyOp2<Op, scalar1, scalar2, IndexType, ADims, BDims, step>::apply(
      a, b, op, ::min(step, static_cast<int>(totalElements - linearIndex)),
      linearIndex);
  }
}

} // anonymous namespace

template <typename scalar1, typename scalar2, int step, typename Op,
          int max_threads_per_block=AT_APPLY_THREADS_PER_BLOCK,
          int min_blocks_per_sm=AT_APPLY_BLOCKS_PER_SM>
inline bool CUDA_tensor_apply2(at::TensorBase a,
                               at::TensorBase b,
                               const Op op,
                               TensorArgType aType = TensorArgType::ReadWrite,
                               TensorArgType bType = TensorArgType::ReadOnly) {
  TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda(),
              "CUDA_tensor_apply2: Expected tensors to have CUDA DeviceType, but got "
              "tensors with type ", a.device().type(), " and ", b.device().type());
  int64_t totalElements = a.numel();

  if (totalElements != b.numel()) {
    return false;
  }

  if (a.dim() > MAX_TENSORINFO_DIMS ||
      b.dim() > MAX_TENSORINFO_DIMS) {
    return false;
  }

  if (a.numel() == 0) {
    // Empty tensor; do nothing
    return true;
  }
  const dim3 block = getApplyBlock(max_threads_per_block);

  dim3 grid;
  auto curDevice = current_device();
  if (curDevice == -1) return false;
  if (!getApplyGrid<step>(totalElements, grid, curDevice, max_threads_per_block)) {
    return false;
  }

  /*
  Expands readable/writable tensors whose indices may be "overlapped."
  This ensures that each element of the tensor is operated on once and only
  once.
  */
  TensorBase oldA;
  TensorBase oldB;

  if (aType == TensorArgType::ReadWrite && detail::maybeOverlappingIndices(a)) {
    // Must perform in contiguous space
    oldA = std::exchange(a, a.contiguous());
  }
  if (bType == TensorArgType::ReadWrite && detail::maybeOverlappingIndices(b)) {
    // Must perform in contiguous space
    oldB = std::exchange(b, b.contiguous());
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.

#define HANDLE_CASE(TYPE, A, B)                                        \
  kernelPointwiseApply2<Op,                                            \
                        scalar1,                                       \
                        scalar2,                                       \
                        TYPE, A, B, step,                              \
                        max_threads_per_block,                         \
                        min_blocks_per_sm>                             \
   <<<grid, block, 0, at::cuda::getCurrentCUDAStream(curDevice)>>>(    \
       aInfo, bInfo, static_cast<TYPE>(totalElements), op);            \
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

  if (detail::canUse32BitIndexMath(a) &&
      detail::canUse32BitIndexMath(b)) {
    detail::TensorInfo<scalar1, unsigned int> aInfo =
      detail::getTensorInfo<scalar1, unsigned int>(a);

    detail::TensorInfo<scalar2, unsigned int> bInfo =
      detail::getTensorInfo<scalar2, unsigned int>(b);
    rearrangeDims(&aInfo, &bInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims);
  } else {
    detail::TensorInfo<scalar1, uint64_t> aInfo =
      detail::getTensorInfo<scalar1, uint64_t>(a);

    detail::TensorInfo<scalar2, uint64_t> bInfo =
      detail::getTensorInfo<scalar2, uint64_t>(b);
    rearrangeDims(&aInfo, &bInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();

    /*
    Only instantiates the all 1D special case and the fallback all nD case for
    large (64-bit indexed) tensors to reduce compilation time.
    */
    if (aInfo.dims == 1 && bInfo.dims == 1) {
      HANDLE_CASE(uint64_t, 1, 1);
    } else {
      HANDLE_CASE(uint64_t, -1, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA.defined()) {
    at::native::copy_ignoring_overlaps(oldA, a);
  }

  if (oldB.defined()) {
    at::native::copy_ignoring_overlaps(oldB, b);
  }

  return true;
}

/* Provides default step = 1 to CUDA_tensor_apply2. */
template <typename scalar1, typename scalar2, typename Op,
          int max_threads_per_block=AT_APPLY_THREADS_PER_BLOCK,
          int min_blocks_per_sm=AT_APPLY_BLOCKS_PER_SM>
inline bool CUDA_tensor_apply2(const at::TensorBase &a,
                               const at::TensorBase &b,
                               const Op op,
                               TensorArgType aType = TensorArgType::ReadWrite,
                               TensorArgType bType = TensorArgType::ReadOnly) {
  return CUDA_tensor_apply2<scalar1, scalar2, 1, Op,
                            max_threads_per_block, min_blocks_per_sm>(a, b, op, aType, bType);
}

} // namespace at::cuda
