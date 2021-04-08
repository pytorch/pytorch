#include <ATen/ATen.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Sorting.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <c10/util/Exception.h>

#include <THC/THCGeneral.h>
#include <THC/THCTensorInfo.cuh>
#include <THC/THCSortUtils.cuh>

#include <iostream>

namespace at { namespace native {

namespace {

template <typename T, bool handleNaN = false>
struct LTComp {
  __device__ inline bool operator()(const T& a, const T& b) const {
    return (handleNaN && at::_isnan(b) && !at::_isnan(a)) || (a < b);
  }
};

template <typename T, bool handleNaN = false>
struct GTComp {
  __device__ inline bool operator()(const T& a, const T& b) const {
    return (handleNaN && at::_isnan(a) && !at::_isnan(b)) || (a > b);
  }
};

template <typename T, typename IndT, bool handleNaN = true>
struct ThrustSliceLTOp {
ThrustSliceLTOp(int64_t size) : sliceSize(size) {}
  __device__ bool operator()(const thrust::tuple<int64_t, T>& lhs, const thrust::tuple<int64_t, T>& rhs) const {
    IndT segA = (IndT)thrust::get<0>(lhs) / sliceSize;
    IndT segB = (IndT)thrust::get<0>(rhs) / sliceSize;
    if (segA != segB)
        return segA < segB;
    else
        return (handleNaN && at::_isnan(thrust::get<1>(rhs)) && !at::_isnan(thrust::get<1>(lhs))) || (thrust::get<1>(lhs) < thrust::get<1>(rhs));
  }
  const IndT sliceSize;
};


template <typename T, typename IndT, bool handleNaN = true>
struct ThrustSliceGTOp {
ThrustSliceGTOp(int64_t size) : sliceSize(size) {}
  __device__ bool operator()(const thrust::tuple<int64_t, T>& lhs, const thrust::tuple<int64_t, T>& rhs) const {
    IndT segA = (IndT)thrust::get<0>(lhs) / sliceSize;
    IndT segB = (IndT)thrust::get<0>(rhs) / sliceSize;
    if (segA != segB)
        return segA < segB;
    else
        return (handleNaN && at::_isnan(thrust::get<1>(lhs)) && !at::_isnan(thrust::get<1>(rhs))) || (thrust::get<1>(lhs) > thrust::get<1>(rhs));
  }
  const IndT sliceSize;
};

template <typename T>
__device__ inline void swapVars(T& t1, T& t2) {
  T tmp = t1;
  t1 = t2;
  t2 = tmp;
}

template <typename Comparator, typename K>
__device__ inline void bitonicSwapKeys(K& kA, bool& validA,
                                       K& kB, bool& validB,
                                       bool dir,
                                       const Comparator& comp) {
  bool swap = (comp(kA, kB) && validA) || !validB;
  if (swap == dir) {
    swapVars(kA, kB);
    swapVars(validA, validB);
  }
}

template <typename Comparator, typename K, typename V,
          typename IndexType, int Power2SortSize>
__device__ inline void bitonicSort(K keys[Power2SortSize],
                                   V values[Power2SortSize],
                                   bool valid[Power2SortSize],
                                   const Comparator& comp) {
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
  for (unsigned int size = 2; size < Power2SortSize; size *= 2) {
    bool flag = ((threadIdx.x & (size / 2)) != 0);

#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {

      __syncthreads();

      unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      bitonicSwap<Comparator, K, V>(
        keys[pos], values[pos], valid[pos],
        keys[pos + stride], values[pos + stride], valid[pos + stride],
        flag, comp);
    }
  }

#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
  for (unsigned int stride = Power2SortSize / 2; stride > 0; stride /= 2) {

    __syncthreads();

    unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    bitonicSwap<Comparator, K, V>(
      keys[pos], values[pos], valid[pos],
      keys[pos + stride], values[pos + stride], valid[pos + stride],
      false, comp);
  }

  __syncthreads();

}

// Sorts (key, value) pairs (in different tensors) in-place; i.e.,
// modifies the input `keys` and `values`
template <typename K, typename V,
          int KeyDims, int ValueDims,
          typename Comparator, typename IndexType, int Power2SortSize>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void
bitonicSortKVInPlace(cuda::detail::TensorInfo<K, IndexType> keys,
                     IndexType keySlices,
                     IndexType keySliceSize,
                     IndexType keySliceStride,
                     cuda::detail::TensorInfo<V, IndexType> values,
                     IndexType valueSliceStride,
                     Comparator comp) {
  // Find the slice of the tensor that we are sorting
  const IndexType linearIndex = getLinearBlockId<IndexType>();
  // Tiling the slices could have us be out of bounds, if there are a
  // lot of slices to sort
  if (linearIndex >= keySlices) {
    return;
  }

  __shared__ K sharedKeys[Power2SortSize];
  __shared__ V sharedValues[Power2SortSize];
  __shared__ bool sharedValid[Power2SortSize];

  const IndexType keyStartOffset =
    cuda::detail::IndexToOffset<K, IndexType, KeyDims>::get(linearIndex, keys);
  const IndexType valueStartOffset =
    cuda::detail::IndexToOffset<V, IndexType, ValueDims>::get(linearIndex, values);

  // If the sort size is 1, the data is already sorted
  if (Power2SortSize == 1) {
    return;
  } else {
    // Otherwise, each thread is responsible for loading and storing 2
    // elements. The sort size is guaranteed to be >= 2
    const int elem1 = threadIdx.x;
    const int elem2 = threadIdx.x + (Power2SortSize / 2);

    bool valid1 = (elem1 < keySliceSize);
    K k1 = valid1 ?
      keys.data[keyStartOffset + elem1 * keySliceStride] : static_cast<K>(0);
    V v1 = valid1 ?
      values.data[valueStartOffset + elem1 * valueSliceStride] : static_cast<V>(0);

    sharedKeys[elem1] = k1;
    sharedValues[elem1] = v1;
    sharedValid[elem1] = valid1;

    bool valid2 = (elem2 < keySliceSize);
    K k2 = valid2 ?
      keys.data[keyStartOffset + elem2 * keySliceStride] : static_cast<K>(0);
    V v2 = valid2 ?
      values.data[valueStartOffset + elem2 * valueSliceStride] : static_cast<V>(0);

    sharedKeys[elem2] = k2;
    sharedValues[elem2] = v2;
    sharedValid[elem2] = valid2;

    bitonicSort<Comparator, K, V, IndexType, Power2SortSize>(
      sharedKeys, sharedValues, sharedValid, comp);

    // elem1 and elem2 values might be out-of-range, if the data size we are
    // sorting is smaller than half the power2 size
    if (valid1) {
      keys.data[keyStartOffset + elem1 * keySliceStride] =
        sharedKeys[elem1];
      values.data[valueStartOffset + elem1 * valueSliceStride] =
        sharedValues[elem1];
    }

    if (valid2) {
      keys.data[keyStartOffset + elem2 * keySliceStride] =
        sharedKeys[elem2];
      values.data[valueStartOffset + elem2 * valueSliceStride] =
        sharedValues[elem2];
    }
  }
}

template <typename scalar_t>
void sortKeyValueInplace(Tensor& key,
                         Tensor& value,
                         int64_t dim,
                         bool dir) {
  TORCH_CHECK(key.sizes().equals(value.sizes()), "Key tensor must have same size as value tensor");
  TORCH_CHECK(value.dim() <= MAX_CUTORCH_DIMS, CUTORCH_DIM_WARNING);
  TORCH_CHECK(key.dim() <= MAX_CUTORCH_DIMS, CUTORCH_DIM_WARNING);

  int64_t inElements = key.numel();
  if (inElements == 0) {
    return;
  }

  int64_t keySliceSize = key.dim() == 0 ? 1 : key.size(dim);
  int64_t keySlices = inElements / keySliceSize;

  // The amount of shared memory and block size is based on
  // 2^ceil(lg(n)); we choose that sorting implementation for a given
  // size.
  int64_t ceilPowerOf2 = nextHighestPowerOf2(keySliceSize);

  // FIXME: We'd have to find some other trick with Thrust to perform a
  // vectorized (key, value) sort by slice segment
  TORCH_CHECK(ceilPowerOf2 <= 2048, "sortKeyValueInplace only works for sizes <= 2048 at present");

  // The grid is based on the number of independent slices that we
  // have to sort; one block per slice
  dim3 grid;
  if (!getGridFromTiles(keySlices, grid)) {
    AT_ERROR("Slice to sort is too large");
  }

#define HANDLE_CASE(TYPE, A, SIZE)                                      \
  do {                                                                  \
    int blockSize = SIZE / 2;                                           \
    if (blockSize < 1) {                                                \
      blockSize = 1;                                                    \
    }                                                                   \
                                                                        \
    dim3 block(blockSize);                                              \
                                                                        \
    if (dir) {                                                          \
      bitonicSortKVInPlace<scalar_t, int64_t, A, -1,                    \
                           GTComp<scalar_t, true>, TYPE, SIZE>          \
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(        \
          keyInfo,                                                      \
          static_cast<TYPE>(keySlices),                                 \
          static_cast<TYPE>(keySliceSize),                              \
          static_cast<TYPE>(keyInfo.strides[collapseKeyDim]),           \
          valueInfo,                                                    \
          static_cast<TYPE>(valueInfo.strides[collapseValueDim]),       \
          GTComp<scalar_t, true>());                                    \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                   \
    } else {                                                            \
      bitonicSortKVInPlace<scalar_t, int64_t, A, -1,                    \
                           LTComp<scalar_t, true>, TYPE, SIZE>          \
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(        \
          keyInfo,                                                      \
          static_cast<TYPE>(keySlices),                                 \
          static_cast<TYPE>(keySliceSize),                              \
          static_cast<TYPE>(keyInfo.strides[collapseKeyDim]),           \
          valueInfo,                                                    \
          static_cast<TYPE>(valueInfo.strides[collapseValueDim]),       \
          LTComp<scalar_t, true>());                                    \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                   \
    }                                                                   \
  } while (0)

#define HANDLE_SORT_CASE(TYPE, A)                       \
  {                                                     \
    switch (ceilPowerOf2) {                             \
      case 2048:                                        \
      HANDLE_CASE(TYPE, A, 2048);                       \
      break;                                            \
      case 1024: case 512: case 256:                    \
      HANDLE_CASE(TYPE, A, 1024);                       \
      break;                                            \
      case 128: case 64:                                \
      HANDLE_CASE(TYPE, A, 128);                        \
      break;                                            \
      case 32: case 16: case 8: case 4: case 2:         \
      HANDLE_CASE(TYPE, A, 32);                         \
      break;                                            \
      case 1:                                           \
      /* Nothing to do, data already sorted */          \
      break;                                            \
      default:                                          \
      TORCH_INTERNAL_ASSERT(false);                     \
    }                                                   \
  }

  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis
  if (cuda::detail::canUse32BitIndexMath(key)) {
    auto keyInfo = cuda::detail::getTensorInfo<scalar_t, unsigned int>(key);
    keyInfo.reduceDim(dim);
    int collapseKeyDim = keyInfo.collapseDims(dim);

    auto valueInfo = cuda::detail::getTensorInfo<int64_t, unsigned int>(value);
    valueInfo.reduceDim(dim);
    int collapseValueDim = valueInfo.collapseDims(dim);

    if (keyInfo.isContiguous()) {
      HANDLE_SORT_CASE(unsigned int, -2);
    } else {
      switch (keyInfo.dims) {
        case 2:
          HANDLE_SORT_CASE(unsigned int, 2);
          break;
        default:
          HANDLE_SORT_CASE(unsigned int, -1);
          break;
      }
    }
  } else {
    auto keyInfo = cuda::detail::getTensorInfo<scalar_t, uint64_t>(key);
    keyInfo.reduceDim(dim);
    int collapseKeyDim = keyInfo.collapseDims(dim);

    auto valueInfo = cuda::detail::getTensorInfo<int64_t, uint64_t>(value);
    valueInfo.reduceDim(dim);
    int collapseValueDim = valueInfo.collapseDims(dim);

    // int64_t case is rare, just instantiate the generic version
    HANDLE_SORT_CASE(uint64_t, -1);
  }
#undef HANDLE_CASE
#undef HANDLE_SORT_CASE
}

template <typename scalar_t>
void sortViaThrust(Tensor& sorted,
                   Tensor& indices,
                   int64_t dim,
                   bool dir) {
  int64_t nDims = sorted.dim();

  int64_t totalElements = sorted.numel();
  int64_t sliceSize = sorted.dim() == 0 ? 1 : sorted.size(dim);
  int64_t sliceStride = sorted.dim() == 0 ? 1 : sorted.stride(dim);

  // We perform a vectorized segmented sort in Thrust.
  // Say we are sorting a (2, 3) tensor. We have in flattened form:
  // values 0.4 1.2 5.3 6.2 1.3 2.3
  // indices  0   1   2   3   4   5
  // where indices is a global index (across all slices)

  // First we sort by values, globally:
  // values 6.2 5.3 2.3 1.2 1.3 0.4
  // indices  3   2   5   1   4   0

  // Then we stable sort by segment, which is index / 3:
  // values 5.3 1.2 0.4 6.2 2.3 1.3
  // indices  2   1   0   3   5   4

  // Then we translate the global index to a per-slice Lua index
  // (index % 3) + 1:
  // values 5.3 1.2 0.4 6.2 2.3 1.3
  // indices  3   2   1   1   3   2

  // This method can only work if the slice we are sorting (`dim`) is
  // innermost, and both values and indices are contiguous. We do this
  // by re-arranging the input into this form as needed, which will
  // unfortunately allocate memory if the request is not in this form.
  // Vectorized sort is slower than iterated sort if the number of
  // slices is small (since we're sorting twice, instead of invoking a
  // smaller sort `numSlices` times), but the Thrust sort
  // implementation here is a catch-all, so we're not looking for
  // efficiency, but instead correctness.
  Tensor trKeys = at::alias(sorted);
  Tensor trIndices = at::alias(indices);

  // Transpose dim to innermost
  if (dim != nDims - 1) {
    trKeys.transpose_(dim, nDims - 1);
    trIndices.transpose_(dim, nDims - 1);
  }

  // Thrust must operate on a contiguous layout
  Tensor trContigKey = trKeys.contiguous();
  Tensor trContigIndices = trIndices.contiguous();

  auto thrustAlloc = THCThrustAllocator(globalContext().lazyInitCUDA());

  thrust::device_ptr<scalar_t> keyIter(trContigKey.data_ptr<scalar_t>());

  // Since we are composing a global index across all segments rather
  // than a per-segment index, we treat the memory as int so we don't
  // have problems sorting slices < 2^24 but where the entire tensor
  // has more than 2^24 elements
  thrust::device_ptr<int64_t> indexIter(trContigIndices.data_ptr<int64_t>());

  // Fill the indices with a global index across all slices
  thrust::counting_iterator<int64_t> countIter(0);
  thrust::copy(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
    thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#endif
    countIter, countIter + totalElements, indexIter
  );

  auto begin = thrust::make_zip_iterator(thrust::make_tuple(indexIter, keyIter));

  auto _sort = [&](auto comp) {
    thrust::sort(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
       thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#endif
       begin, begin + totalElements, comp);
  };

  if (dir) {
    if (cuda::detail::canUse32BitIndexMath(trContigKey)) {
      _sort(ThrustSliceGTOp<scalar_t, int, true>(sliceSize));
    } else {
      _sort(ThrustSliceGTOp<scalar_t, int64_t, true>(sliceSize));
    }
  } else {
    if (cuda::detail::canUse32BitIndexMath(trContigKey)) {
      _sort(ThrustSliceLTOp<scalar_t, int, true>(sliceSize));
    } else {
      _sort(ThrustSliceLTOp<scalar_t, int64_t, true>(sliceSize));
    }
  }

  // Translate the global integer 0-based index to a per-slice real
  // Lua index
  thrust::for_each(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
    thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#endif
    indexIter, indexIter + totalElements,
    GlobalIndexToPerSliceIndex(sliceSize)
  );

  // Reverse the transposition as needed
  if (dim != nDims - 1) {
    trContigKey.transpose_(dim, nDims - 1);
    trContigIndices.transpose_(dim, nDims - 1);
  }

  // Then copy back to the expected output
  sorted.copy_(trContigKey);
  indices.copy_(trContigIndices);
}

} // namespace

void sort_kernel(Tensor& sorted,
                 Tensor& indices,
                 int64_t dim,
                 bool order,
                 bool stable) {
  TORCH_CHECK(!stable, "stable=True is not implemented on CUDA yet.");

  TensorArg sorted_arg{sorted, "sorted", 1};
  TensorArg indices_arg{indices, "indices", 2};

  checkAllSameGPU("sort_kernel", {sorted_arg, indices_arg});
  checkScalarType("sort_kernel", indices_arg, ScalarType::Long);

  TORCH_CHECK(sorted.dim() <= MAX_CUTORCH_DIMS, CUTORCH_DIM_WARNING);
  TORCH_CHECK(indices.dim() <= MAX_CUTORCH_DIMS, CUTORCH_DIM_WARNING);

  dim = at::maybe_wrap_dim(dim, sorted);

  // How large are the slices that we are sorting?
  int64_t sliceSize = sorted.dim() == 0 ? 1 : sorted.size(dim);

  // Workaround:
  // CUDA 8 uses more shared memory than 7.5 for bitonicSortKVInPlace,
  // and so for the double word types,
  // we get "too many resources requested for launch" in the 2048 case
#if CUDA_VERSION >= 8000
#if defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_LONG)
  int maxSliceSize = 1024;
#else
  int maxSliceSize = 2048;
#endif
#else
  int maxSliceSize = 2048;
#endif

  if (sliceSize <= maxSliceSize) {
    // Fill `indices` (the values) with the
    // slice-relative index.
    // THCudaLongTensor_fillSliceWithIndex(state, indices, dim);

    // We sort k/v pairs in-place; copy unsorted input to output
    // THCTensor_(copy)(state, sorted, input);

    // Sort using our in-place k/v kernel that supports arbitrary
    // layout
    AT_DISPATCH_ALL_TYPES_AND(
      ScalarType::Half, sorted.scalar_type(),
      "sortKeyValueInplace", [&]() {
          sortKeyValueInplace<scalar_t>(sorted, indices, dim, order);
    });
  } else {
    // Otherwise, fall back upon Thrust, which handles all other cases
    // (potentially slowly, with extra copies/memory allocations)
    AT_DISPATCH_ALL_TYPES_AND(
      ScalarType::Half, sorted.scalar_type(),
      "sortViaThrust", [&]() {
          sortViaThrust<scalar_t>(sorted, indices, dim, order);
    });
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

REGISTER_DISPATCH(sort_stub, &sort_kernel);

}} // at::native