#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/Sort.h>
#include <ATen/core/TensorBase.h>
#include <ATen/core/Array.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/cub.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/SortUtils.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>

#include <limits>
#include <c10/core/DeviceArray.h>

namespace at::native {

template <typename T>
static int minimum_grid_for_occupancy(T kernel, int max_block_size) {
  int minGridSize;
  int blockSize;
  C10_CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
      &minGridSize,
      &blockSize,
      kernel,
      /*dynamicSMemSize=*/0,
      max_block_size));
  return minGridSize;
}

// For very small sorts, use bitonicSortKVInPlace which performs
// better because it can sort multiple arrays within the same block of
// threads, improving occupancy.
//
// TODO: cub in CUDA 11.6 has a WarpMergeSort primitive that could
// replace the bitonic sort here.
struct SmallBitonicSort {
  template <int A, typename K, typename V, typename IndexType>
  void sort(
      at::cuda::detail::TensorInfo<K, IndexType> keyInfo,
      IndexType keySlices,
      IndexType keySliceSize,
      IndexType keySliceStride,
      at::cuda::detail::TensorInfo<V, IndexType> valueInfo,
      IndexType valueSliceStride,
      bool descending) {
    constexpr int sort_size = 32;
    constexpr int max_block_y = 16;
    constexpr int items_per_thread = 2;
    static_assert(sort_size % items_per_thread == 0, "");
    constexpr int block_x = sort_size / items_per_thread;

    TORCH_INTERNAL_ASSERT(keySliceSize <= sort_size);

    // Scale batch size down if the grid would be too small
    const auto min_grid = minimum_grid_for_occupancy(
        bitonicSortKVInPlace<
            A, -1, block_x, max_block_y,
            K, V, LTOp<K, true>, IndexType>,
        block_x * max_block_y);
    const auto max_batch = std::max(IndexType{1}, keySlices / min_grid);
    const int block_y = std::min(IndexType(max_block_y), max_batch);
    dim3 block(block_x, block_y);

    dim3 grid;
    const int grid_count = (keySlices + block_y - 1) / block_y;
    TORCH_INTERNAL_ASSERT(getGridFromTiles(grid_count, grid),
                          "Too many slices to sort");
    const auto stream = at::cuda::getCurrentCUDAStream();

    if (descending) {
      bitonicSortKVInPlace<A, -1, block_x, max_block_y>
        <<<grid, block, 0, stream>>>(
          keyInfo,
          keySlices,
          keySliceSize,
          keySliceStride,
          valueInfo,
          valueSliceStride,
          GTOp<K, true>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      bitonicSortKVInPlace<A, -1, block_x, max_block_y>
        <<<grid, block, 0, stream>>>(
          keyInfo,
          keySlices,
          keySliceSize,
          keySliceStride,
          valueInfo,
          valueSliceStride,
          LTOp<K, true>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
};

// For medium sizes (32 < n <= 4096) use radixSortKVInplace for better
// performance than the bitonic sort kernel.
struct MediumRadixSort {

  template <int A, typename K, typename V, typename IndexType>
  void sort(
      at::cuda::detail::TensorInfo<K, IndexType> keyInfo,
      IndexType keySlices,
      IndexType keySliceSize,
      IndexType keySliceStride,
      at::cuda::detail::TensorInfo<V, IndexType> valueInfo,
      IndexType valueSliceStride,
      bool descending) {

#define HANDLE_CASE(SIZE, ITEMS_PER_THREAD)         \
    fixed_size_sort<A, SIZE, ITEMS_PER_THREAD>(     \
        keyInfo,                                    \
        keySlices,                                  \
        keySliceSize,                               \
        keySliceStride,                             \
        valueInfo,                                  \
        valueSliceStride,                           \
        descending)

    int64_t ceilPowerOf2 = nextHighestPowerOf2(keySliceSize);
    TORCH_INTERNAL_ASSERT(ceilPowerOf2 <= 4096);
    switch (ceilPowerOf2) {
      case 4096:
        HANDLE_CASE(4096, 32);
        break;
      case 2048:
        HANDLE_CASE(2048, 32);
        break;
      case 1024:
      case 512:
      case 256:
        HANDLE_CASE(1024, 32);
        break;
      case 128:
      case 64:
        HANDLE_CASE(128, 4);
        break;
      case 32:
      case 16:
      case 8:
      case 4:
      case 2:
        HANDLE_CASE(32, 2);
        break;
      case 1:
        /* Nothing to do, data already sorted */
        break;
      default:
        TORCH_INTERNAL_ASSERT(false);
    }
#undef HANDLE_CASE

  }

  template <int A, int sort_size, int items_per_thread,
            typename K, typename V, typename IndexType>
  void fixed_size_sort(
      at::cuda::detail::TensorInfo<K, IndexType> keyInfo,
      IndexType keySlices,
      IndexType keySliceSize,
      IndexType keySliceStride,
      at::cuda::detail::TensorInfo<V, IndexType> valueInfo,
      IndexType valueSliceStride,
      bool descending) {
    static_assert(sort_size % items_per_thread == 0, "");
    constexpr int block = sort_size / items_per_thread;
    dim3 grid;
    TORCH_INTERNAL_ASSERT(getGridFromTiles(keySlices, grid),
                          "Too many slices to sort");

    const auto stream = at::cuda::getCurrentCUDAStream();
    radixSortKVInPlace<A, -1, block, items_per_thread>
        <<<grid, block, 0, stream>>>(
          keyInfo,
          keySlices,
          keySliceSize,
          keySliceStride,
          valueInfo,
          valueSliceStride,
          descending);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
};

template <typename Sorter>
void sortCommon(Sorter sorter, const TensorBase &key, const TensorBase &value,
                int dim, bool descending) {
  TORCH_CHECK(key.sizes() == value.sizes(),
              "Key tensor must have same size as value tensor");
  int dims = value.dim();
  TORCH_CHECK(dims <= MAX_DIMS, "value tensor has too many dimensions");
  // if key and value tensors have the same size, we do not need to check both

  ptrdiff_t inElements = key.numel();

  if (inElements == 0) {
    return;
  }

  int64_t keySliceSize = key.size(dim);
  ptrdiff_t keySlices = inElements / keySliceSize;

#define HANDLE_SORT_CASE(TYPE, A)                   \
  sorter.template sort<A>(                          \
      keyInfo,                                      \
      (TYPE) keySlices,                             \
      (TYPE) keySliceSize,                          \
      (TYPE) keyInfo.strides[collapseKeyDim],       \
      valueInfo,                                    \
      (TYPE) valueInfo.strides[collapseValueDim],   \
      descending)

  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis
  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, key.scalar_type(), "sortKeyValueInplace", [&]  {
    if (at::cuda::detail::canUse32BitIndexMath(key)) {
      at::cuda::detail::TensorInfo<scalar_t, unsigned int> keyInfo =
        at::cuda::detail::getTensorInfo<scalar_t, unsigned int>(key);
      at::cuda::detail::TensorInfo<int64_t, unsigned int> valueInfo =
        at::cuda::detail::getTensorInfo<int64_t, unsigned int>(value);

      auto strideKey = keyInfo.strides[dim];
      keyInfo.sizes[dim] = 1;
      int collapseKeyDim = keyInfo.collapseDims(dim);
      keyInfo.strides[collapseKeyDim] = strideKey;
      auto strideValue = valueInfo.strides[dim];
      valueInfo.sizes[dim]=1;
      int collapseValueDim = valueInfo.collapseDims(dim);
      valueInfo.strides[collapseValueDim] = strideValue;

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
      at::cuda::detail::TensorInfo<scalar_t, uint64_t> keyInfo =
        at::cuda::detail::getTensorInfo<scalar_t, uint64_t>(key);
      at::cuda::detail::TensorInfo<int64_t, uint64_t> valueInfo =
        at::cuda::detail::getTensorInfo<int64_t, uint64_t>(value);

      auto strideKey = keyInfo.strides[dim];
      keyInfo.sizes[dim] = 1;
      int collapseKeyDim = keyInfo.collapseDims(dim);
      keyInfo.strides[collapseKeyDim] = strideKey;
      auto strideValue = valueInfo.strides[dim];
      valueInfo.sizes[dim]=1;
      int collapseValueDim = valueInfo.collapseDims(dim);
      valueInfo.strides[collapseValueDim] = strideValue;

      // int64_t case is rare, just instantiate the generic version
      HANDLE_SORT_CASE(uint64_t, -1);
    }
  });
#undef HANDLE_SORT_CASE
}

void sortKeyValueInplace(
    const TensorBase& key,
    const TensorBase& value,
    int dim,
    bool descending,
    bool stable) {
  if (!stable && key.size(dim) <= 32) {
    // NOTE: Bitonic sort is unstable
    sortCommon(SmallBitonicSort{}, key, value, dim, descending);
  } else {
    sortCommon(MediumRadixSort{}, key, value, dim, descending);
  }
}

}  // namespace at::native
