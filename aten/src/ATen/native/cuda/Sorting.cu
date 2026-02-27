#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/Sorting.h>
#include <ATen/core/TensorBase.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <c10/macros/Macros.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/SortingRadixSelect.cuh>

#include <c10/cuda/CUDAStream.h>

#include <cassert>
#include <cstdlib>

namespace at::native {

namespace {

// Finds the rank k element, and its index, of the values along dimension dim
template <typename scalar_t, typename index_t, int Dim>
__global__ void gatherKthValue(
    cuda::detail::TensorInfo<const scalar_t, index_t> input,
    index_t inputSliceSize,
    index_t k,
    index_t numInputSlices,
    index_t inputWithinSliceStride,
    cuda::detail::TensorInfo<scalar_t, index_t> kthValue,
    cuda::detail::TensorInfo<int64_t, index_t> indices) {
  // Indices are limited to integer fp precision, so counts can fit in
  // int32, regardless of index_t
#ifndef USE_ROCM
  __shared__ int smem[C10_WARP_SIZE]; // one per each warp, up to warp limit
#else
  // Maximum shared memory size for radix select (used in countRadixAggregateCounts): NUM_BUFFERS * MAX_WARPS * RADIX_SIZE.
  // HIP workgroups have at most 1024 threads. Warp size is at least 32 (can be 64 on some
  // architectures), so we use 32 for safety: 2 buffers * (1024/32) warps * 4 radix bins = 256.
  __shared__ int smem[256];
#endif

  index_t slice = getLinearBlockId<index_t>();
  if (slice >= numInputSlices) {
    return;
  }

  // Find the start offset for our slice
  index_t sliceStartIndex =
      cuda::detail::IndexToOffset<const scalar_t, index_t, Dim>::get(slice, input);
  index_t kthValueSliceStartIndex =
      cuda::detail::IndexToOffset<scalar_t, index_t, Dim>::get(slice, kthValue);
  index_t indicesSliceStartIndex =
      cuda::detail::IndexToOffset<int64_t, index_t, Dim>::get(slice, indices);

  const scalar_t* inputSliceStart = &input.data[sliceStartIndex];
  scalar_t* kthValueSliceStart = &kthValue.data[kthValueSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  // Find the k-th highest element in our input
  scalar_t kValue = static_cast<scalar_t>(0);
  radixSelect<
      scalar_t,
      typename TopKTypeConfig<scalar_t>::RadixType,
      index_t>(
      inputSliceStart,
      k,
      false,
      inputSliceSize,
      inputWithinSliceStride,
      smem,
      &kValue);

  // Find the index of the k-th highest element
  __shared__ int32_t minIndexFound;

  if (threadIdx.x == 0) {
      minIndexFound = static_cast<int32_t>(inputSliceSize);
  }
  __syncthreads();

  for (index_t i = threadIdx.x; i < inputSliceSize; i += blockDim.x) {
      // Early exit based on best-so-far
      if (i >= minIndexFound) {
          break;
      }

      scalar_t v = doLdg(&inputSliceStart[i * inputWithinSliceStride]);
      bool isKValue =
          ((v == kValue) || (at::_isnan(v) && at::_isnan(kValue)));

      if (isKValue) {
          atomicMin(&minIndexFound, static_cast<int32_t>(i));
          break;
      }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
      indicesSliceStart[0] = static_cast<index_t>(minIndexFound);
      kthValueSliceStart[0] = kValue;
  }
}

// CUDA kernel to find the median, and its index, of the values along dimension dim
template <typename scalar_t, typename index_t, int Dim>
__global__ void gatherMedian(
    cuda::detail::TensorInfo<scalar_t, index_t> values,
    cuda::detail::TensorInfo<int64_t, index_t> indices,
    cuda::detail::TensorInfo<const scalar_t, index_t> input,
    index_t inputSliceSize,
    index_t numInputSlices,
    index_t inputWithinSliceStride,
    bool ignore_nan) {
  // Shared memory for the subroutine RadixSelect. Note that RadixSelect converts the
  // floating point type to int with the same relative ordering.
#ifndef USE_ROCM
  __shared__ int smem[C10_WARP_SIZE]; // one per each warp, up to warp limit
#else
  // Maximum shared memory size for radix select (used in countRadixAggregateCounts): NUM_BUFFERS * MAX_WARPS * RADIX_SIZE.
  // HIP workgroups have at most 1024 threads. Warp size is at least 32 (can be 64 on some
  // architectures), so we use 32 for safety: 2 buffers * (1024/32) warps * 4 radix bins = 256.
  __shared__ int smem[256];
#endif

  index_t slice = getLinearBlockId<index_t>();
  if (slice >= numInputSlices) {
    return;
  }

  // Finds the start offset for our slice
  index_t valuesSliceStartIndex =
      cuda::detail::IndexToOffset<scalar_t, index_t, Dim>::get(slice, values);
  index_t indicesSliceStartIndex =
      cuda::detail::IndexToOffset<int64_t, index_t, Dim>::get(slice, indices);
  index_t inputSliceStartIndex =
      cuda::detail::IndexToOffset<const scalar_t, index_t, Dim>::get(slice, input);

  scalar_t* valuesSliceStart = &values.data[valuesSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];
  const scalar_t* inputSliceStart = &input.data[inputSliceStartIndex];

  index_t nan_count = 0;
  for (index_t i = threadIdx.x; i < inputSliceSize; i += blockDim.x) {
    scalar_t val = doLdg(&inputSliceStart[i * inputWithinSliceStride]);
    nan_count += at::_isnan(val) ? 1 : 0;
  }

  // Counts number of nan values
  // This code performs a parallel sum reduction (not the most efficient code)
  __shared__ int64_t num_nan;
  if (threadIdx.x == 0) {
    num_nan = 0;
  }
  __syncthreads();
  if (nan_count > 0) {
    gpuAtomicAddNoReturn(&num_nan, nan_count);
  }
  __syncthreads();

  // For torch.median, if we found nan set k to last index so the computed value
  // is nan, otherwise set k to the middle element of the non-nan values
  index_t k = (!ignore_nan && num_nan > 0) ? inputSliceSize - 1
                                           : (inputSliceSize - num_nan - 1) / 2;

  // Find the median
  scalar_t median = static_cast<scalar_t>(0);
  radixSelect<
      scalar_t,
      typename TopKTypeConfig<scalar_t>::RadixType,
      index_t>(
      inputSliceStart,
      k + 1,
      false,
      inputSliceSize,
      inputWithinSliceStride,
      smem,
      &median);

  valuesSliceStart[0] = median;

  // Find the index of the median value in the slice
  for (index_t i = threadIdx.x; i < inputSliceSize; i += blockDim.x) {
    scalar_t val = doLdg(&inputSliceStart[i * inputWithinSliceStride]);
    if (val == median || (at::_isnan(val) && at::_isnan(median))) {
      indicesSliceStart[0] = i;
      break;
    }
  }
}

struct KthValueLauncher {
  int64_t k;

  KthValueLauncher(int64_t k) : k(k) {}

  template <typename scalar_t, typename index_t, int all_dims>
  inline void launch(
      cuda::detail::TensorInfo<scalar_t, index_t> values_info,
      int collapse_values_dim,
      cuda::detail::TensorInfo<int64_t, index_t> indices_info,
      [[maybe_unused]] int collapse_indices_dim,
      cuda::detail::TensorInfo<const scalar_t, index_t> self_info,
      int collapse_self_dim,
      int64_t num_slices,
      int64_t slice_size) {
    dim3 grid;
    if (!getGridFromTiles(num_slices, grid)) {
      TORCH_CHECK(false, "slices are too many");
    }

    dim3 block(std::min(
        round_up(slice_size, (int64_t)at::cuda::warp_size()), (int64_t)1024));
    auto stream = at::cuda::getCurrentCUDAStream();
    gatherKthValue<scalar_t, index_t, all_dims><<<grid, block, 0, stream>>>(
        self_info,
        slice_size,
        k,
        num_slices,
        /* The actual dimension that the k-selection is running in */
        /* may have changed from collapseDims() */
        self_info.strides[collapse_self_dim],
        values_info,
        indices_info);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
};

struct MedianLauncher {
  bool ignore_nan;

  MedianLauncher(bool ignore_nan) : ignore_nan(ignore_nan) {}

  template <typename scalar_t, typename index_t, int all_dims>
  inline void launch(
      cuda::detail::TensorInfo<scalar_t, index_t> values_info,
      [[maybe_unused]] int collapse_values_dim,
      cuda::detail::TensorInfo<int64_t, index_t> indices_info,
      [[maybe_unused]] int collapse_indices_dim,
      cuda::detail::TensorInfo<const scalar_t, index_t> self_info,
      int collapse_self_dim,
      int64_t num_slices,
      int64_t slice_size) {
    dim3 grid;
    if (!getGridFromTiles(num_slices, grid)) {
      TORCH_CHECK(false, "slices are too many");
    }

    dim3 block(std::min(
        round_up(slice_size, (int64_t)at::cuda::warp_size()), (int64_t)1024));
    auto stream = at::cuda::getCurrentCUDAStream();
    gatherMedian<scalar_t, index_t, all_dims><<<grid, block, 0, stream>>>(
        values_info,
        indices_info,
        self_info,
        slice_size,
        num_slices,
        self_info.strides[collapse_self_dim],
        ignore_nan);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
};

}  // namespace (anonymous)

void launch_kthvalue_kernel(
    const TensorBase &values, const TensorBase &indices,
    const TensorBase &self, int64_t dim, int64_t k) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "kthvalue_cuda", [&] {
    AT_DISPATCH_INDEX_TYPES(
        cuda::detail::canUse32BitIndexMath(self) &&
        cuda::detail::canUse32BitIndexMath(values) &&
        cuda::detail::canUse32BitIndexMath(indices) ? ScalarType::Int : ScalarType::Long,
        "kth_value_launcher", [&] {
          run_launcher<scalar_t, index_t>(
              values, indices, self, dim, KthValueLauncher(k));
    });
  });
}

void launch_median_kernel(
    const TensorBase &vals, const TensorBase &inds,
    const TensorBase &self, int64_t dim, bool ignore_nan) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "median_out_impl", [&] {
        if (cuda::detail::canUse32BitIndexMath(vals) &&
            cuda::detail::canUse32BitIndexMath(inds) &&
            cuda::detail::canUse32BitIndexMath(self)) {
          run_launcher<scalar_t, uint32_t>(
              vals, inds, self, dim, MedianLauncher(ignore_nan));
        } else {
          run_launcher<scalar_t, uint64_t>(
              vals, inds, self, dim, MedianLauncher(ignore_nan));
        }
      });
}

} // namespace at::native
