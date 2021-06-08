#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/SortingUtils.h>
#include <c10/macros/Macros.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/SortingRadixSelect.cuh>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/MemoryOverlap.h>
#include <THC/THCDeviceUtils.cuh> // only for THCRoundUp?
#include <THC/THCNumerics.cuh>
#include <THC/THCScanUtils.cuh>
#include <THC/THCTensorMathReduce.cuh> // AddOp

#include <cassert>
#include <cstdlib>

namespace at {
namespace native {

namespace {

// Finds the rank k element, and its index, of the values along dimension dim
template <typename scalar_t, typename index_t, int Dim>
__global__ void gatherKthValue(
    cuda::detail::TensorInfo<scalar_t, index_t> input,
    index_t inputSliceSize,
    index_t k,
    index_t numInputSlices,
    index_t inputWithinSliceStride,
    cuda::detail::TensorInfo<scalar_t, index_t> kthValue,
    cuda::detail::TensorInfo<int64_t, index_t> indices) {
  // Indices are limited to integer fp precision, so counts can fit in
  // int32, regardless of index_t
  __shared__ int smem[C10_WARP_SIZE]; // one per each warp, up to warp limit

  index_t slice = getLinearBlockId<index_t>();
  if (slice >= numInputSlices) {
    return;
  }

  // Find the start offset for our slice
  index_t sliceStartIndex =
      cuda::detail::IndexToOffset<scalar_t, index_t, Dim>::get(slice, input);
  index_t kthValueSliceStartIndex =
      cuda::detail::IndexToOffset<scalar_t, index_t, Dim>::get(slice, kthValue);
  index_t indicesSliceStartIndex =
      cuda::detail::IndexToOffset<int64_t, index_t, Dim>::get(slice, indices);

  scalar_t* inputSliceStart = &input.data[sliceStartIndex];
  scalar_t* kthValueSliceStart = &kthValue.data[kthValueSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  // Find the k-th highest element in our input
  scalar_t kValue = static_cast<scalar_t>(0);
  radixSelect<
      scalar_t,
      typename TopKTypeConfig<scalar_t>::RadixType,
      index_t,
      false>(
      inputSliceStart,
      k,
      inputSliceSize,
      inputWithinSliceStride,
      smem,
      &kValue);

  // Find the index of the k-th highest element
  index_t kValueIndex = 0;
  bool foundKValue = false;

  for (index_t i = threadIdx.x; i < inputSliceSize; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    scalar_t v = inRange ? doLdg(&inputSliceStart[i * inputWithinSliceStride])
                         : static_cast<scalar_t>(0);
    bool isKValue = inRange &&
        ((v == kValue) ||
         (THCNumerics<scalar_t>::isnan(v) &&
          THCNumerics<scalar_t>::isnan(kValue)));
    if (isKValue) {
      kValueIndex = i;
      foundKValue = true;
      break;
    }
  }

  if (foundKValue) {
    kthValueSliceStart[0] = kValue;
    indicesSliceStart[0] = kValueIndex;
  }
}

// CUDA kernel to find the median, and its index, of the values along dimension dim
template <typename scalar_t, typename index_t, int Dim>
__global__ void gatherMedian(
    cuda::detail::TensorInfo<scalar_t, index_t> values,
    cuda::detail::TensorInfo<int64_t, index_t> indices,
    cuda::detail::TensorInfo<scalar_t, index_t> input,
    index_t inputSliceSize,
    index_t numInputSlices,
    index_t inputWithinSliceStride,
    bool ignore_nan) {
  // Shared memory for the subroutine RadixSelect. Note that RadixSelect converts the
  // floating point type to int with the same relative ordering.
  __shared__ int smem[C10_WARP_SIZE]; // one per each warp, up to warp limit

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
      cuda::detail::IndexToOffset<scalar_t, index_t, Dim>::get(slice, input);

  scalar_t* valuesSliceStart = &values.data[valuesSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];
  scalar_t* inputSliceStart = &input.data[inputSliceStartIndex];

  index_t nan_count = 0;
  for (index_t i = threadIdx.x; i < inputSliceSize; i += blockDim.x) {
    scalar_t val = doLdg(&inputSliceStart[i * inputWithinSliceStride]);
    nan_count += THCNumerics<scalar_t>::isnan(val) ? 1 : 0;
  }

  // Counts number of nan values
  // This code performs a parallel sum reduction (not the most efficient code)
  __shared__ int64_t num_nan;
  if (threadIdx.x == 0) {
    num_nan = 0;
  }
  __syncthreads();
  if (nan_count > 0) {
    atomicAdd(&num_nan, nan_count);
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
      index_t,
      false>(
      inputSliceStart,
      k + 1,
      inputSliceSize,
      inputWithinSliceStride,
      smem,
      &median);

  valuesSliceStart[0] = median;

  // Find the index of the median value in the slice
  for (index_t i = threadIdx.x; i < inputSliceSize; i += blockDim.x) {
    scalar_t val = doLdg(&inputSliceStart[i * inputWithinSliceStride]);
    if (val == median ||
        (THCNumerics<scalar_t>::isnan(val) &&
         THCNumerics<scalar_t>::isnan(median))) {
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
      int collapse_indices_dim,
      cuda::detail::TensorInfo<scalar_t, index_t> self_info,
      int collapse_self_dim,
      int64_t num_slices,
      int64_t slice_size) {
    dim3 grid;
    if (!getGridFromTiles(num_slices, grid)) {
      AT_ERROR("slices are too many");
    }

    dim3 block(std::min(
        THCRoundUp(slice_size, (int64_t)C10_WARP_SIZE), (int64_t)1024));
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
      int collapse_values_dim,
      cuda::detail::TensorInfo<int64_t, index_t> indices_info,
      int collapse_indices_dim,
      cuda::detail::TensorInfo<scalar_t, index_t> self_info,
      int collapse_self_dim,
      int64_t num_slices,
      int64_t slice_size) {
    dim3 grid;
    if (!getGridFromTiles(num_slices, grid)) {
      AT_ERROR("slices are too many");
    }

    dim3 block(std::min(
        THCRoundUp(slice_size, (int64_t)C10_WARP_SIZE), (int64_t)1024));
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

template <typename scalar_t>
void kthvalue_cuda_template(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  int64_t slicesize = self.dim() == 0 ? 1 : self.size(dim);
  zero_numel_check_dims(self, dim, "kth_value()");

  TORCH_CHECK(k >= 1 && k <= slicesize, "selected number k out of range");

  at::assert_no_overlap(self, values);

  _reduction_with_indices_allocate_or_resize_output(
      values, indices, self, dim, keepdim);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return;
  }

  TORCH_CHECK(
      self.dim() <= MAX_TENSORINFO_DIMS,
      "cannot operate on more than ",
      MAX_TENSORINFO_DIMS,
      " dimensions");

  // Based on required index size, run the algorithm with the
  // appropriate index type
  if (self.numel() != 0) {
    AT_DISPATCH_INDEX_TYPES(
        cuda::detail::canUse32BitIndexMath(self) &&
        cuda::detail::canUse32BitIndexMath(values) &&
        cuda::detail::canUse32BitIndexMath(indices) ? ScalarType::Int : ScalarType::Long,
        "kth_value_launcher", [&] {
          run_launcher<scalar_t, index_t>(
            values, indices, self, dim, KthValueLauncher(k));
        });
  }

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
}

std::tuple<Tensor&, Tensor&> kthvalue_out_impl_cuda(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim) {
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, self.scalar_type(), "kthvalue_cuda", [&] {
        kthvalue_cuda_template<scalar_t>(
            values, indices, self, k, dim, keepdim);
      });
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor&, Tensor&> median_with_indices_impl(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    bool ignore_nan) {
  // See note [Writing Nondeterministic Operations]
  // If there are duplicate elements of a median value, the procedure for choosing which
  // of the duplicates to use for the indices output is nondeterministic.
  at::globalContext().alertNotDeterministic("median CUDA with indices output");
  NoNamesGuard guard;

  dim = at::maybe_wrap_dim(dim, self.dim());
  Tensor in = self.dim() > 0 ? self.contiguous() : self.unsqueeze(0);

  checkDeviceType("median", {values, indices}, self.device().type());
  checkScalarType("median", {indices, "indices", 1}, kLong);
  checkSameType("median", {values, "values", 0}, {self, "self", 2});

  TORCH_CHECK(
      self.dim() <= MAX_TENSORINFO_DIMS,
      "median() cannot operate on more than ",
      MAX_TENSORINFO_DIMS,
      " dimensions");

  std::vector<int64_t> out_shape = self.sizes().vec();
  zero_numel_check_dims(self, dim, "median()");
  if (self.dim() > 0) {
    if (keepdim) {
      out_shape[dim] = 1;
    } else {
      out_shape.erase(out_shape.begin() + dim);
    }
  }

  values.resize_(out_shape);
  indices.resize_(out_shape);

  // Only launch kernel for non-empty tensors
  if (self.numel() > 0) {
    // Ensure #dim is the same for all tensors required for reduction
    Tensor vals = keepdim && self.dim() > 0 ? values : values.unsqueeze(dim);
    Tensor inds = keepdim && self.dim() > 0 ? indices : indices.unsqueeze(dim);

    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, self.scalar_type(), "median_out_impl", [&] {
          if (cuda::detail::canUse32BitIndexMath(vals) &&
              cuda::detail::canUse32BitIndexMath(inds) &&
              cuda::detail::canUse32BitIndexMath(in)) {
            run_launcher<scalar_t, uint32_t>(
                vals, inds, in, dim, MedianLauncher(ignore_nan));
          } else {
            run_launcher<scalar_t, uint64_t>(
                vals, inds, in, dim, MedianLauncher(ignore_nan));
          }
        });
  }

  guard.reset();
  namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);

  return std::forward_as_tuple(values, indices);
}

Tensor median_impl(const Tensor& self, bool ignore_nan) {
  NoNamesGuard guard;

  int64_t size = self.numel();
  TORCH_CHECK(size > 0, "median() input tensor cannot be empty");

  // Sort input tensor to efficiently query for median element
  Tensor sorted = std::get<0>(self.flatten().sort());

  if (!ignore_nan) {
    // For torch.median return either the middle element or nan (sorted as
    // largest) if there are any
    int64_t k = (size - 1) / 2;
    return at::where(sorted[-1].isnan(), sorted[-1], sorted[k]);
  } else {
    // For torch.nanmedian return the middle element among the non-nan values
    Tensor k = ((size - 1) - sorted.isnan().sum()) / 2;
    return sorted[k.toType(kLong)];
  }
}

} // namespace

// Mark: kthvalue

std::tuple<Tensor&, Tensor&> kthvalue_out_cuda(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  // See note [Writing Nondeterministic Operations]
  // If there are duplicate elements of the kth value, the procedure for choosing which
  // of the duplicates to use for the indices output is nondeterministic.
  at::globalContext().alertNotDeterministic("kthvalue CUDA");
  auto result = [&]() {
    NoNamesGuard guard;
    // `kthvalue_out_impl_cuda` expects contiguous in input `self`.
    return kthvalue_out_impl_cuda(values, indices, self.contiguous(), k, dim, keepdim);
  }();
  namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);
  return result;
}

// Mark: median

std::tuple<Tensor&, Tensor&> median_out_cuda(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  return median_with_indices_impl(
      values, indices, self, dim, keepdim, /*ignore_nan=*/false);
}

Tensor median_cuda(const Tensor& self) {
  return median_impl(self, /*ignore_nan=*/false);
}

std::tuple<Tensor&, Tensor&> nanmedian_out_cuda(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  return median_with_indices_impl(
      values, indices, self, dim, keepdim, /*ignore_nan=*/true);
}

Tensor nanmedian_cuda(const Tensor& self) {
  return median_impl(self, /*ignore_nan=*/true);
}

} // namespace native
} // namespace at
