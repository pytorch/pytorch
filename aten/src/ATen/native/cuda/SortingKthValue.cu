#include <ATen/ATen.h>
#include <ATen/native/SortingUtils.h>
#include <assert.h>
#include <c10/macros/Macros.h>
#include <stdlib.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <THC/THCDeviceUtils.cuh> // only for THCRoundUp?
#include <THC/THCNumerics.cuh>
#include <THC/THCScanUtils.cuh>
#include <THC/THCTensorMathReduce.cuh> // AddOp

#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/SortingRadixSelect.cuh>
#include <ATen/NamedTensorUtils.h>

namespace at {
namespace native {

namespace {

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
    bool isKValue = inRange && THCNumerics<scalar_t>::eq_with_nan(v, kValue);

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

    dim3 block(
        std::min(THCRoundUp(slice_size, (int64_t)C10_WARP_SIZE), (int64_t)1024));
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
  int64_t slicesize = self.size(dim);
  // FIXME: This seems bogus, I only do this because it was the old behaviour.
  //        The reductions are fine, as long as the axis being reduced along
  //        isn't of 0 elements (and the output has elements).
  TORCH_CHECK(
      self.numel() > 0,
      "cannot perform reduction function kthvalue",
      " on tensor with no elements because the operation does not have an identity");
  TORCH_CHECK(k >= 1 && k <= slicesize, "selected number k out of range");

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
  if (cuda::detail::canUse32BitIndexMath(self) &&
      cuda::detail::canUse32BitIndexMath(values) &&
      cuda::detail::canUse32BitIndexMath(indices)) {
    run_launcher<scalar_t, uint32_t>(
        values, indices, self, dim, KthValueLauncher(k));
  } else {
    run_launcher<scalar_t, uint64_t>(
        values, indices, self, dim, KthValueLauncher(k));
  }

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

// this does not reduce to median with dim because we don't want to copy twice
template <typename scalar_t>
Tensor median_cuda_template(const Tensor& self) {
  TORCH_CHECK(self.numel() > 0, "median cannot be called with empty tensor");
  if (self.dim() == 0 && self.numel() == 1) {
    return self.clone(at::MemoryFormat::Contiguous);
  }
  auto self_copy = self.clone(at::MemoryFormat::Contiguous).view(-1);
  auto values = at::empty({1}, self.options());
  auto indices = at::empty({1}, self.options().dtype(kLong));
  TORCH_CHECK(
      self.dim() <= MAX_TENSORINFO_DIMS,
      "cannot operate on more than ",
      MAX_TENSORINFO_DIMS,
      " dimensions");

  // Based on required index size, run the algorithm with the
  // appropriate index type
  if (cuda::detail::canUse32BitIndexMath(self) &&
      cuda::detail::canUse32BitIndexMath(values) &&
      cuda::detail::canUse32BitIndexMath(indices)) {
    run_launcher<scalar_t, uint32_t>(
        values,
        indices,
        self_copy,
        0,
        KthValueLauncher((self_copy.size(0) + 1) / 2)); // KthValue is 1-based
  } else {
    run_launcher<scalar_t, uint64_t>(
        values,
        indices,
        self_copy,
        0,
        KthValueLauncher((self_copy.size(0) + 1) / 2)); // KthValue is 1-based
  }
  return values.view({});
}

} // namespace

static std::tuple<Tensor&, Tensor&> kthvalue_out_impl_cuda(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, self.scalar_type(), "kthvalue_cuda", [&] {
    kthvalue_cuda_template<scalar_t>(values, indices, self, k, dim, keepdim);
  });
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor&, Tensor&> kthvalue_out_cuda(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim) {
  auto result = [&]() {
    NoNamesGuard guard;
    return kthvalue_out_impl_cuda(values, indices, self, k, dim, keepdim);
  }();
  namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);
  return result;
}

Tensor median_cuda(const Tensor& self) {
  NoNamesGuard guard;
  return AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, self.scalar_type(), "median", [&] {
    return median_cuda_template<scalar_t>(self);
  });
}

} // namespace native
} // namespace at
