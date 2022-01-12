#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/Sort.h>
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/sort_native.h>
#include <ATen/ops/zeros.h>
#endif

#include <limits>

namespace at { namespace native {

// We perform a segmented sort in cub with inputs that have
// more than 1024/2048 elements along the selected dimension.
// Otherwise, we do an inplace bitonic sort (see sortKeyValueInplace).
bool should_use_small_sort(const TensorBase &self, int64_t dim) {
  int64_t nsort = self.sizes()[dim];
  int64_t threshold;
  if (self.scalar_type() == kLong || self.scalar_type() == kDouble) {
    threshold = 1024;
  } else {
    threshold = 2048;
  }
  return nsort <= threshold;
}

std::vector<int64_t> infer_dense_strides_dim_last(const Tensor & self, int64_t dim);

void fillSliceWithIndex(Tensor& t,int dim) {
  if (t.numel()) {
    auto sizes = DimVector(t.dim(), 1);
    sizes[dim] = t.sizes()[dim];
    auto range = at::arange(t.sizes()[dim], t.options());
    auto rangeview = range.view(sizes);
    t.copy_(rangeview);
  }
}

// We perform a segmented sort in cub with inputs that have
// more than 1024/2048 elements along the selected dimension.
// Otherwise, we do an inplace bitonic sort (see sortKeyValueInplace).
std::tuple<Tensor &,Tensor &> sort_out_stable_cuda(const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, Tensor & values, Tensor & indices) {
  // this algorithm is always stable
  TORCH_INTERNAL_ASSERT(stable.has_value(), "sort_out(): c10::optional<bool> for stable has to have value.");
  TensorArg self_arg{self, "self", 1}, values_arg{values, "values", 2}, indices_arg{indices, "indices", 3};
  checkAllSameGPU(__func__, {self_arg, values_arg, indices_arg});

  bool is_non_overlapping_and_dense = self.is_non_overlapping_and_dense();
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  int64_t nsort = self.sizes()[dim];

  TORCH_CHECK(nsort <= std::numeric_limits<int>::max(),
    "The dimension being sorted can not have more than INT_MAX elements.");

  const auto self_dtype = self.dtype();
  // FIXME: remove this check once cub sort supports bool
  TORCH_CHECK(self_dtype != ScalarType::Bool,
    "Sort currently does not support bool dtype on CUDA.");
  TORCH_CHECK(self_dtype != ScalarType::ComplexFloat && self_dtype != ScalarType::ComplexDouble,
    "Sort currently does not support complex dtypes on CUDA.");

  if (ndim == 0) {
    if (!values.defined()) {
      values = self.clone();
    } else {
      values.resize_as_(self);
      values.copy_(self);
    }
    if (!indices.defined()) {
      indices = at::zeros({}, self.options().dtype(kLong));
    } else {
      indices.resize_as_(self);
      indices.zero_();
    }
    return std::forward_as_tuple(values, indices);
  }

  // use inplace algorithm for smaller input sizes without stable=True
  if (should_use_small_sort(self, dim) && !stable.value()) {
    // from thc: sorted->values, indices->indices, input->self

    if (!values.defined()) {
      values = at::empty_like(self);
    }
    if (!indices.defined()) {
      indices = at::empty_like(self, self.options().dtype(kLong));
    }

    // Make sure sufficient output space is allocated
    auto self_size = self.sizes();
    at::native::resize_output(values, self_size);
    at::native::resize_output(indices, self_size);
    fillSliceWithIndex(indices, dim);

    // We sort k/v pairs in-place; copy unsorted input to output
    values.copy_(self);

    // Sort using our in-place k/v kernel that supports arbitrary
    // layout
    sortKeyValueInplace(values, indices, dim, descending);
    return std::forward_as_tuple(values, indices);
  }

  Tensor self_;
  bool newself = false;
  if (is_non_overlapping_and_dense && self.stride(dim) == 1) {
    self_ = self;
  } else {
    auto new_strides_unsort = infer_dense_strides_dim_last(self, dim);
    self_ = at::empty_strided(self.sizes(), new_strides_unsort, self.options());
    self_.copy_(self);
    newself = true;
  }

  c10::MaybeOwned<Tensor> values_tmp, indices_tmp;
  if (!values.defined()) {
    if (is_non_overlapping_and_dense) {
      values = at::empty_strided(self.sizes(), self.strides(), self.options());
    } else {
      auto strides = at::infer_dense_strides(self.sizes(), self.strides());
      values = at::empty_strided(self.sizes(), strides, self.options());
    }
  } else {
    TORCH_CHECK(self_.scalar_type() == values.scalar_type(),
      "Unexpected dtype for values, expect ", self_.scalar_type(), ", got ", values.scalar_type());
    values.resize_as_(self);
  }

  if (values.strides() == self_.strides() && (newself || get_overlap_status(self, values) == MemOverlapStatus::NO)) {
    values_tmp = c10::MaybeOwned<Tensor>::borrowed(values);
  } else {
    values_tmp = c10::MaybeOwned<Tensor>::owned(
        at::empty_strided(self_.sizes(), self_.strides(), self_.options()));
  }

  if (!indices.defined()) {
    if (is_non_overlapping_and_dense) {
      indices = at::empty_strided(self.sizes(), self.strides(), self.options().dtype(kLong));
    } else {
      auto strides = at::infer_dense_strides(self.sizes(), self.strides());
      indices = at::empty_strided(self.sizes(), strides, self.options().dtype(kLong));
    }
  } else {
    TORCH_CHECK(kLong == indices.scalar_type(),
      "Unexpected dtype for values, expect torch.long, got ", indices.scalar_type());
    indices.resize_as_(self);
  }
  if (indices.strides() != self_.strides()) {
    indices_tmp = c10::MaybeOwned<Tensor>::owned(
        at::empty_strided(self_.sizes(), self_.strides(), self_.options().dtype(kLong)));
  } else {
    indices_tmp = c10::MaybeOwned<Tensor>::borrowed(indices);
  }

  launch_stable_sort_kernel(self_, dim, descending, *values_tmp, *indices_tmp);

  if (!values_tmp->is_same(values)) {
    values.copy_(*values_tmp);
  }
  if (!indices_tmp->is_same(indices)) {
    indices.copy_(*indices_tmp);
  }
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor &,Tensor &> sort_out_cuda(const Tensor & self, int64_t dim, bool descending, Tensor & values, Tensor & indices) {
  return sort_out_stable_cuda(self, /*stable=*/false, dim, descending, values, indices);
}

std::tuple<Tensor,Tensor> sort_stable_cuda(const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending) {
  Tensor values, indices;
  return sort_out_stable_cuda(self, stable, dim, descending, values, indices);
}

std::tuple<Tensor,Tensor> sort_cuda(const Tensor & self, int64_t dim, bool descending) {
  return sort_stable_cuda(self, /*stable=*/false, dim, descending);
}

}}  // namespace at::native
