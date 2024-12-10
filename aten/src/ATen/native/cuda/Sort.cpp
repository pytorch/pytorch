#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/Sort.h>
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Sorting.h>
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

namespace at::native {

std::vector<int64_t> infer_dense_strides_dim_last(const Tensor & self, int64_t dim);

void fillSliceWithIndex(const Tensor& t, int64_t dim) {
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
void sort_cuda_kernel(
    const TensorBase& self_base,
    const TensorBase& values_base,
    const TensorBase& indices_base,
    int64_t dim,
    bool descending,
    bool stable) {
  // this algorithm is always stable

  // Macro for converting `TensorBase` -> `Tensor` without
  // reference count bumps.
#define TOTENSOR(BASE, VAR)           \
  OptionalTensorRef opt_##BASE(BASE); \
  const Tensor& VAR = *opt_##BASE;

  // Converting TensorBase into Tensor.
  // We will need Tensor's methods from this point onwards.
  TOTENSOR(self_base, self);
  TOTENSOR(values_base, values);
  TOTENSOR(indices_base, indices);

  TORCH_CHECK(self.sizes()[dim] <= std::numeric_limits<int>::max(),
    "The dimension being sorted can not have more than INT_MAX elements.");

  const auto self_dtype = self.dtype();
  TORCH_CHECK(self_dtype != ScalarType::ComplexFloat && self_dtype != ScalarType::ComplexDouble,
    "Sort currently does not support complex dtypes on CUDA.");

  // use inplace algorithm for smaller input sizes without stable=True
  if (should_use_small_sort(self, dim)) {
    // from thc: sorted->values, indices->indices, input->self
    fillSliceWithIndex(indices, dim);

    // We sort k/v pairs in-place; copy unsorted input to output
    values.copy_(self);

    // Sort using our in-place k/v kernel that supports arbitrary
    // layout
    sortKeyValueInplace(values, indices, dim, descending, stable);
    return;
  }

  Tensor self_;
  bool newself = false;
  if (self.is_non_overlapping_and_dense() && self.stride(dim) == 1) {
    self_ = self;
  } else {
    auto new_strides_unsort = infer_dense_strides_dim_last(self, dim);
    self_ = at::empty_strided(self.sizes(), new_strides_unsort, self.options());
    self_.copy_(self);
    newself = true;
  }

  c10::MaybeOwned<Tensor> values_tmp, indices_tmp;
  if (values.strides() == self_.strides() && (newself || get_overlap_status(self, values) == MemOverlapStatus::No)) {
    values_tmp = c10::MaybeOwned<Tensor>::borrowed(values);
  } else {
    values_tmp = c10::MaybeOwned<Tensor>::owned(
        at::empty_strided(self_.sizes(), self_.strides(), self_.options()));
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
}

// TODO: we should handle this accordingly when we start using REGISTER_HIP_DISPATCH,
// since REGISTER_DISPATCH won't work in this cpp file.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CUDA_DISPATCH(sort_stub, &sort_cuda_kernel)

}  // namespace at::native
