#pragma once
#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>

namespace at::native {

TORCH_API at::Tensor clone_preserve_strides(const at::Tensor& self);

inline bool cat_should_skip_tensor(const Tensor& t) {
  return t.sym_numel() == 0 && t.dim() == 1;
}

inline void check_cat_no_zero_dim(const MaterializedITensorListRef& tensors) {
  [[maybe_unused]] int64_t i = 0;
  for (const Tensor& t : tensors) {
    TORCH_CHECK(
        t.dim() > 0,
        "zero-dimensional tensor (at position ",
        i,
        ") cannot be concatenated");
    i++;
  }
}

inline int64_t get_num_splits(
    const Tensor& self,
    int64_t split_size,
    int64_t dim) {
  TORCH_CHECK(self.dim() != 0, "split expects at least a 1-dimensional tensor");
  TORCH_CHECK(
      split_size >= 0,
      "split expects split_size be non-negative, but got split_size=",
      split_size);
  int64_t dim_size = self.size(dim);
  TORCH_CHECK(
      split_size > 0 || dim_size == 0,
      "split_size can only be 0 if dimension size is 0, "
      "but got dimension size of ",
      dim_size);
  // if split_size is 0 and dimension size is 0, there is 1 split.
  int64_t num_splits = 1;
  if (split_size != 0) {
    // ensuring num_splits is at least 1 makes consistent the case where
    // split_size > dim_size (returns a single split).  We might want to error
    // here, but keep it for BC.
    num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  }
  return num_splits;
}

inline bool have_same_ndims(const TensorList& tensors) {
  auto ndim = tensors[0].dim();
  for (auto const& tensor : tensors) {
    if (tensor.dim() != ndim) {
      return false;
    }
  }
  return true;
}

inline void leading_dimension_matches(const TensorList& tensors, int64_t dim) {
  auto tensor_zero_size = tensors[0].sizes();
  std::vector<c10::SymInt> leading_dim_sizes(
      tensor_zero_size.begin(), tensor_zero_size.begin() + dim);
  for (const auto& tensor : tensors) {
    for (const auto j : c10::irange(dim)) {
      TORCH_CHECK(
          tensor.size(j) == tensor_zero_size[j],
          "_chunk_cat expects same sizes of 0,...,dim-1 dimensions for all tensors");
    }
  }
}

inline int64_t preprocess_chunk_cat_inputs(
    const TensorList& tensors,
    int64_t dim,
    int64_t num_chunks) {
  TORCH_CHECK(num_chunks >= 1, "_chunk_cat expects positive num_chunks");
  TORCH_CHECK(
      !tensors.empty(), "_chunk_cat expects a non-empty input tensor list");
  auto const& expected_dtype = tensors[0].dtype();
  auto const& expected_device = tensors[0].device();
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.numel() > 0, "_chunk_cat expects non-empty tensor");
    TORCH_CHECK(
        tensor.dtype() == expected_dtype,
        "_chunk_cat expects all input tensors with the same dtype");
    TORCH_CHECK(
        tensor.device() == expected_device,
        "_chunk_cat expects all inputs tensors on the same device");
  }
  if (have_same_ndims(tensors)) {
    dim = maybe_wrap_dim(dim, tensors[0].dim());
  } else {
    TORCH_CHECK(
        dim >= 0,
        "_chunk_cat expects non-negative dim when input tensors have different ndims")
    for (const auto& tensor : tensors) {
      TORCH_CHECK(
          dim < tensor.ndimension(),
          "_chunk_cat expects dim < ndim for all input tensors");
    }
  }
  leading_dimension_matches(tensors, dim);
  return dim;
}

} // namespace at::native
