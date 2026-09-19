#pragma once
#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>

namespace at::native {

TORCH_API at::Tensor clone_preserve_strides(const at::Tensor& self);

inline bool cat_should_skip_tensor(const Tensor& t) {
  return t.sym_numel() == 0 && t.dim() == 1;
}

// Check to see if the shape of tensors is compatible
// for being concatenated along a given dimension.
inline void check_cat_shape_except_dim(
    const Tensor& first,
    const Tensor& second,
    int64_t dimension,
    int64_t index) {
  int64_t first_dims = first.dim();
  int64_t second_dims = second.dim();
  TORCH_CHECK(
      first_dims == second_dims,
      "Tensors must have same number of dimensions: got ",
      first_dims,
      " and ",
      second_dims);
  for (const auto dim : c10::irange(first_dims)) {
    if (dim == dimension) {
      continue;
    }
    int64_t first_dim_size = first.sizes()[dim];
    int64_t second_dim_size = second.sizes()[dim];
    TORCH_CHECK(
        first_dim_size == second_dim_size,
        "Sizes of tensors must match except in dimension ",
        dimension,
        ". Expected size ",
        static_cast<long long>(first_dim_size),
        " but got size ",
        static_cast<long long>(second_dim_size),
        " for tensor number ",
        index,
        " in the list.");
  }
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

inline bool have_same_ndims(TensorList tensors) {
  auto ndim = tensors[0].dim();
  for (const auto& tensor : tensors) {
    if (tensor.dim() != ndim) {
      return false;
    }
  }
  return true;
}

inline void leading_dimension_matches(TensorList tensors, int64_t dim) {
  auto tensor_zero_size = tensors[0].sizes();
  std::vector<c10::SymInt> leading_dim_sizes(
      tensor_zero_size.begin(), tensor_zero_size.begin() + dim);
  for (const auto& tensor : tensors) {
    for (const auto j : c10::irange(dim)) {
      TORCH_CHECK(
          tensor.size(j) == leading_dim_sizes[j],
          "_chunk_cat expects same sizes of 0,...,dim-1 dimensions for all tensors");
    }
  }
}

inline int64_t preprocess_chunk_cat_inputs(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks) {
  TORCH_CHECK(num_chunks >= 1, "_chunk_cat expects positive num_chunks");
  TORCH_CHECK(
      !tensors.empty(), "_chunk_cat expects a non-empty input tensor list");
  auto expected_dtype = tensors[0].dtype();
  auto expected_device = tensors[0].device();
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

inline void check_split_with_sizes_copy_with_prefixes_inputs(
    const Tensor& self,
    IntArrayRef split_sizes,
    IntArrayRef num_prefixes,
    int64_t num_chunks,
    TensorList out) {
  TORCH_CHECK(num_chunks > 0, "expected positive num_chunks");
  TORCH_CHECK(
      self.layout() == kStrided && self.is_contiguous(),
      "expected a contiguous strided input");
  TORCH_CHECK(
      split_sizes.size() == out.size() && num_prefixes.size() == out.size(),
      "expected one split size and prefix count per output");
  TORCH_CHECK(
      self.numel() % num_chunks == 0,
      "input size must be divisible by num_chunks");
  int64_t remaining = self.numel() / num_chunks;
  for (const auto i : c10::irange(out.size())) {
    TORCH_CHECK(
        split_sizes[i] >= 0 && split_sizes[i] <= remaining,
        "split sizes must be non-negative and sum to the input chunk size");
    remaining -= split_sizes[i];
    TORCH_CHECK(
        num_prefixes[i] > 0 && split_sizes[i] % num_prefixes[i] == 0,
        "split size must be divisible by its positive prefix count");
    TORCH_CHECK(
        out[i].layout() == kStrided && out[i].is_contiguous(),
        "expected contiguous strided outputs");
    TORCH_CHECK(
        out[i].device() == self.device(),
        "input and outputs must be on the same device");
    TORCH_CHECK(
        self.scalar_type() == kByte || out[i].dtype() == self.dtype(),
        "output dtype must match the input unless the input has dtype uint8");
    TORCH_CHECK(
        out[i].numel() % num_chunks == 0 &&
            out[i].numel() / num_chunks * out[i].element_size() ==
                split_sizes[i] * self.element_size(),
        "output size must match the split size times num_chunks");
  }
  TORCH_CHECK(remaining == 0, "split sizes must sum to the input chunk size");
}

inline void check_chunk_cat_with_prefixes_inputs(
    TensorList tensors,
    IntArrayRef num_leading_dims,
    int64_t num_chunks,
    const Tensor& out) {
  TORCH_CHECK(num_chunks > 0, "expected positive num_chunks");
  TORCH_CHECK(!tensors.empty(), "expected a non-empty input tensor list");
  TORCH_CHECK(
      tensors.size() == num_leading_dims.size(),
      "expected one leading dimension count per input");
  TORCH_CHECK(
      out.layout() == kStrided && out.is_contiguous(),
      "expected a contiguous strided output");
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    const auto dim = num_leading_dims[i];
    TORCH_CHECK(
        tensor.layout() == kStrided && (dim == 0 || tensor.is_contiguous()),
        "prefix copies require contiguous strided inputs");
    TORCH_CHECK(tensor.numel() > 0, "expected non-empty inputs");
    TORCH_CHECK(
        dim >= 0 && dim < tensor.dim(),
        "leading dimension count must be non-negative and less than input ndim");
    TORCH_CHECK(
        dim == 0 || tensor.size(dim) % num_chunks == 0,
        "prefix copies require an evenly divisible shard dimension");
    TORCH_CHECK(
        tensor.dtype() == tensors[0].dtype(),
        "inputs must have the same dtype");
    TORCH_CHECK(
        tensor.device() == out.device(),
        "inputs and output must be on the same device");
  }
}

} // namespace at::native
