#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <c10/util/Optional.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_nested_tensor_size_native.h>
#include <ATen/ops/_nested_tensor_storage_offsets_native.h>
#include <ATen/ops/_nested_tensor_strides_native.h>
#include <ATen/ops/chunk_native.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/split_with_sizes_native.h>
#endif

namespace at {
namespace native {

/**
 * Thin wrapper around get_nested_sizes that is registered as a native function
 *
 * @return The nested tensors' size tensor.
 */
at::Tensor _nested_tensor_size(const at::Tensor& self) {
  return get_nested_sizes(self);
}

at::Tensor _nested_tensor_strides(const at::Tensor& self){
  return  get_nested_tensor_impl(self) -> get_nested_strides();
}
at::Tensor _nested_tensor_storage_offsets(const at::Tensor& self){
  return get_nested_tensor_impl(self) -> get_storage_offsets();
}

// Helper functions for getting information about a nested tensor's shape.
std::vector<int64_t> NestedTensor_get_max_size_from_size_tensor(
    const Tensor& sizes) {
  if (sizes.dim() == 0) {
    return {};
  }
  const auto sizes_ptr = sizes.data_ptr<int64_t>();
  const auto sizes_size_0 = sizes.sizes()[0];
  const auto sizes_size_1 = sizes.sizes()[1];
  TORCH_INTERNAL_ASSERT(sizes_size_1 > 0);
  std::vector<int64_t> results(sizes_size_1, 0);
  for (const auto ii : c10::irange(sizes_size_0)) {
    for (const auto jj : c10::irange(sizes_size_1)) {
      auto val = sizes_ptr[ii * sizes_size_1 + jj];
      if (results[jj] < val) {
        results[jj] = val;
      }
    }
  }
  return results;
}

std::vector<int64_t> NestedTensor_get_max_size(const NestedTensorImpl& nt) {
  return NestedTensor_get_max_size_from_size_tensor(
      nt.get_nested_sizes());
}

int64_t get_consistent_last_dim_of_nested_tensor(const NestedTensorImpl& nt) {
  c10::optional<int64_t> last_dim = nt.opt_size(-1);
  TORCH_CHECK(
      last_dim != c10::nullopt,
      "Expected all tensors in nested tensor to have the same trailing dimension, instead last dimension equals: ",
      nt.get_nested_sizes().select(1, -1));
  return *last_dim;
}

std::vector<Tensor> chunk_nested_tensor(const Tensor& self, int64_t chunks, int64_t dim) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "chunk() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  TORCH_CHECK(self.dim() - 1 == dim,
           "Chunk for nested tensors is currently only supported for the last dimension.");
  TORCH_CHECK(chunks > 0,"chunk expects `chunks` to be greater than 0, got: ", chunks);
  TORCH_CHECK(self.is_contiguous(), "chunk expects `self` to be contiguous.");
  auto self_impl = get_nested_tensor_impl(self);
  const int64_t last_dim_size = get_consistent_last_dim_of_nested_tensor(*self_impl);
    TORCH_CHECK(last_dim_size % chunks == 0,
           "Chunk for nested tensors is only supported for nested tensors with trailing dimension divisible by chunks, got: ",
           last_dim_size, " % ", chunks, " != 0");
  int64_t n_tensors = self.size(0);
  int64_t split_size = last_dim_size / chunks;
  std::vector<Tensor> splits(chunks);
  const auto& sizes = self_impl->get_nested_sizes();
  const auto& strides = self_impl->get_nested_strides();
  const auto offsets = self_impl->get_storage_offsets();
  int64_t *offsets_ptr = offsets.data_ptr<int64_t>();
  // Account for the implicit batch dim
  --dim;
  int64_t tensor_dim = sizes.size(1);
  for (const auto split_idx : c10::irange(chunks)) {
      auto new_sizes = at::native::empty_like(sizes);
      auto new_strides = strides.clone();
      auto new_offsets = at::native::empty_like(offsets);
      int64_t *size_ptr = new_sizes.data_ptr<int64_t>();
      int64_t *new_offsets_ptr = new_offsets.data_ptr<int64_t>();
      // Get start val for each split
      int64_t start_val = split_idx * split_size;
      for (int64_t i : c10::irange(n_tensors)) {
        const int64_t index = i * tensor_dim + dim;
        new_offsets_ptr[i] = offsets_ptr[i] + start_val;
        size_ptr[index] = split_size;
      }
      splits[split_idx] = create_nested_view_tensor(
        self, std::move(new_sizes), std::move(new_strides), std::move(new_offsets));
  }
  return splits;
}

std::vector<Tensor> split_with_sizes_nested(
    const Tensor& self,
    c10::IntArrayRef split_sizes,
    int64_t dim) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "split_with_sizes() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  TORCH_CHECK(self.dim() - 1 == dim,
           "split_with_sizes for nested tensors is currently only supported for the last dimension.");
  auto num_splits = split_sizes.size();
  TORCH_CHECK(num_splits > 0,
           "split_with_sizes expects number of splits to be greater than 0, got: ", num_splits);
  TORCH_CHECK(self.is_contiguous(), "split_with_sizes expects `self` to be contiguous.");

  // Ensure entire dim is split.
  int64_t total_size = 0;
  for (const auto split_size : split_sizes) {
      total_size += split_size;
  }
  auto self_impl = get_nested_tensor_impl(self);
  auto self_size = get_consistent_last_dim_of_nested_tensor(*self_impl);
  TORCH_CHECK(total_size == self_size,
          "split_with_sizes expects split_sizes to sum exactly to ", self_size,
          " (input tensor's size at dimension ", dim, "), but got split_sizes=", split_sizes);

  int64_t n_tensors = self.size(0);
  std::vector<Tensor> splits(num_splits);
  const auto& sizes = self_impl->get_nested_sizes();
  const auto& strides = self_impl->get_nested_strides();
  const auto offsets = self_impl->get_storage_offsets();
  int64_t *offsets_ptr = offsets.data_ptr<int64_t>();
  // Account for the implicit batch dim
  --dim;
  int64_t tensor_dim = sizes.size(1);
  int64_t start_val = 0;
  for (const auto split_idx : c10::irange(num_splits)) {
    auto split_size = split_sizes[split_idx];
    auto new_sizes = at::native::empty_like(sizes);
    auto new_strides = strides.clone();
    auto new_offsets = at::native::empty_like(offsets);
    int64_t *size_ptr = new_sizes.data_ptr<int64_t>();
    int64_t *new_offsets_ptr = new_offsets.data_ptr<int64_t>();

    // Get start val for each split
    for (int64_t i : c10::irange(n_tensors)) {
      const int64_t index = i * tensor_dim + dim;
      new_offsets_ptr[i] = offsets_ptr[i] + start_val;
      size_ptr[index] = split_size;
    }
    start_val += split_size;
    splits[split_idx] = create_nested_view_tensor(
      self, std::move(new_sizes), std::move(new_strides), std::move(new_offsets));
  }
  return splits;
}

} // namespace native
} // namespace at
