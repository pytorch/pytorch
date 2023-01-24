#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <c10/util/Optional.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_nested_tensor_size_native.h>
#include <ATen/ops/_nested_tensor_strides_native.h>
#include <ATen/ops/_nested_tensor_offsets_native.h>
#include <ATen/ops/chunk_native.h>
#endif

namespace at {
namespace native {

/**
 * Thin wrapper around get_nested_size_tensor that is registered as a native function
 *
 * @return The nested tensors' size tensor.
 */
at::Tensor _nested_tensor_size(const at::Tensor& self) {
  return get_nested_size_tensor(self);
}

at::Tensor _nested_tensor_strides(const at::Tensor& self){
  return  get_nested_tensor_impl(self) -> get_nested_stride_tensor();
}
std::vector<int64_t> _nested_tensor_offsets(const at::Tensor& self){
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
      nt.get_nested_size_tensor());
}

int64_t get_consistent_last_dim_of_nested_tensor(const NestedTensorImpl& nt) {
  c10::optional<int64_t> last_dim = nt.opt_size(-1);
  TORCH_CHECK(
      last_dim != c10::nullopt,
      "Expected all tensors in nested tensor to have the same trailing dimension, instead last dimension equals: ",
      nt.get_nested_size_tensor().select(1, -1));
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
  const auto& sizes = self_impl->get_nested_size_tensor();
  const auto& strides = self_impl->get_nested_stride_tensor();
  const std::vector<int64_t>& offsets = self_impl->get_storage_offsets();
  // Account for the implicit batch dim
  --dim;
  int64_t tensor_dim = sizes.size(1);
  for (const auto split_idx : c10::irange(chunks)) {
      auto new_sizes = sizes.clone() ;
      auto new_strides = strides.clone();
      // This copys offsets so we are safe to move
      auto new_offsets = std::vector<int64_t>(offsets);
      int64_t *size_ptr = new_sizes.data_ptr<int64_t>();
      // Get start val for each split
      int64_t start_val = split_idx * split_size;
      for (int64_t i : c10::irange(n_tensors)) {
        const int64_t index = i * tensor_dim + dim;
        new_offsets[i] = offsets[i] + start_val;
        size_ptr[index] = split_size;
    }
    splits[split_idx] = create_nested_view_tensor(self, new_sizes, new_strides, std::move(new_offsets));
  }
  return splits;
}

} // namespace native
} // namespace at
