#include <ATen/ATen.h>

namespace at {
namespace native {

// _make_dual is actually implemented in VariableTypeManual.cpp, this is just a stub
// because codegen assumes that a function of this name exists in the native namespace
// TODO: check the behavior of this and _fw_primal in inference mode
Tensor _make_dual(const Tensor& primal, const Tensor& tangnet, int64_t level) {
  AT_ERROR("This is just a stub.");
}

/// This function can be used to unpack a given dual Tensor to get its primal and tangent. The returned primal
/// is a view of the dual and the tangent is returned as is.
/// This function is backward differentiable.
std::tuple<at::Tensor, at::Tensor> _unpack_dual(const at::Tensor& tensor, int64_t level) {
  return std::tuple<at::Tensor, at::Tensor>(tensor._fw_primal(level), tensor._fw_grad(level));
}

Tensor _new_zeros_with_same_feature_meta(
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<int64_t> self_num_batch_dims) {
  // NB: This function can be called directly from _set_fw_grad or
  //     if self is batched, from this function's batching rule.
  auto other_sizes = other.sizes();
  auto other_strides = other.strides();
  auto other_storage_offset = other.storage_offset();
  int64_t other_storage_numel = other.storage().nbytes() / other.itemsize();

  if (!self_num_batch_dims.has_value()) {
    auto new_tensor = at::zeros({other_storage_numel}, other.options());
    return new_tensor.as_strided(other_sizes, other_strides, other_storage_offset);
  }

  int64_t _self_num_batch_dims = self_num_batch_dims.value();

  auto self_sizes = self.sizes();
  auto self_strides = self.strides();

  // NB: We don't check that the sizes of self is the same as that of other
  //     because this function is also used in the inplace over view case

  std::vector<int64_t> out_sizes;
  out_sizes.reserve(other.dim() + _self_num_batch_dims);
  out_sizes.insert(out_sizes.end(), self_sizes.begin(), self_sizes.begin() + _self_num_batch_dims);
  out_sizes.insert(out_sizes.end(), other_sizes.begin(), other_sizes.end());

  // We use the strides of other, and tack on the strides computed with
  // the batch dims of self, so that the slices are arranged contiguously
  std::vector<int64_t> out_strides;
  out_strides.reserve(other.dim() + _self_num_batch_dims);

  int64_t prod = other_storage_numel;
  for (size_t i = 0; i < _self_num_batch_dims; ++i) {
    out_strides.insert(out_strides.begin(), prod);
    prod *= self_strides[i];
  }
  out_strides.insert(out_strides.end(), other_strides.begin(), other_strides.end());

  int64_t storage_numel = prod;

  // Inherit the TensorOptions of the primal
  auto new_tensor = at::zeros({storage_numel}, other.options());
  return new_tensor.as_strided(out_sizes, out_strides, other_storage_offset);
}

} // namespace native

} // namespace at
