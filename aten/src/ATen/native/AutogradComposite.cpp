#include <ATen/ATen.h>
#include <ATen/native/ResizeCommon.h>

namespace at {
namespace native {

/// This function can be used to create a dual Tensor that holds a tangent to compute forward mode gradients.
/// Note that the dual Tensor's primal is a view of the given primal and the given tangent is used as-is.
/// This function is backward differentiable.
at::Tensor _make_dual(const at::Tensor& primal, const at::Tensor& tangent, int64_t level) {
  TORCH_CHECK(!primal._fw_grad(level).defined(), "Making a dual Tensor based on a Tensor that "
              "already has a forward gradient at the same level ", level, " is not supported.");

  auto dual_tensor = primal.view(primal.sizes());
  dual_tensor._set_fw_grad(tangent, level, /* is_inplace_op */ false);
  return dual_tensor;
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
    int64_t self_num_batch_dims) {
  // This implementation only applies to the batched grad case
  auto other_sizes = other.sizes();
  auto other_strides = other.strides();
  auto other_storage_offset = other.storage_offset();

  auto self_sizes = self.sizes();
  auto self_strides = self.strides();
  auto self_num_feature_dims = self.dim() - self_num_batch_dims;

  // NB: We don't check that the sizes of self is the same as that of other
  //     because this function is also used in the inplace over view case

  std::vector<int64_t> out_sizes;
  out_sizes.reserve(self.dim());
  out_sizes.insert(out_sizes.begin(), other_sizes.begin(), other_sizes.end());
  out_sizes.insert(out_sizes.begin(), self_sizes.begin(), self_sizes.end() - self_num_feature_dims);

  // We use the strides of other, and tack on the strides computed with
  // the batch dims of self, so that the slices are arranged contiguously
  std::vector<int64_t> out_strides;
  out_strides.reserve(self.dim());
  out_strides.insert(out_strides.begin(), other_strides.begin(), other_strides.end());

  int64_t prod = other.storage().nbytes() / other.itemsize();
  for (size_t i = 0; i < self_num_batch_dims; ++i) {
    out_strides.insert(out_strides.begin(), prod);
    prod *= self_strides[i];
  }

  int64_t storage_numel = prod;

  // Inherit the TensorOptions of the primal
  auto new_tensor = at::zeros({storage_numel}, other.options());
  return new_tensor.as_strided(out_sizes, out_strides, other_storage_offset);
}

} // namespace native

} // namespace at
