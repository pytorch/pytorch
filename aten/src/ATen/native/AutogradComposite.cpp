#include <ATen/ATen.h>
#include <c10/util/SmallBuffer.h>

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

// NB: This function can be called directly from _set_fw_grad or
//     if self is batched, from this function's batching rule.
//     See NOTE: [_new_zeros_with_same_feature_meta] for more information.
Tensor _new_zeros_with_same_feature_meta(
    const at::Tensor& self,
    const at::Tensor& other,
    int64_t self_num_batch_dims) {
  auto other_sizes = other.sizes();
  auto other_strides = other.strides();
  auto other_storage_offset = other.storage_offset();
  int64_t other_storage_numel = other.storage().nbytes() / other.itemsize();

  if (self_num_batch_dims == 0) {
    auto new_tensor = at::zeros({other_storage_numel}, other.options());
    return new_tensor.as_strided(other_sizes, other_strides, other_storage_offset);
  }

  auto self_sizes = self.sizes();
  auto self_strides = self.strides();

  // NB: We don't check that the sizes of self is the same as that of other
  //     because this function is also used in the inplace over view case
  //     In the inplace over view case we cannot rely on self and other being
  //     the same size. So we will use the size of other, and simply tack on
  //     the batch dims from self. For example: If self.sizes: [B, 2, 3],
  //     and other.size: [6], we return [B, 6].
  //     Also see the test test_inplace_on_view_not_same_layout, for when we reach
  //     this case.
  constexpr int64_t kSmallBufferSizeHint = 8;

  auto out_sizes = c10::SmallBuffer<int64_t, kSmallBufferSizeHint>(other.dim() + self_num_batch_dims);
  std::copy(self_sizes.begin(), self_sizes.begin() + self_num_batch_dims, out_sizes.begin());
  std::copy(other_sizes.begin(), other_sizes.end(), out_sizes.begin() + self_num_batch_dims);

  // We use the strides of other, and tack on the strides computed with
  // the batch dims of self, so that the slices are arranged contiguously
  auto out_strides = c10::SmallBuffer<int64_t, kSmallBufferSizeHint>(other.dim() + self_num_batch_dims);
  int64_t prod = other_storage_numel;

  for (int64_t i = self_num_batch_dims - 1; i >= 0; --i) {
    out_strides[i] = prod;
    prod *= self_sizes[i];
  }
  std::copy(other_strides.begin(), other_strides.end(), out_strides.begin() + self_num_batch_dims);

  int64_t storage_numel = prod;

  // Inherit the TensorOptions of the primal
  auto new_tensor = at::zeros({storage_numel}, other.options());
  return new_tensor.as_strided(out_sizes, out_strides, other_storage_offset);
}

} // namespace native

} // namespace at
