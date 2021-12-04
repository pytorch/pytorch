#include <ATen/ATen.h>
#include <c10/util/SmallBuffer.h>

namespace at {
namespace native {

// _make_dual is actually implemented in VariableTypeManual.cpp, this is just a stub
// because codegen assumes that a function of this name exists in the native namespace
// TODO: check the behavior of this and _fw_primal in inference mode
Tensor _make_dual(const Tensor& primal, const Tensor& tangnet, int64_t level) {
  TORCH_INTERNAL_ASSERT(false, "This is just a stub.");
  return Tensor();
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
