#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/MapAllocator.h>
#include <c10/util/SmallBuffer.h>
#include <c10/core/impl/COW.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_has_same_storage_numel_native.h>
#include <ATen/ops/_make_dual_native.h>
#include <ATen/ops/_new_zeros_with_same_feature_meta_native.h>
#include <ATen/ops/_unpack_dual_native.h>
#include <ATen/ops/_lazy_clone_native.h>
#include <ATen/ops/_lazy_clone_alias.h>
#include <ATen/ops/_lazy_clone_alias_native.h>
#include <ATen/ops/_lazy_clone_future.h>
#include <ATen/ops/_lazy_clone_future_native.h>
#include <ATen/ops/alias.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

// We expect this code to only be reached in inference mode and when all inputs are inference tensors
Tensor _make_dual(const Tensor& primal, const Tensor& tangent, int64_t level) {
  TORCH_INTERNAL_ASSERT(
      InferenceMode::is_enabled() && primal.is_inference() && tangent.is_inference(),
      "Expected this function to only be reached in inference mode and when all the "
      "inputs are inference tensors. You should NOT call this function directly as "
      "native::_make_dual. Please use the dispatcher, i.e., at::_make_dual. Please "
      "file an issue if you come across this error otherwise.");
  return at::alias(primal);
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
  auto other_sizes = other.sym_sizes();
  auto other_strides = other.sym_strides();
  auto other_storage_offset = other.storage_offset();
  auto other_storage_numel = other.storage().sym_nbytes() / other.itemsize();

  if (self_num_batch_dims == 0) {
    auto new_tensor = at::zeros_symint({other_storage_numel}, other.options());
    return new_tensor.as_strided_symint(other_sizes, other_strides, other_storage_offset);
  }

  auto self_sizes = self.sym_sizes();

  // NB: We don't check that the sizes of self is the same as that of other
  //     because this function is also used in the inplace over view case
  //     In the inplace over view case we cannot rely on self and other being
  //     the same size. So we will use the size of other, and simply tack on
  //     the batch dims from self. For example: If self.sizes: [B, 2, 3],
  //     and other.size: [6], we return [B, 6].
  //     Also see the test test_inplace_on_view_not_same_layout, for when we reach
  //     this case.
  constexpr int64_t kSmallBufferSizeHint = 8;

  auto out_sizes = c10::SmallVector<c10::SymInt, kSmallBufferSizeHint>(other.dim() + self_num_batch_dims);
  std::copy(self_sizes.begin(), self_sizes.begin() + self_num_batch_dims, out_sizes.begin());
  std::copy(other_sizes.begin(), other_sizes.end(), out_sizes.begin() + self_num_batch_dims);

  // We use the strides of other, and tack on the strides computed with
  // the batch dims of self, so that the slices are arranged contiguously
  auto out_strides = c10::SmallVector<c10::SymInt, kSmallBufferSizeHint>(other.dim() + self_num_batch_dims);
  auto prod = other_storage_numel;

  for (int64_t i = self_num_batch_dims - 1; i >= 0; --i) {
    out_strides[i] = prod;
    prod *= self_sizes[i];
  }
  std::copy(other_strides.begin(), other_strides.end(), out_strides.begin() + self_num_batch_dims);

  auto storage_numel = prod;

  // Inherit the TensorOptions of the primal
  auto new_tensor = at::zeros_symint({storage_numel}, other.options());
  return new_tensor.as_strided_symint(out_sizes, out_strides, other_storage_offset);
}

bool _has_same_storage_numel(const at::Tensor& base, const at::Tensor& other) {
  return base.storage().sym_nbytes() / base.itemsize() == other.storage().sym_nbytes() / other.itemsize();
}

static Tensor _lazy_clone_impl(Tensor const& self, bool future) {
  c10::StorageImpl* self_storage = self.storage().unsafeGetStorageImpl();

  // If data pointer is shared between processes, we cannot convert it to
  // a COW data pointer. So for future behavior, we just clone it, and for
  // simulated behavior, we view it and emit a warning.
  if (MapAllocator::fromDataPtr(self_storage->_data_ptr_no_checks())) {
    if (future) {
      return self.clone();
    } else {
      TORCH_WARN(
          "This operation creates a conditional view of a tensor that is ",
          "shared between multiple processes. This behavior is deprecated, ",
          "and in the future it will unconditionally create a clone instead.");
      return self.view_symint(self.sym_sizes());
    }
  }

  c10::intrusive_ptr<c10::StorageImpl> storage =
    c10::impl::cow::lazy_clone_storage(*self_storage, future);

  if (storage == nullptr) {
    if (future) {
      return self.clone();
    } else {
      // It's not easy to give more information about why the tensor cannot be
      // lazily cloned here. For instance, if it is a numpy-based tensor, there
      // is nothing currently attached to the tensor that says so.
      TORCH_WARN(
          "This operation creates a conditional view of a tensor that has a ",
          "non-standard data pointer. This behavior is deprecated, and in the ",
          "future it will unconditionally create a clone instead.");
      return self.view_symint(self.sym_sizes());
    }
  }
  auto tensor = self.view_symint(self.sym_sizes());
  tensor.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(storage));
  return tensor;
}

Tensor _lazy_clone_alias(Tensor const& self) {
  return _lazy_clone_impl(self, /*future=*/false);
}

Tensor _lazy_clone_future(Tensor const& self) {
  return _lazy_clone_impl(self, /*future=*/true);
}

Tensor _lazy_clone(Tensor const& self, bool _force_alias) {
  if (_force_alias) {
    return self._lazy_clone_alias();
  }

  if (c10::impl::cow::get_future_lazy_clone()) {
    return self._lazy_clone_future();
  } else {
    return self._lazy_clone_alias();
  }
}

} // namespace at::native
