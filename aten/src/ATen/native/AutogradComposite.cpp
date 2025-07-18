#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <c10/util/SmallBuffer.h>
#include <c10/core/impl/COW.h>
#include <c10/core/DispatchKey.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_has_same_storage_numel_native.h>
#include <ATen/ops/_make_dual_native.h>
#include <ATen/ops/_new_zeros_with_same_feature_meta_native.h>
#include <ATen/ops/_unpack_dual_native.h>
#include <ATen/ops/_lazy_clone_native.h>
#include <ATen/ops/alias.h>
#include <ATen/ops/empty.h>
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

Tensor _lazy_clone(Tensor const& self, std::optional<c10::Device> device_opt) {
  c10::StorageImpl* self_storage = self.storage().unsafeGetStorageImpl();
  c10::intrusive_ptr<c10::StorageImpl> storage = nullptr;

  if (device_opt.has_value()) {
    c10::Device src_device = self.device();
    c10::Device dst_device = device_opt.value();
    c10::DeviceType src_device_type = src_device.type();
    c10::DeviceType dst_device_type = dst_device.type();
    TORCH_CHECK(
      (src_device_type == dst_device_type)
      || (src_device_type == c10::kCPU && dst_device_type == c10::kMPS)
      || (src_device_type == c10::kMPS && dst_device_type == c10::kCPU),
      "Lazy cloning is only supported in the following cases: ",
      "cpu to mps, ",
      "mps to cpu, ",
      "or between the same device. Got ", src_device, " to ", dst_device
    );

    if (src_device_type == dst_device_type) {
      c10::DeviceIndex src_index = at::empty({}, at::TensorOptions().device(src_device)).device().index();
      c10::DeviceIndex dst_index = at::empty({}, at::TensorOptions().device(dst_device)).device().index();

      // NOTE: This case might already work, but we do not have any tests for it
      // yet, so throw an error.
      TORCH_CHECK(
        src_index == dst_index,
        "Lazy cloning between the same device type is only supported if the ",
        "device indices are the same. Got ", src_device, " to ", dst_device
      );
    }

    c10::Allocator* allocator = nullptr;

    if (src_device_type == c10::kMPS && dst_device_type == c10::kCPU) {
      // For MPS-to-CPU, need the output to use the pinned MPS allocator, not
      // the regular CPU allocator.
      allocator = at::globalContext().getPinnedMemoryAllocator(c10::kMPS);
      TORCH_INTERNAL_ASSERT(allocator != nullptr);
    } else {
      allocator = at::empty({}, at::TensorOptions().device(dst_device)).storage().allocator();
      TORCH_INTERNAL_ASSERT(allocator != nullptr);
    }

    if (src_device_type == c10::kCPU && dst_device_type == c10::kMPS) {
      TORCH_CHECK(self.is_pinned(),
        "It is only possible to lazy clone a CPU tensor to MPS if the tensor ",
        "is pinned.");
    }
    storage = c10::impl::cow::lazy_clone_storage(*self_storage, device_opt.value(), *allocator);
  } else {
    storage = c10::impl::cow::lazy_clone_storage(*self_storage);
  }
  TORCH_CHECK(storage != nullptr);
  c10::DispatchKeySet key_set = self.key_set();
  // If the target device differs, then we must change the key set
  if (device_opt.has_value() && device_opt.value().type() != self.device().type()) {
    c10::BackendComponent old_backend = c10::toBackendComponent(self.device().type());
    c10::BackendComponent new_backend = c10::toBackendComponent(device_opt.value().type());
    key_set = key_set.remove_backend(old_backend) | c10::DispatchKeySet(new_backend);
  }
  auto tensor = c10::make_intrusive<c10::TensorImpl>(
      c10::Storage(std::move(storage)),
      key_set,
      self.dtype());
  tensor->set_sizes_and_strides(self.sym_sizes(),
                                self.sym_strides(),
                                self.sym_storage_offset());
  // When cloning from MPS to CPU, need to synchronize the source device before
  // this function returns the lazy cloned tensor. The MPS device may have
  // pending write operations on the source tensor's data, and we must force
  // them to finish before trying to read it from the CPU.
  if (device_opt.has_value() && device_opt.value() != self.device()) {
    if (self.device().type() == c10::kMPS) {
      at::detail::getMPSHooks().deviceSynchronize();
    }
  } else if (self.device().type() == c10::kMPS) {
    at::detail::getMPSHooks().deviceSynchronize();
  }
  return Tensor(std::move(tensor));
}

} // namespace at::native
