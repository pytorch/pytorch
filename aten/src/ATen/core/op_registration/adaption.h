#pragma once

#include <ATen/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/List.h>
#include <c10/core/TensorOptions.h>

/*
 * [Note: hacky wrapper removal for optional tensor]
 *
 * The kernel implementation takes an optional tensor marked in the schema as
 * Tensor? but the C++ function takes Tensor instead of the std::optional<Tensor>
 * expected by the dispatcher.
 *
 * To remove the hacky wrapper, the C++ function is changed to take
 * std::optional<Tensor> and unwrap the Tensor value at the beginning of
 * the function, e.g.:
 *   > c10::MaybeOwned<Tensor> weight_maybe_owned =
 *   >     at::borrow_from_optional_tensor(weight_opt);
 *   > const Tensor& weight = *weight_maybe_owned;
 *
 * We may want to make the kernel handle optional directly without
 * going through the creation of a default-constructed Tensor in
 * at::borrow_from_optional_tensor.
 */

/*
 * [Note: hacky wrapper removal for TensorOptions]
 *
 * The kernel implementation takes a TensorOptions argument but the dispatcher
 * expects separate arguments for dtype, layout, device, pin_memory.
 *
 * To remove the hacky wrapper, the kernel implementation is changed to take
 * the 4 arguments (dtype, layout, device, pin_memory), and assemble the
 * TensorOptions value at the beginning of the function, e.g.:
 *   > TensorOptions options = TensorOptions().dtype(dtype).layout(layout)
 *   >    .device(device).pinned_memory(pin_memory);
 *
 * We may want make the kernel handle these parameters directly without going
 * through the creation of a TensorOptions value.
 */

namespace c10::impl {

TORCH_API void common_device_check_failure(Device common_device, const at::Tensor& tensor, at::CheckedFrom methodName, at::CheckedFrom argName);

inline void check_and_update_common_device(std::optional<Device>& common_device, const at::Tensor& tensor, at::CheckedFrom methodName, at::CheckedFrom argName) {
  // TODO: Remove this once the following issue is addressed:
  // https://github.com/pytorch/pytorch/issues/57380
  if (!tensor.defined()) {
    return;
  }

  if (!common_device.has_value()) {
    common_device = tensor.device();
    return;
  }

  if (C10_UNLIKELY(common_device != tensor.device())) {
    common_device_check_failure(*common_device, tensor, methodName, argName);
  }
}

inline void check_and_update_common_device(std::optional<Device>& common_device, const std::optional<at::Tensor>& tensor, at::CheckedFrom methodName, at::CheckedFrom argName) {
  if (tensor.has_value()) {
    check_and_update_common_device(common_device, tensor.value(), methodName, argName);
  }
}

inline void check_and_update_common_device(std::optional<Device>& common_device, at::ITensorListRef tensors, at::CheckedFrom methodName, at::CheckedFrom argName) {
  for (const auto& tensor : tensors) {
    check_and_update_common_device(common_device, tensor, methodName, argName);
  }
}

inline void check_and_update_common_device(std::optional<Device>& common_device, const List<std::optional<at::Tensor>>& tensors, at::CheckedFrom methodName, at::CheckedFrom argName) {
  for (const auto& tensor : tensors) {
    check_and_update_common_device(common_device, tensor, methodName, argName);
  }
}
} // namespace c10::impl
