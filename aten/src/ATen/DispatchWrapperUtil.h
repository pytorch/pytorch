#pragma once

#include <ATen/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/List.h>
#include <c10/core/TensorOptions.h>

/*
 * [Note: hacky wrapper removal for optional tensor]
 *
 * The kernel implementation takes an optional tensor marked in the schema as
 * Tensor? but the C++ function takes Tensor instead of the optional<Tensor>
 * expected by the dispatcher.
 *
 * To remove the hacky wrapper, the C++ function is changed to take
 * optional<Tensor> and unwrap the Tensor value at the beginning of
 * the function, e.g.:
 *   > const Tensor& weight =
     >     c10::value_or_else(weight_opt, [] {returnTensor();});
 *
 * We may want make the kernel handle optional directly without going through
 * the creation of a default constructed tensor.
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

namespace c10 {
namespace detail {

inline c10::optional<MemoryFormat>
check_tensor_options_and_extract_memory_format(
    const TensorOptions& options,
    c10::optional<MemoryFormat> memory_format) {
  TORCH_CHECK(
      options.requires_grad_opt() == c10::nullopt ||
          options.requires_grad_opt().value() == false,
      "Operators taking TensorOptions cannot take a TensorOptions with "
      "options.requires_grad set as true. This isn't implemented yet.");
  TORCH_CHECK(
      !(options.has_memory_format() && memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");
  if (memory_format.has_value()) {
    return memory_format;
  } else {
    return options.memory_format_opt();
  }
}

inline void common_device_check_failure(optional<Device>& common_device, const at::Tensor& tensor, at::CheckedFrom methodName, at::CheckedFrom argName) {
  TORCH_CHECK(false,
    "Expected all tensors to be on the same device, but "
    "found at least two devices, ", common_device.value(), " and ", tensor.device(), "! "
    "(when checking arugment for argument ", argName, " in method ", methodName, ")");
}

inline void undefined_device_check_failure(at::CheckedFrom methodName, at::CheckedFrom argName) {
  TORCH_CHECK(false,
    "Tensor is undefined."
    "(when checking arugment for argument ", argName, " in method ", methodName, ")");
}

inline void check_or_update_common_device(optional<Device>& common_device, const at::Tensor& tensor, at::CheckedFrom methodName, at::CheckedFrom argName) {
  if (C10_UNLIKELY(!tensor.defined())) {
    // TODO: Don't allow undefined tensor input
    return;
  }

  if (!common_device.has_value()) {
    common_device = tensor.device();
    return;
  }

  if (C10_UNLIKELY(common_device != tensor.device())) {
    common_device_check_failure(common_device, tensor, methodName, argName);
  }
}

inline void check_or_update_common_device(optional<Device>& common_device, const optional<at::Tensor>& tensor, at::CheckedFrom methodName, at::CheckedFrom argName) {
  if (tensor.has_value()) {
    check_or_update_common_device(common_device, tensor.value(), methodName, argName);
  }
}

inline void check_or_update_common_device(optional<Device>& common_device, at::TensorList tensors, at::CheckedFrom methodName, at::CheckedFrom argName) {
  for (const auto& tensor : tensors) {
    check_or_update_common_device(common_device, tensor, methodName, argName);
  }
}

inline void check_or_update_common_device(optional<Device>& common_device, const List<optional<at::Tensor>>& tensors, at::CheckedFrom methodName, at::CheckedFrom argName) {
  for (const auto& tensor : tensors) {
    check_or_update_common_device(common_device, tensor, methodName, argName);
  }
}
} // namespace detail
} // namespace c10
