#pragma once

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
namespace impl {

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
} // namespace impl
} // namespace c10
