#include <c10/core/TensorOptions.h>

namespace c10::impl {

inline std::optional<MemoryFormat>
check_tensor_options_and_extract_memory_format(
    const TensorOptions& options,
    std::optional<MemoryFormat> memory_format) {
  TORCH_CHECK(
      options.requires_grad_opt() == std::nullopt ||
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

} // namespace impl namespace c10
