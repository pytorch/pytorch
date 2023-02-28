#include <torch/csrc/utils/out_types.h>

namespace torch {
namespace utils {

// Used by python binding codegen to ensure any TensorOptions arguments are
// consistent with the out tensor's options
void check_out_type_matches(
    const at::Tensor& result,
    c10::optional<at::ScalarType> scalarType,
    bool scalarType_is_none,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    bool device_is_none) {
  if (scalarType_is_none && !layout && device_is_none) { // common case
    return;
  }
  if (!scalarType_is_none && result.scalar_type() != scalarType.value()) {
    AT_ERROR(
        "dtype ",
        *scalarType,
        " does not match dtype of out parameter (",
        result.scalar_type(),
        ")");
  }
  if (layout && result.layout() != *layout) {
    AT_ERROR(
        "layout ",
        *layout,
        " does not match layout of out parameter (",
        result.layout(),
        ")");
  }
  if (!device_is_none && result.device().type() != device.value().type()) {
    AT_ERROR(
        "device type ",
        device->type(),
        " does not match device type of out parameter (",
        result.device().type(),
        ")");
  }
}

} // namespace utils
} // namespace torch
