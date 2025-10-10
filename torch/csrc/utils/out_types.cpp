#include <torch/csrc/utils/out_types.h>

namespace torch::utils {

// Used by python binding codegen to ensure any TensorOptions arguments are
// consistent with the out tensor's options
void check_out_type_matches(
    const at::Tensor& result,
    std::optional<at::ScalarType> scalarType,
    bool scalarType_is_none,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    bool device_is_none) {
  if (scalarType_is_none && !layout && device_is_none) { // common case
    return;
  }
  if (!scalarType_is_none && result.scalar_type() != scalarType) {
    TORCH_CHECK(
        false,
        "dtype ",
        scalarType,
        " does not match dtype of out parameter (",
        result.scalar_type(),
        ")");
  }
  if (layout && result.layout() != *layout) {
    TORCH_CHECK(
        false,
        "layout ",
        *layout,
        " does not match layout of out parameter (",
        result.layout(),
        ")");
  }
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  if (!device_is_none && result.device().type() != device.value().type()) {
    TORCH_CHECK(
        false,
        "device type ",
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        device->type(),
        " does not match device type of out parameter (",
        result.device().type(),
        ")");
  }
}

} // namespace torch::utils
