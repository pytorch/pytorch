#include <torch/csrc/utils/out_types.h>

namespace torch {
namespace utils {

// Used by python binding codegen to ensure any TensorOptions arguments are consistent
// with the out tensor's options
void check_out_type_matches(const at::Tensor& result,
                            c10::optional<at::ScalarType> scalarType,
                            c10::optional<at::Layout> layout,
                            const at::Device& device, bool device_is_none) {
  if (!scalarType && !layout && device_is_none) {  // common case
    return;
  }
  if (scalarType && result.scalar_type() != *scalarType) {
    AT_ERROR(
        "dtype ", *scalarType,
        " does not match dtype of out parameter (", result.scalar_type(), ")");
  }
  if (scalarType && result.scalar_type() != *scalarType) {
    AT_ERROR(
        "scalar type ", *scalarType,
        " does not match scalar type of out parameter (", result.scalar_type(), ")");
  }
  if (layout && result.layout() != *layout) {
    AT_ERROR(
        "layout ", *layout,
        " does not match layout of out parameter (", result.layout(), ")");
  }
  auto device_type_arg = device_is_none ? result.device().type() : device.type();
  if (result.device().type() != device_type_arg) {
    AT_ERROR(
        "device type ", device_type_arg,
        " does not match device type of out parameter (", result.device().type(), ")");
  }
}

}}
