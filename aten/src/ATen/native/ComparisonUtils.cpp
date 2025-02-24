#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/util/OptionalArrayRef.h>

#ifdef AT_PER_OPERATOR_HEADERS
#include <ATen/ops/_assert_tensor_metadata_native.h>
#endif

namespace at {

class Tensor;

namespace native {

template<typename O, typename C>
void _assert_match(const O& original, const C& compared, const std::string& name) {
  if (compared) {
    bool equal = (original == compared.value());
    if (!equal) {
      std::stringstream msg;
      msg << "Tensor " << name << " mismatch!";
      if (!equal) {
        throw std::runtime_error(msg.str());
      }
    }
  }
}

void _assert_tensor_metadata_meta_symint(at::Tensor const& tensor, at::OptionalSymIntArrayRef sizes, at::OptionalSymIntArrayRef strides, std::optional<c10::ScalarType> dtype, std::optional<c10::Device> device, std::optional<c10::Layout> layout) {
  _assert_match(tensor.sym_sizes(), sizes, "sizes");
  _assert_match(tensor.sym_strides(), strides, "strides");
  _assert_match(tensor.dtype(), dtype, "dtype");
  _assert_match(tensor.device(), device, "device");
  _assert_match(tensor.layout(), layout, "layout");
}

void _assert_tensor_metadata(at::Tensor const& tensor, at::OptionalIntArrayRef sizes, at::OptionalIntArrayRef strides, std::optional<c10::ScalarType> dtype, std::optional<c10::Device> device, std::optional<c10::Layout> layout) {
  _assert_match(tensor.sizes(), sizes, "sizes");
  _assert_match(tensor.strides(), strides, "strides");
  _assert_match(tensor.dtype(), dtype, "dtype");
  _assert_match(tensor.device(), device, "device");
  _assert_match(tensor.layout(), layout, "layout");
}

}
}  // namespace at::native
