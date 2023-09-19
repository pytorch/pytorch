#include <ATen/core/TensorBase.h>
#include <algorithm>
#include <vector>
#include <ATen/core/TensorBody.h>
#include <c10/util/OptionalArrayRef.h>

#ifdef AT_PER_OPERATOR_HEADERS
#include <ATen/ops/_assert_tensor_metadata_native.h>
#endif

namespace at {

class Tensor;

namespace native {

template<typename O, typename C>
void _assert_match_sym(const O& original, const C& compared, const std::string& name) {
  if (compared) {
    auto sym_compared = c10::SymIntArrayRef(
      reinterpret_cast<const c10::SymInt*>(compared.value().data()), compared.value().size());
    bool equal = (original == sym_compared);
    if (!equal) {
      std::stringstream msg;
      msg << "Tensor " << name << " mismatch!";
      AT_ASSERT(equal, msg.str());
    }
  }
}

template<typename O, typename C>
void _assert_match(const O& original, const C& compared, const std::string& name) {
  if (compared) {
    bool equal = (original == compared.value());
    if (!equal) {
      std::stringstream msg;
      msg << "Tensor " << name << " mismatch!";
      AT_ASSERT(equal, msg.str());
    }
  }
}

void _assert_tensor_metadata(at::Tensor const& tensor, at::OptionalIntArrayRef sizes, at::OptionalIntArrayRef strides, c10::optional<c10::ScalarType> dtype) {
  _assert_match_sym(tensor.sym_sizes(), sizes, "sizes");
  _assert_match_sym(tensor.sym_strides(), strides, "strides");
  _assert_match(tensor.dtype(), dtype, "dtype");
}

Tensor _functional_assert_tensor_metadata(at::Tensor const& tensor, at::OptionalIntArrayRef sizes, at::OptionalIntArrayRef strides, c10::optional<c10::ScalarType> dtype) {
  _assert_tensor_metadata(tensor, sizes, strides, dtype);
  return tensor.clone();
}


}
}  // namespace at::native
