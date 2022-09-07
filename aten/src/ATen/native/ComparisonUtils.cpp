#include <ATen/core/TensorBase.h>
#include <algorithm>
#include <vector>
#include <ATen/core/TensorBody.h>

namespace at {

class Tensor;

namespace native {

void _assert_true(bool value, c10::basic_string_view<char> message) {
  AT_ASSERT(value, message);
}

bool _tensor_equal(at::Tensor const& tensor, c10::OptionalArrayRef<long> sizes, c10::OptionalArrayRef<long> strides, c10::optional<c10::ScalarType> dtype) {
  if (sizes) {
    if (tensor.sizes() != sizes.value()) {
      return false;
    }
  }
  if (strides) {
    if (tensor.strides() != strides.value()) {
      return false;
    }
  }
  if (dtype) {
    if (tensor.dtype() != dtype.value()) {
      return false;
    }
  }
  return true;
}

}
}  // namespace at::native
