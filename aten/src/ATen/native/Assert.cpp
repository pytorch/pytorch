#include <cstddef>
#include <string>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

void _assert(bool condition, std::string message) {
  if (!condition) {
    throw c10::AssertionError(message.c_str(), "");
  }
}

void _assert(const Tensor& condition, std::string message) {
  TORCH_CHECK(condition.dtype() == ScalarType::Bool,
    "The input to _assert must be a boolean.");
  if (!condition.is_nonzero()) {
    throw c10::AssertionError(message.c_str(), "");
  }
}

} // namespace native
} // namespace at
