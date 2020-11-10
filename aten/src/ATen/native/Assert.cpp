#include <cstddef>
#include <string>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

void Assert(bool condition, std::string message) {
  if (!condition) {
    throw c10::AssertionError(message.c_str(), "");
  }
}

void Assert(const Tensor& condition, std::string message) {
  if (!at::native::all(condition).item<bool>()) {
    throw c10::AssertionError(message.c_str(), "");
  }
}

} // namespace native
} // namespace at
