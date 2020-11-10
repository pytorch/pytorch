#include <cstddef>
#include <string>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

// TODO before land: move to a better place
void Assert(bool condition, std::string message) {
  if (!condition) {
    throw c10::AssertionError(message.c_str(), "");
  }
}

} // namespace native
} // namespace at
