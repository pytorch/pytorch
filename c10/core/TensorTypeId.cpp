#include <c10/core/TensorTypeId.h>
#include <c10/util/C++17.h>

namespace c10 {

std::ostream& operator<<(std::ostream& str, at::TensorTypeId rhs) {
  return str << c10::guts::to_string(rhs.underlyingId());
}

} // namespace c10
