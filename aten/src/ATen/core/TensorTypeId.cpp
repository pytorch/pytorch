#include "ATen/core/TensorTypeId.h"

namespace at {

std::ostream& operator<<(std::ostream& str, at::TensorTypeId rhs) {
  return str << rhs.underlyingId();
}

} // namespace at
