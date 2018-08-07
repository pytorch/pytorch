#include "ATen/core/TensorTypeId.h"

std::ostream& operator<<(std::ostream& str, at::TensorTypeId rhs) {
  return str << rhs.underlyingId();
}
