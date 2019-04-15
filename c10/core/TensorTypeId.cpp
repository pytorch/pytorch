#include "c10/core/TensorTypeId.h"
#include "c10/util/string_utils.h"

namespace c10 {

std::ostream& operator<<(std::ostream& str, c10::TensorTypeId rhs) {
  return str << c10::to_string(rhs.underlyingId());
}

std::string toString(TensorTypeId id) {
  return c10::to_string(id.underlyingId());
}

} // namespace c10
