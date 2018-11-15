#include "c10/util/TensorTypeId.h"
#include "c10/util/string_utils.h"

namespace at {

std::ostream& operator<<(std::ostream& str, at::TensorTypeId rhs) {
  return str << caffe2::to_string(rhs.underlyingId());
}

} // namespace at
