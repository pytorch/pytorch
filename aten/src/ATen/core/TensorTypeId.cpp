#include "ATen/core/TensorTypeId.h"
#include "caffe2/utils/string_utils.h"

namespace at {

std::ostream& operator<<(std::ostream& str, at::TensorTypeId rhs) {
  return str << caffe2::to_string(rhs.underlyingId());
}

} // namespace at
