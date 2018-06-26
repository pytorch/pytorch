#include "caffe2/core/dispatch/TensorTypeId.h"

namespace c10 {

std::ostream& operator<<(std::ostream& str, TensorTypeId rhs) {
  return str << rhs.underlyingId();
}

}  // namespace c10
